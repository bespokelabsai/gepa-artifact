"""
Training script for prompt router with (task, candidate, reward) format.

Trains a cross-encoder model to predict the reward for each (task, candidate) pair.
"""

import sys
import os
from pathlib import Path
import json
import pickle
import argparse
from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

# Add gepa_artifact to path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from router.model import PromptRouterCrossEncoder


class PromptRewardDataset(Dataset):
    """
    Dataset for training prompt router with explicit rewards.

    Each example is a (task, candidate, reward) tuple with the prompt included.
    """

    def __init__(
        self,
        training_data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            training_data: List of dicts with keys: claim, candidate_idx, candidate_prompt, reward
            tokenizer: Tokenizer for encoding
            max_length: Max sequence length
        """
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single (task, candidate, reward) example.
        """
        example = self.training_data[idx]

        return {
            'claim': example['claim'],
            'candidate_idx': example['candidate_idx'],
            'candidate_prompt': example['candidate_prompt'],
            'reward': example['reward'],
            'task_idx': example['task_idx'],
        }


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, max_length: int):
    """Collate batch of samples into tensors."""
    claims = []
    prompts = []
    rewards = []
    candidate_indices = []

    for sample in batch:
        claim = sample['claim']
        candidate_idx = sample['candidate_idx']
        candidate_prompt = sample['candidate_prompt']
        reward = sample['reward']

        claims.append(claim)
        prompts.append(candidate_prompt)
        rewards.append(reward)
        candidate_indices.append(candidate_idx)

    if not claims:
        # Return empty batch
        return {
            'input_ids': torch.empty(0, max_length, dtype=torch.long),
            'attention_mask': torch.empty(0, max_length, dtype=torch.long),
            'rewards': torch.empty(0, dtype=torch.float),
        }

    # Tokenize all pairs
    encoded = tokenizer(
        claims,
        prompts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded.get('token_type_ids', None),
        'rewards': torch.tensor(rewards, dtype=torch.float),
        'candidate_indices': torch.tensor(candidate_indices, dtype=torch.long),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    loss_type: str = "bce"
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if batch['input_ids'].size(0) == 0:
            continue

        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if batch['token_type_ids'] is not None else None
        rewards = batch['rewards'].to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(input_ids, attention_mask, token_type_ids)

        # Compute loss
        if loss_type == "bce":
            # Binary cross-entropy (treat as classification)
            loss = nn.functional.binary_cross_entropy_with_logits(scores, rewards)
        elif loss_type == "mse":
            # Mean squared error (treat as regression)
            scores = torch.sigmoid(scores)  # Scale to [0, 1]
            loss = nn.functional.mse_loss(scores, rewards)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_data: List[Dict],
    system_prompts: Dict[int, str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation set.

    For each task, score all candidates and pick the best one.
    Compare with ground truth rewards.
    """
    model.eval()

    # Group data by task
    tasks = {}
    for ex in val_data:
        task_idx = ex['task_idx']
        if task_idx not in tasks:
            tasks[task_idx] = []
        tasks[task_idx].append(ex)

    # Metrics
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_tasks = 0

    # Reward prediction metrics
    all_pred_scores = []
    all_true_rewards = []

    for task_idx, task_examples in tqdm(tasks.items(), desc="Evaluating"):
        # Get claim (same for all examples of this task)
        claim = task_examples[0]['claim']

        # Score all candidates for this task
        candidate_scores = {}
        true_rewards = {}

        for ex in task_examples:
            candidate_idx = ex['candidate_idx']
            true_rewards[candidate_idx] = ex['reward']

            if candidate_idx not in system_prompts:
                continue

            # Tokenize
            encoded = tokenizer(
                [claim],
                [system_prompts[candidate_idx]],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            # Score
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            token_type_ids = encoded.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            score = model(input_ids, attention_mask, token_type_ids)
            score = torch.sigmoid(score).cpu().item()

            candidate_scores[candidate_idx] = score
            all_pred_scores.append(score)
            all_true_rewards.append(true_rewards[candidate_idx])

        # Get top predictions
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        # Find best candidate according to ground truth
        best_true_candidates = [c for c, r in true_rewards.items() if r == max(true_rewards.values())]

        if not sorted_candidates:
            continue

        # Check if predicted best is in true best
        pred_best = sorted_candidates[0][0]
        pred_top3 = [c for c, _ in sorted_candidates[:3]]
        pred_top5 = [c for c, _ in sorted_candidates[:5]]

        if pred_best in best_true_candidates:
            correct_top1 += 1

        if any(c in best_true_candidates for c in pred_top3):
            correct_top3 += 1

        if any(c in best_true_candidates for c in pred_top5):
            correct_top5 += 1

        total_tasks += 1

    # Compute accuracies
    top1_acc = correct_top1 / total_tasks if total_tasks > 0 else 0
    top3_acc = correct_top3 / total_tasks if total_tasks > 0 else 0
    top5_acc = correct_top5 / total_tasks if total_tasks > 0 else 0

    # Compute reward prediction metrics
    mse = np.mean([(p - t)**2 for p, t in zip(all_pred_scores, all_true_rewards)])
    mae = np.mean([abs(p - t) for p, t in zip(all_pred_scores, all_true_rewards)])

    metrics = {
        'top_1_accuracy': top1_acc,
        'top_3_accuracy': top3_acc,
        'top_5_accuracy': top5_acc,
        'total_tasks': total_tasks,
        'reward_mse': mse,
        'reward_mae': mae,
    }

    return top1_acc, metrics


def main():
    parser = argparse.ArgumentParser(description="Train prompt router with reward prediction")

    # Data arguments
    parser.add_argument(
        '--training_data',
        type=str,
        default='router/training_data_all.json',
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--prog_candidates_dir',
        type=str,
        default='experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates',
        help='Directory with program candidates (optional, only for reference)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )

    # Model arguments
    parser.add_argument(
        '--backbone',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Backbone model name'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )

    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio'
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        choices=['bce', 'mse'],
        default='bce',
        help='Loss function: bce (binary cross-entropy) or mse (mean squared error)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='router/checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=2,
        help='Save checkpoint every N epochs'
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading training data from {args.training_data}...")
    with open(args.training_data) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} examples")

    # Check if prompts are included in the data
    if 'candidate_prompt' not in all_data[0]:
        print("\nERROR: Training data doesn't include candidate_prompt field!")
        print("Please regenerate the dataset with:")
        print("  python router/construct_training_data.py --strategy all_candidates")
        sys.exit(1)

    # Extract unique prompts for later use (e.g., for evaluation)
    system_prompts = {}
    for ex in all_data:
        if ex['candidate_idx'] not in system_prompts:
            system_prompts[ex['candidate_idx']] = ex['candidate_prompt']

    print(f"Found {len(system_prompts)} unique system prompt candidates in data")

    # Split by task (not by example) to avoid leakage
    # Group by task first
    tasks = {}
    for ex in all_data:
        task_idx = ex['task_idx']
        if task_idx not in tasks:
            tasks[task_idx] = []
        tasks[task_idx].append(ex)

    task_indices = list(tasks.keys())
    random.shuffle(task_indices)

    val_size = int(len(task_indices) * args.val_split)
    val_task_indices = set(task_indices[:val_size])
    train_task_indices = set(task_indices[val_size:])

    # Split examples based on task assignment
    train_data = [ex for ex in all_data if ex['task_idx'] in train_task_indices]
    val_data = [ex for ex in all_data if ex['task_idx'] in val_task_indices]

    print(f"\nSplit: {len(train_data)} train examples ({len(train_task_indices)} tasks), "
          f"{len(val_data)} val examples ({len(val_task_indices)} tasks)")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # Create datasets
    train_dataset = PromptRewardDataset(
        train_data,
        tokenizer,
        max_length=args.max_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
        num_workers=0
    )

    # Create model
    print(f"\nInitializing model with backbone: {args.backbone}")
    model = PromptRouterCrossEncoder(
        backbone_name=args.backbone,
        dropout=args.dropout
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"\nTraining for {args.num_epochs} epochs ({num_training_steps} steps)")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Loss type: {args.loss_type}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training args
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_acc = 0
    training_history = []

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.loss_type
        )
        print(f"\nTrain loss: {train_loss:.4f}")

        # Evaluate
        print("\nEvaluating on validation set...")
        val_acc, val_metrics = evaluate(
            model,
            val_data,
            system_prompts,
            tokenizer,
            device,
            max_length=args.max_length
        )

        print(f"\nValidation Results:")
        print(f"  Top-1 Accuracy: {val_metrics['top_1_accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {val_metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {val_metrics['top_5_accuracy']:.4f}")
        print(f"  Reward MSE: {val_metrics['reward_mse']:.4f}")
        print(f"  Reward MAE: {val_metrics['reward_mae']:.4f}")
        print(f"  Total tasks: {val_metrics['total_tasks']}")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics
        })

        # Save checkpoint
        if epoch % args.save_every == 0 or val_acc > best_val_acc:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = output_dir / 'best_checkpoint.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'args': vars(args)
                }, best_checkpoint_path)
                print(f"New best validation accuracy: {val_acc:.4f}")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
