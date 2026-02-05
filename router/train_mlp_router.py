"""
Training script for MLP-based prompt router.

This script trains a cross-encoder model to predict the reward/compatibility
for (task, candidate) pairs.
"""

import sys
import os
from pathlib import Path
import json
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


class RouterDataset(Dataset):
    """
    Dataset for training MLP router.

    New format: Each training example contains a task with all its candidates.
    During training, we expand this into individual (task, candidate) pairs.
    """

    # Mapping of benchmark names to their input field names
    BENCHMARK_INPUT_FIELDS = {
        'hoverBench': 'claim',
        'HotpotQABench': 'question',
        'IFBench': 'prompt',
        'AIMEBench': 'problem',
        'Papillon': 'user_query',
    }

    def __init__(
        self,
        training_data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        benchmark_name: str = None,
    ):
        """
        Args:
            training_data: List of task dicts, each containing:
                - task_idx: int
                - task input field (e.g., 'claim', 'question')
                - candidates: List of dicts with candidate_idx, candidate_system_prompt, reward
            tokenizer: Tokenizer for encoding
            max_length: Max sequence length
            benchmark_name: Name of the benchmark (auto-detected if not provided)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Auto-detect input field
        if benchmark_name is None:
            self.input_field = self._detect_input_field(training_data[0])
        else:
            self.input_field = self.BENCHMARK_INPUT_FIELDS.get(benchmark_name, 'claim')

        print(f"Dataset using input field: '{self.input_field}'")

        # Expand the data into (task, candidate, reward) pairs
        self.pairs = []
        for task in training_data:
            task_input = task.get(self.input_field, '')
            task_idx = task['task_idx']

            for candidate in task['candidates']:
                self.pairs.append({
                    'task_idx': task_idx,
                    'task_input': task_input,
                    'candidate_idx': candidate['candidate_idx'],
                    'candidate_prompt': candidate['candidate_system_prompt'],
                    'reward': candidate['reward'],
                })

        print(f"Expanded {len(training_data)} tasks into {len(self.pairs)} (task, candidate) pairs")

    def _detect_input_field(self, example: Dict) -> str:
        """Auto-detect the input field from available fields."""
        for benchmark, field in self.BENCHMARK_INPUT_FIELDS.items():
            if field in example:
                return field
        return 'claim'

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """Returns a single (task, candidate, reward) pair."""
        return self.pairs[idx]


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, max_length: int):
    """Collate batch of (task, candidate, reward) samples."""
    task_inputs = [sample['task_input'] for sample in batch]
    prompts = [sample['candidate_prompt'] for sample in batch]
    rewards = [sample['reward'] for sample in batch]
    candidate_indices = [sample['candidate_idx'] for sample in batch]

    # Tokenize all pairs
    encoded = tokenizer(
        task_inputs,
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
            loss = nn.functional.binary_cross_entropy_with_logits(scores, rewards)
        elif loss_type == "mse":
            scores = torch.sigmoid(scores)
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
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
    input_field: str = 'claim'
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation set.

    For each task, score all candidates and pick the best one.
    """
    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_tasks = 0

    all_pred_scores = []
    all_true_rewards = []

    for task in tqdm(val_data, desc="Evaluating"):
        task_input = task.get(input_field, '')
        candidates = task['candidates']

        if not candidates:
            continue

        # Score all candidates
        candidate_scores = {}
        true_rewards = {}

        # Process candidates in batches
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]

            task_inputs = [task_input] * len(batch_candidates)
            prompts = [c['candidate_system_prompt'] for c in batch_candidates]

            # Tokenize
            encoded = tokenizer(
                task_inputs,
                prompts,
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

            batch_scores = model(input_ids, attention_mask, token_type_ids)
            batch_scores = torch.sigmoid(batch_scores).cpu().numpy()

            for candidate, score in zip(batch_candidates, batch_scores):
                candidate_idx = candidate['candidate_idx']
                candidate_scores[candidate_idx] = float(score)
                true_rewards[candidate_idx] = candidate['reward']

        # Track all predictions
        for idx in candidate_scores:
            all_pred_scores.append(candidate_scores[idx])
            all_true_rewards.append(true_rewards[idx])

        # Get top predictions
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Find best candidate(s) according to ground truth
        best_reward = max(true_rewards.values())
        best_candidates = [idx for idx, r in true_rewards.items() if r == best_reward]

        # Check top-k accuracy
        pred_top1 = [sorted_candidates[0][0]] if sorted_candidates else []
        pred_top3 = [c[0] for c in sorted_candidates[:3]]
        pred_top5 = [c[0] for c in sorted_candidates[:5]]

        if any(c in best_candidates for c in pred_top1):
            correct_top1 += 1
        if any(c in best_candidates for c in pred_top3):
            correct_top3 += 1
        if any(c in best_candidates for c in pred_top5):
            correct_top5 += 1

        total_tasks += 1

    # Compute metrics
    top1_acc = correct_top1 / total_tasks if total_tasks > 0 else 0
    top3_acc = correct_top3 / total_tasks if total_tasks > 0 else 0
    top5_acc = correct_top5 / total_tasks if total_tasks > 0 else 0

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
    parser = argparse.ArgumentParser(description="Train MLP prompt router")

    # Data arguments
    parser.add_argument(
        '--training_data',
        type=str,
        required=True,
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default=None,
        help='Benchmark name (auto-detected if not specified)'
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
        help='Loss function'
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
    print(f"Loaded {len(all_data)} tasks")

    # Auto-detect benchmark if needed
    if args.benchmark_name is None:
        # Use the dataset's detection logic
        benchmark_name = RouterDataset._detect_input_field(None, all_data[0])
        input_field = benchmark_name
    else:
        input_field = RouterDataset.BENCHMARK_INPUT_FIELDS.get(
            args.benchmark_name, 'claim'
        )

    # Split by task to avoid leakage
    task_indices = list(range(len(all_data)))
    random.shuffle(task_indices)

    val_size = int(len(task_indices) * args.val_split)
    val_task_indices = set(task_indices[:val_size])
    train_task_indices = set(task_indices[val_size:])

    train_data = [all_data[i] for i in train_task_indices]
    val_data = [all_data[i] for i in val_task_indices]

    print(f"\nSplit: {len(train_data)} train tasks, {len(val_data)} val tasks")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # Create datasets
    train_dataset = RouterDataset(
        train_data,
        tokenizer,
        max_length=args.max_length,
        benchmark_name=args.benchmark_name
    )

    # Create dataloader
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
            model, val_data, tokenizer, device,
            max_length=args.max_length,
            input_field=input_field
        )

        print(f"\nValidation Results:")
        print(f"  Top-1 Accuracy: {val_metrics['top_1_accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {val_metrics['top_3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {val_metrics['top_5_accuracy']:.4f}")
        print(f"  Reward MSE: {val_metrics['reward_mse']:.4f}")
        print(f"  Reward MAE: {val_metrics['reward_mae']:.4f}")

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
