"""
Training script for prompt router.

Trains a cross-encoder model to predict which system prompt candidate
works best for a given task.
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


class PromptRouterDataset(Dataset):
    """
    Dataset for training prompt router.

    For each task, creates positive and negative (query, prompt) pairs.
    """

    def __init__(
        self,
        training_data: List[Dict],
        system_prompts: Dict[int, str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        num_negatives: int = 3,
        mode: str = "train"
    ):
        self.training_data = training_data
        self.system_prompts = system_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.mode = mode

        # Pre-process all candidates
        self.all_candidates = sorted(system_prompts.keys())

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a batch of (query, prompt) pairs for one task.

        Includes:
        - 1 positive pair (best candidate)
        - N negative pairs (other candidates from pareto frontier or random)
        """
        task = self.training_data[idx]
        query = task['claim']
        best_candidate = task['selected_candidate']
        pareto_frontier = task['pareto_frontier']

        # Collect pairs
        pairs = []
        labels = []

        # Positive pair (best candidate)
        if best_candidate in self.system_prompts:
            pairs.append((query, self.system_prompts[best_candidate]))
            labels.append(1)

        # Negative pairs
        if self.mode == "train":
            # Sample negatives from pareto frontier (excluding best)
            negative_candidates = [c for c in pareto_frontier if c != best_candidate]

            # Also add some random negatives not in pareto frontier
            non_pareto = [c for c in self.all_candidates if c not in pareto_frontier]

            # Mix pareto and non-pareto negatives
            num_from_pareto = min(self.num_negatives // 2, len(negative_candidates))
            num_from_non_pareto = self.num_negatives - num_from_pareto

            selected_negatives = []
            if negative_candidates:
                selected_negatives.extend(random.sample(
                    negative_candidates,
                    min(num_from_pareto, len(negative_candidates))
                ))
            if non_pareto:
                selected_negatives.extend(random.sample(
                    non_pareto,
                    min(num_from_non_pareto, len(non_pareto))
                ))

            for neg_cand in selected_negatives:
                if neg_cand in self.system_prompts:
                    pairs.append((query, self.system_prompts[neg_cand]))
                    labels.append(0)

        return {
            'pairs': pairs,
            'labels': labels,
            'task_idx': task['task_idx'],
            'best_candidate': best_candidate
        }


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, max_length: int):
    """Collate batch of samples."""
    all_pairs = []
    all_labels = []
    batch_indices = []

    for i, sample in enumerate(batch):
        for pair, label in zip(sample['pairs'], sample['labels']):
            all_pairs.append(pair)
            all_labels.append(label)
            batch_indices.append(i)

    if not all_pairs:
        # Return empty batch
        return {
            'input_ids': torch.empty(0, max_length, dtype=torch.long),
            'attention_mask': torch.empty(0, max_length, dtype=torch.long),
            'labels': torch.empty(0, dtype=torch.float),
            'batch_indices': torch.empty(0, dtype=torch.long)
        }

    # Tokenize all pairs
    encoded = tokenizer(
        [p[0] for p in all_pairs],  # queries
        [p[1] for p in all_pairs],  # prompts
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded.get('token_type_ids', None),
        'labels': torch.tensor(all_labels, dtype=torch.float),
        'batch_indices': torch.tensor(batch_indices, dtype=torch.long)
    }


def load_system_prompts(prog_candidates_dir: str) -> Dict[int, str]:
    """Load system prompts from program candidates."""
    system_prompts = {}

    candidate_dirs = sorted([d for d in os.listdir(prog_candidates_dir) if d.isdigit()],
                          key=int)

    print(f"Loading {len(candidate_dirs)} system prompt candidates...")

    for candidate_dir in candidate_dirs:
        candidate_idx = int(candidate_dir)
        prog_path = os.path.join(prog_candidates_dir, candidate_dir, "program.pkl")

        if not os.path.exists(prog_path):
            print(f"  Warning: Candidate {candidate_idx} not found")
            continue

        try:
            with open(prog_path, 'rb') as f:
                program = pickle.load(f)
            system_prompts[candidate_idx] = str(program)
            print(f"  ✓ Candidate {candidate_idx}: {len(str(program))} chars")
        except Exception as e:
            print(f"  ✗ Candidate {candidate_idx}: {e}")

    return system_prompts


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
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
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(input_ids, attention_mask, token_type_ids)

        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)

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
    training_data: List[Dict],
    system_prompts: Dict[int, str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation set.

    For each task, score all candidates and pick the best one.
    Compare with the ground truth best candidate.
    """
    model.eval()

    correct = 0
    total = 0

    # Also track top-k accuracy
    top_k_correct = {1: 0, 3: 0, 5: 0}

    all_candidates = sorted(system_prompts.keys())

    for task in tqdm(training_data, desc="Evaluating"):
        query = task['claim']
        best_candidate = task['selected_candidate']
        best_score = task['selected_candidate_score']

        # Skip if best candidate failed on this task
        if isinstance(best_score, bool):
            best_score = int(best_score)
        if best_score == 0:
            continue

        # Score all candidates
        candidate_scores = {}

        # Process in batches
        for i in range(0, len(all_candidates), batch_size):
            batch_candidates = all_candidates[i:i+batch_size]

            # Create pairs
            queries = [query] * len(batch_candidates)
            prompts = [system_prompts[c] for c in batch_candidates]

            # Tokenize
            encoded = tokenizer(
                queries,
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

            scores = model(input_ids, attention_mask, token_type_ids)
            scores = torch.sigmoid(scores).cpu().numpy()

            for cand, score in zip(batch_candidates, scores):
                candidate_scores[cand] = score

        # Get top predictions
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_1 = sorted_candidates[0][0]
        top_k_candidates = {
            1: [c for c, _ in sorted_candidates[:1]],
            3: [c for c, _ in sorted_candidates[:3]],
            5: [c for c, _ in sorted_candidates[:5]]
        }

        # Check accuracy
        if top_1 == best_candidate:
            correct += 1

        for k in [1, 3, 5]:
            if best_candidate in top_k_candidates[k]:
                top_k_correct[k] += 1

        total += 1

    accuracy = correct / total if total > 0 else 0
    top_k_accuracy = {k: v / total if total > 0 else 0 for k, v in top_k_correct.items()}

    metrics = {
        'accuracy': accuracy,
        'top_1_accuracy': top_k_accuracy[1],
        'top_3_accuracy': top_k_accuracy[3],
        'top_5_accuracy': top_k_accuracy[5],
        'total_evaluated': total
    }

    return accuracy, metrics


def main():
    parser = argparse.ArgumentParser(description="Train prompt router")

    # Data arguments
    parser.add_argument(
        '--training_data',
        type=str,
        default='router/training_data.json',
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--prog_candidates_dir',
        type=str,
        default='experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates',
        help='Directory with program candidates'
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
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--num_negatives',
        type=int,
        default=3,
        help='Number of negative examples per positive'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
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
        default=1,
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
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} examples")

    # Load system prompts
    print(f"\nLoading system prompts from {args.prog_candidates_dir}...")
    system_prompts = load_system_prompts(args.prog_candidates_dir)
    print(f"Loaded {len(system_prompts)} system prompts")

    # Split into train/val
    random.shuffle(training_data)
    val_size = int(len(training_data) * args.val_split)
    train_data = training_data[val_size:]
    val_data = training_data[:val_size]

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # Create datasets
    train_dataset = PromptRouterDataset(
        train_data,
        system_prompts,
        tokenizer,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        mode='train'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length),
        num_workers=0  # Set to 0 to avoid pickling issues
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
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
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
        print(f"  Total evaluated: {val_metrics['total_evaluated']}")

        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_acc,
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
                'val_accuracy': val_acc,
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
                    'val_accuracy': val_acc,
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
