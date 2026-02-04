"""
Inference script for prompt router.

Load trained model and route queries to best system prompt.
"""

import sys
from pathlib import Path
import json
import pickle
import argparse
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add gepa_artifact to path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from router.model import PromptRouterCrossEncoder


class PromptRouter:
    """Wrapper class for prompt routing inference."""

    def __init__(
        self,
        checkpoint_path: str,
        prog_candidates_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize router.

        Args:
            checkpoint_path: Path to trained model checkpoint
            prog_candidates_dir: Directory with program candidates
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract args
        args = checkpoint['args']
        self.backbone = args['backbone']
        self.max_length = args.get('max_length', 512)

        # Load tokenizer
        print(f"Loading tokenizer: {self.backbone}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)

        # Load model
        print(f"Loading model...")
        self.model = PromptRouterCrossEncoder(
            backbone_name=self.backbone,
            dropout=args.get('dropout', 0.1)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (val acc: {checkpoint.get('val_accuracy', 'N/A')})")

        # Load system prompts
        print(f"Loading system prompts from {prog_candidates_dir}...")
        self.system_prompts = self._load_system_prompts(prog_candidates_dir)
        self.all_candidates = sorted(self.system_prompts.keys())

        print(f"Router ready! Loaded {len(self.system_prompts)} system prompts")

    def _load_system_prompts(self, prog_candidates_dir: str) -> Dict[int, str]:
        """Load system prompts from program candidates."""
        import os
        system_prompts = {}

        candidate_dirs = sorted(
            [d for d in os.listdir(prog_candidates_dir) if d.isdigit()],
            key=int
        )

        for candidate_dir in candidate_dirs:
            candidate_idx = int(candidate_dir)
            prog_path = os.path.join(prog_candidates_dir, candidate_dir, "program.pkl")

            if not os.path.exists(prog_path):
                continue

            try:
                with open(prog_path, 'rb') as f:
                    program = pickle.load(f)
                system_prompts[candidate_idx] = str(program)
            except Exception as e:
                print(f"  Warning: Failed to load candidate {candidate_idx}: {e}")

        return system_prompts

    @torch.no_grad()
    def route(
        self,
        query: str,
        top_k: int = 1,
        return_scores: bool = False,
        batch_size: int = 16
    ) -> List[Tuple[int, float]]:
        """
        Route a query to the best system prompt candidate(s).

        Args:
            query: The user query/task
            top_k: Number of top candidates to return
            return_scores: Whether to return scores with candidates
            batch_size: Batch size for scoring

        Returns:
            List of (candidate_idx, score) tuples, sorted by score
        """
        candidate_scores = {}

        # Score all candidates in batches
        for i in range(0, len(self.all_candidates), batch_size):
            batch_candidates = self.all_candidates[i:i+batch_size]

            # Create pairs
            queries = [query] * len(batch_candidates)
            prompts = [self.system_prompts[c] for c in batch_candidates]

            # Tokenize
            encoded = self.tokenizer(
                queries,
                prompts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Score
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            token_type_ids = encoded.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            scores = self.model(input_ids, attention_mask, token_type_ids)
            scores = torch.sigmoid(scores).cpu().numpy()

            for cand, score in zip(batch_candidates, scores):
                candidate_scores[cand] = float(score)

        # Get top-k
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        if return_scores:
            return sorted_candidates
        else:
            return [cand for cand, _ in sorted_candidates]

    @torch.no_grad()
    def route_batch(
        self,
        queries: List[str],
        top_k: int = 1,
        return_scores: bool = False,
        batch_size: int = 16
    ) -> List[List[Tuple[int, float]]]:
        """
        Route multiple queries at once.

        Args:
            queries: List of user queries/tasks
            top_k: Number of top candidates to return per query
            return_scores: Whether to return scores with candidates
            batch_size: Batch size for scoring

        Returns:
            List of results, one per query
        """
        results = []

        for query in tqdm(queries, desc="Routing"):
            result = self.route(query, top_k, return_scores, batch_size)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Prompt router inference")

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='router/checkpoints/best_checkpoint.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prog_candidates_dir',
        type=str,
        default='experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B/prog_candidates',
        help='Directory with program candidates'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query to route'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        help='Path to test data JSON (optional)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of top candidates to return'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch predictions (optional)'
    )

    args = parser.parse_args()

    # Load router
    router = PromptRouter(
        checkpoint_path=args.checkpoint,
        prog_candidates_dir=args.prog_candidates_dir,
        device=args.device
    )

    # Single query mode
    if args.query:
        print(f"\nQuery: {args.query}")
        print(f"\nTop {args.top_k} candidates:")

        results = router.route(args.query, top_k=args.top_k, return_scores=True)

        for i, (candidate_idx, score) in enumerate(results, 1):
            print(f"  {i}. Candidate {candidate_idx} (score: {score:.4f})")

    # Batch mode
    elif args.test_data:
        print(f"\nLoading test data from {args.test_data}...")
        with open(args.test_data) as f:
            test_data = json.load(f)

        queries = [item['claim'] for item in test_data]

        print(f"Routing {len(queries)} queries...")
        results = router.route_batch(queries, top_k=args.top_k, return_scores=True)

        # Calculate accuracy if ground truth available
        if 'selected_candidate' in test_data[0]:
            correct = 0
            top_k_correct = 0

            for item, result in zip(test_data, results):
                best_pred = result[0][0]
                true_best = item['selected_candidate']

                if best_pred == true_best:
                    correct += 1

                pred_candidates = [cand for cand, _ in result]
                if true_best in pred_candidates:
                    top_k_correct += 1

            accuracy = correct / len(test_data)
            top_k_accuracy = top_k_correct / len(test_data)

            print(f"\nResults:")
            print(f"  Top-1 Accuracy: {accuracy:.4f}")
            print(f"  Top-{args.top_k} Accuracy: {top_k_accuracy:.4f}")

        # Save results
        if args.output:
            output_data = []
            for item, result in zip(test_data, results):
                output_data.append({
                    'task_idx': item.get('task_idx', None),
                    'claim': item['claim'],
                    'predictions': [
                        {'candidate': cand, 'score': score}
                        for cand, score in result
                    ],
                    'true_best': item.get('selected_candidate', None)
                })

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\nPredictions saved to {args.output}")

    else:
        print("\nError: Please provide either --query or --test_data")
        parser.print_help()


if __name__ == '__main__':
    main()
