"""
Evaluation API for prompt routers.

This script evaluates a router's performance on a dataset by comparing
its selections against ground truth rewards.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import asyncio

from router_base import Router, RandomRouter, OracleRouter
from qwen_router import QwenRouter
import dspy


class RouterEvaluator:
    """
    Evaluator for prompt routers.

    Computes various metrics to assess how well a router selects candidates.
    """

    def __init__(self, data_path: str, benchmark_name: str = None):
        """
        Initialize the evaluator.

        Args:
            data_path: Path to the training data JSON file
            benchmark_name: Name of the benchmark (auto-detected if None)
        """
        self.data_path = data_path

        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} tasks")

        # Auto-detect benchmark name if not provided
        if benchmark_name is None:
            self.benchmark_name = self._detect_benchmark()
        else:
            self.benchmark_name = benchmark_name

        print(f"Benchmark: {self.benchmark_name}")

        # Statistics
        self._compute_data_statistics()

    def _detect_benchmark(self) -> str:
        """Auto-detect benchmark from data."""
        if not self.data:
            return "hoverBench"

        # Check which input field is present
        first_task = self.data[0]
        if 'claim' in first_task:
            return 'hoverBench'
        elif 'question' in first_task:
            return 'HotpotQABench'
        elif 'prompt' in first_task:
            return 'IFBench'
        elif 'problem' in first_task:
            return 'AIMEBench'
        elif 'user_query' in first_task:
            return 'Papillon'

        return 'hoverBench'

    def _compute_data_statistics(self):
        """Compute statistics about the data."""
        total_candidates = sum(task['num_candidates'] for task in self.data)
        avg_candidates = total_candidates / len(self.data) if self.data else 0

        all_rewards = []
        for task in self.data:
            for candidate in task['candidates']:
                all_rewards.append(candidate['reward'])

        self.stats = {
            'num_tasks': len(self.data),
            'total_candidates': total_candidates,
            'avg_candidates_per_task': avg_candidates,
            'reward_mean': np.mean(all_rewards) if all_rewards else 0,
            'reward_std': np.std(all_rewards) if all_rewards else 0,
            'positive_ratio': sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0,
        }

    def evaluate(
        self,
        router: Router,
        top_k: List[int] = [1, 3, 5],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a router on the dataset.

        Args:
            router: The router to evaluate
            top_k: List of k values for top-k accuracy computation
            verbose: Whether to print progress

        Returns:
            Dict containing evaluation metrics
        """
        if verbose:
            print(f"\nEvaluating router: {router.get_name()}")
            print("="*80)

        # Metrics
        selected_rewards = []
        best_rewards = []
        top_k_correct = {k: 0 for k in top_k}
        pareto_selections = 0
        total_tasks = 0

        # Per-task results
        task_results = []

        iterator = tqdm(self.data, desc="Evaluating") if verbose else self.data

        for task in iterator:
            task_idx = task['task_idx']
            candidates = task['candidates']

            if not candidates:
                continue

            # Extract task input
            task_input = router.extract_task_input(task)

            # Get router's selection with scores
            selected_idx, scores = router.select_candidate(
                task_input,
                candidates,
                return_scores=True
            )

            # Find the selected candidate's reward
            selected_reward = None
            for candidate in candidates:
                if candidate['candidate_idx'] == selected_idx:
                    selected_reward = candidate['reward']
                    is_in_pareto = candidate.get('in_pareto_frontier', False)
                    break

            if selected_reward is None:
                print(f"Warning: Selected candidate {selected_idx} not found in task {task_idx}")
                continue

            # Find best reward for this task
            rewards = [c['reward'] for c in candidates]
            best_reward = max(rewards)

            # Track metrics
            selected_rewards.append(selected_reward)
            best_rewards.append(best_reward)

            if is_in_pareto:
                pareto_selections += 1

            # Check if selected candidate is optimal
            is_optimal = (selected_reward == best_reward)
            if is_optimal:
                top_k_correct[1] += 1  # Top-1 is correct

            # Check top-k accuracy
            # Sort candidates by router's scores
            sorted_candidates = sorted(
                [(c['candidate_idx'], scores.get(c['candidate_idx'], 0)) for c in candidates],
                key=lambda x: x[1],
                reverse=True
            )

            # Get candidates with best reward
            best_candidate_indices = [
                c['candidate_idx'] for c in candidates if c['reward'] == best_reward
            ]

            for k in top_k:
                if k <= len(sorted_candidates):
                    top_k_indices = [idx for idx, _ in sorted_candidates[:k]]
                    if any(idx in best_candidate_indices for idx in top_k_indices):
                        top_k_correct[k] += 1

            total_tasks += 1

            # Store per-task result
            task_results.append({
                'task_idx': task_idx,
                'selected_candidate': selected_idx,
                'selected_reward': selected_reward,
                'best_reward': best_reward,
                'is_optimal': is_optimal,
                'is_in_pareto': is_in_pareto,
                'num_candidates': len(candidates),
            })

        # Compute final metrics
        metrics = {
            'router_name': router.get_name(),
            'total_tasks': total_tasks,
            'avg_selected_reward': np.mean(selected_rewards) if selected_rewards else 0,
            'avg_best_reward': np.mean(best_rewards) if best_rewards else 0,
            'reward_gap': np.mean(best_rewards) - np.mean(selected_rewards) if selected_rewards else 0,
            'pareto_selection_rate': pareto_selections / total_tasks if total_tasks > 0 else 0,
        }

        # Add top-k accuracies
        for k in top_k:
            metrics[f'top_{k}_accuracy'] = top_k_correct[k] / total_tasks if total_tasks > 0 else 0

        # Compute regret (normalized reward gap)
        if best_rewards:
            metrics['normalized_regret'] = metrics['reward_gap'] / np.mean(best_rewards) if np.mean(best_rewards) > 0 else 0

        # Add data statistics
        metrics['data_stats'] = self.stats

        if verbose:
            self._print_metrics(metrics)

        return {
            'metrics': metrics,
            'task_results': task_results,
        }

    async def evaluate_async(
        self,
        router: Router,
        top_k: List[int] = [1, 3, 5],
        verbose: bool = True,
        batch_size: int = 50,
        max_concurrent: int = 10
    ) -> Dict:
        """
        Async version: Evaluate a router on the dataset with parallel processing.

        This is much faster for LM-based routers like QwenRouter that make API calls.

        Args:
            router: The router to evaluate
            top_k: List of k values for top-k accuracy computation
            verbose: Whether to print progress
            batch_size: Number of tasks to process in each batch
            max_concurrent: Maximum concurrent API calls

        Returns:
            Dict containing evaluation metrics
        """
        if verbose:
            print(f"\nEvaluating router (ASYNC): {router.get_name()}")
            print(f"Batch size: {batch_size}, Max concurrent: {max_concurrent}")
            print("="*80)

        # Check if router supports async
        if not hasattr(router, 'batch_select_candidates_async'):
            if verbose:
                print("Router doesn't support async, falling back to sync evaluation")
            return self.evaluate(router, top_k, verbose)

        # Metrics
        selected_rewards = []
        best_rewards = []
        top_k_correct = {k: 0 for k in top_k}
        pareto_selections = 0
        total_tasks = 0

        # Per-task results
        task_results = []

        # Filter valid tasks
        valid_tasks = [task for task in self.data if task['candidates']]

        # Create batches
        batches = [valid_tasks[i:i + batch_size] for i in range(0, len(valid_tasks), batch_size)]

        if verbose:
            print(f"Processing {len(valid_tasks)} tasks in {len(batches)} batches...")

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            if verbose:
                print(f"Processing batch {batch_idx + 1}/{len(batches)}...")

            # Extract inputs and candidates for the batch
            batch_task_inputs = [router.extract_task_input(task) for task in batch]
            batch_candidates = [task['candidates'] for task in batch]

            # Get router's selections with scores using async batch processing
            selected_indices, all_scores = await router.batch_select_candidates_async(
                batch_task_inputs,
                batch_candidates,
                return_scores=True,
                max_concurrent=max_concurrent
            )

            # Process each task in the batch
            for task, selected_idx, scores in zip(batch, selected_indices, all_scores):
                task_idx = task['task_idx']
                candidates = task['candidates']

                # Find the selected candidate's reward
                selected_reward = None
                is_in_pareto = False
                for candidate in candidates:
                    if candidate['candidate_idx'] == selected_idx:
                        selected_reward = candidate['reward']
                        is_in_pareto = candidate.get('in_pareto_frontier', False)
                        break

                if selected_reward is None:
                    print(f"Warning: Selected candidate {selected_idx} not found in task {task_idx}")
                    continue

                # Find best reward for this task
                rewards = [c['reward'] for c in candidates]
                best_reward = max(rewards)

                # Track metrics
                selected_rewards.append(selected_reward)
                best_rewards.append(best_reward)

                if is_in_pareto:
                    pareto_selections += 1

                # Check if selected candidate is optimal
                is_optimal = (selected_reward == best_reward)
                if is_optimal:
                    top_k_correct[1] += 1  # Top-1 is correct

                # Check top-k accuracy
                # Sort candidates by router's scores
                sorted_candidates = sorted(
                    [(c['candidate_idx'], scores.get(c['candidate_idx'], 0)) for c in candidates],
                    key=lambda x: x[1],
                    reverse=True
                )

                # Get candidates with best reward
                best_candidate_indices = [
                    c['candidate_idx'] for c in candidates if c['reward'] == best_reward
                ]

                for k in top_k:
                    if k <= len(sorted_candidates):
                        top_k_indices = [idx for idx, _ in sorted_candidates[:k]]
                        if any(idx in best_candidate_indices for idx in top_k_indices):
                            top_k_correct[k] += 1

                total_tasks += 1

                # Store per-task result
                task_results.append({
                    'task_idx': task_idx,
                    'selected_candidate': selected_idx,
                    'selected_reward': selected_reward,
                    'best_reward': best_reward,
                    'is_optimal': is_optimal,
                    'is_in_pareto': is_in_pareto,
                    'num_candidates': len(candidates),
                })

        # Compute final metrics
        metrics = {
            'router_name': router.get_name(),
            'total_tasks': total_tasks,
            'avg_selected_reward': np.mean(selected_rewards) if selected_rewards else 0,
            'avg_best_reward': np.mean(best_rewards) if best_rewards else 0,
            'reward_gap': np.mean(best_rewards) - np.mean(selected_rewards) if selected_rewards else 0,
            'pareto_selection_rate': pareto_selections / total_tasks if total_tasks > 0 else 0,
        }

        # Add top-k accuracies
        for k in top_k:
            metrics[f'top_{k}_accuracy'] = top_k_correct[k] / total_tasks if total_tasks > 0 else 0

        # Compute regret (normalized reward gap)
        if best_rewards:
            metrics['normalized_regret'] = metrics['reward_gap'] / np.mean(best_rewards) if np.mean(best_rewards) > 0 else 0

        # Add data statistics
        metrics['data_stats'] = self.stats

        if verbose:
            self._print_metrics(metrics)

        return {
            'metrics': metrics,
            'task_results': task_results,
        }

    def _print_metrics(self, metrics: Dict):
        """Print metrics in a readable format."""
        print("\nEvaluation Results:")
        print("-" * 80)
        print(f"  Router: {metrics['router_name']}")
        print(f"  Total tasks evaluated: {metrics['total_tasks']}")
        print()
        print("Performance Metrics:")
        print(f"  Avg selected reward: {metrics['avg_selected_reward']:.4f}")
        print(f"  Avg best reward: {metrics['avg_best_reward']:.4f}")
        print(f"  Reward gap: {metrics['reward_gap']:.4f}")
        print(f"  Normalized regret: {metrics.get('normalized_regret', 0):.4f}")
        print(f"  Pareto selection rate: {metrics['pareto_selection_rate']:.4f}")
        print()
        print("Top-K Accuracy:")
        for key in sorted(metrics.keys()):
            if key.startswith('top_'):
                print(f"  {key.replace('_', '-').title()}: {metrics[key]:.4f}")
        print("-" * 80)

    def compare_routers(
        self,
        routers: List[Router],
        top_k: List[int] = [1, 3, 5],
        output_file: str = None
    ) -> Dict:
        """
        Compare multiple routers on the same dataset.

        Args:
            routers: List of routers to compare
            top_k: List of k values for top-k accuracy
            output_file: Optional path to save comparison results

        Returns:
            Dict mapping router names to their evaluation results
        """
        print(f"\nComparing {len(routers)} routers...")
        print("="*80)

        results = {}
        for router in routers:
            result = self.evaluate(router, top_k=top_k, verbose=True)
            results[router.get_name()] = result

        # Print comparison table
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        # Create comparison table
        print(f"\n{'Router':<30} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Top-5 Acc':<12} {'Avg Reward':<12}")
        print("-" * 80)

        for router_name, result in results.items():
            metrics = result['metrics']
            print(f"{router_name:<30} "
                  f"{metrics.get('top_1_accuracy', 0):<12.4f} "
                  f"{metrics.get('top_3_accuracy', 0):<12.4f} "
                  f"{metrics.get('top_5_accuracy', 0):<12.4f} "
                  f"{metrics['avg_selected_reward']:<12.4f}")

        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare results for JSON (remove task_results for brevity)
            save_results = {
                name: {'metrics': res['metrics']}
                for name, res in results.items()
            }

            with open(output_file, 'w') as f:
                json.dump(save_results, f, indent=2)

            print(f"\nResults saved to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt routers")

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default=None,
        help='Name of the benchmark (auto-detected if not specified)'
    )
    parser.add_argument(
        '--router',
        type=str,
        choices=['random', 'oracle', 'qwen', 'all'],
        default='random',
        help='Which router to evaluate: random, oracle, qwen, or all'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for RandomRouter'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        nargs='+',
        default=[1, 3, 5],
        help='Values of k for top-k accuracy (default: 1 3 5)'
    )
    parser.add_argument(
        '--use_async',
        action='store_true',
        help='Use async evaluation for faster processing (recommended for Qwen router)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for async evaluation (default: 50)'
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=10,
        help='Maximum concurrent API calls for async evaluation (default: 10)'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RouterEvaluator(
        data_path=args.data_path,
        benchmark_name=args.benchmark_name
    )

    # Create routers
    routers = []
    if args.router == 'random' or args.router == 'all':
        routers.append(RandomRouter(
            benchmark_name=evaluator.benchmark_name,
            seed=args.seed
        ))

    if args.router == 'oracle' or args.router == 'all':
        routers.append(OracleRouter(
            benchmark_name=evaluator.benchmark_name
        ))

    if args.router == 'qwen' or args.router == 'all':
        routers.append(QwenRouter(
            lm=dspy.LM('openai/Qwen/Qwen3-4B-Instruct-2507', api_key='nothing', api_base='http://localhost:8000/v1/', cache=True),
            benchmark_name=evaluator.benchmark_name
        ))

    # Evaluate
    if len(routers) == 1:
        # Use async evaluation if requested
        if args.use_async:
            result = asyncio.run(evaluator.evaluate_async(
                routers[0],
                top_k=args.top_k,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent
            ))
        else:
            result = evaluator.evaluate(routers[0], top_k=args.top_k)

        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, 'w') as f:
                json.dump({
                    'metrics': result['metrics'],
                    # Optionally include per-task results
                    # 'task_results': result['task_results']
                }, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
    else:
        # Compare multiple routers
        evaluator.compare_routers(
            routers,
            top_k=args.top_k,
            output_file=args.output_file
        )


if __name__ == '__main__':
    main()
