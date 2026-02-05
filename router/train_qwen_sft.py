"""
Supervised Fine-Tuning (SFT) pipeline for Qwen Router using DSPy.

This script fine-tunes a language model to better select candidate system prompts
using DSPy's optimization framework.
"""

import dspy
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from qwen_router import CandidateSelection, CandidateSelectionWithReasoning


class RouterTrainingExample:
    """
    Training example for router SFT.

    Wraps a task with its candidates and converts to DSPy Example format.
    """

    def __init__(self, task_data: Dict, input_field: str = 'claim'):
        self.task_data = task_data
        self.input_field = input_field

        self.task_input = task_data.get(input_field, '')
        self.candidates = task_data['candidates']

        # Find best candidate(s) based on reward
        best_reward = max(c['reward'] for c in self.candidates)
        self.best_candidates = [
            c for c in self.candidates if c['reward'] == best_reward
        ]
        self.best_candidate_idx = self.best_candidates[0]['candidate_idx']

    def to_dspy_example(self, max_candidates: int = 10, max_prompt_length: int = 200) -> dspy.Example:
        """
        Convert to DSPy Example format.

        Args:
            max_candidates: Maximum number of candidates to include
            max_prompt_length: Maximum length for each prompt

        Returns:
            DSPy Example with task_input, candidates_info, and label
        """
        # Format candidates
        candidates_lines = []
        candidates_to_show = self.candidates[:max_candidates]

        for candidate in candidates_to_show:
            prompt = candidate['candidate_system_prompt']
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."

            candidates_lines.append(
                f"Candidate {candidate['candidate_idx']}:\n{prompt}\n"
            )

        if len(self.candidates) > max_candidates:
            remaining_indices = [
                str(c['candidate_idx'])
                for c in self.candidates[max_candidates:]
            ]
            candidates_lines.append(
                f"\n... and {len(self.candidates) - max_candidates} more candidates "
                f"(indices: {', '.join(remaining_indices)})"
            )

        candidates_info = "\n".join(candidates_lines)

        return dspy.Example(
            task_input=self.task_input,
            candidates_info=candidates_info,
            selected_candidate_idx=str(self.best_candidate_idx)
        ).with_inputs("task_input", "candidates_info")


def metric_correct_selection(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metric function to check if the selected candidate is correct.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace (unused)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract predicted candidate index
    pred_idx_str = prediction.selected_candidate_idx.strip()
    true_idx_str = example.selected_candidate_idx.strip()

    # Try to extract numbers from the strings
    import re
    pred_match = re.search(r'\d+', pred_idx_str)
    true_match = re.search(r'\d+', true_idx_str)

    if pred_match and true_match:
        pred_idx = int(pred_match.group())
        true_idx = int(true_match.group())
        return 1.0 if pred_idx == true_idx else 0.0

    # Fallback to string comparison
    return 1.0 if pred_idx_str == true_idx_str else 0.0


def evaluate_on_dataset(
    predictor,
    examples: List[dspy.Example],
    verbose: bool = False
) -> Dict:
    """
    Evaluate predictor on a dataset.

    Args:
        predictor: DSPy predictor (ChainOfThought or Predict)
        examples: List of DSPy examples
        verbose: Whether to show progress

    Returns:
        Dict with evaluation metrics
    """
    correct = 0
    total = 0

    iterator = tqdm(examples, desc="Evaluating") if verbose else examples

    for example in iterator:
        try:
            prediction = predictor(
                task_input=example.task_input,
                candidates_info=example.candidates_info
            )

            score = metric_correct_selection(example, prediction)
            correct += score
            total += 1

        except Exception as e:
            if verbose:
                print(f"Error during evaluation: {e}")
            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': int(correct),
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen router with DSPy")

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
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum number of training samples to use'
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default='hoverBench',
        help='Benchmark name'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-3.5-turbo',
        help='Model name for DSPy (e.g., openai/gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for the model'
    )
    parser.add_argument(
        '--use_reasoning',
        action='store_true',
        default=True,
        help='Use chain-of-thought reasoning'
    )
    parser.add_argument(
        '--no_reasoning',
        action='store_true',
        help='Disable chain-of-thought reasoning'
    )

    # Training arguments
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['bootstrap', 'copro', 'miprov2'],
        default='bootstrap',
        help='DSPy optimizer to use'
    )
    parser.add_argument(
        '--max_bootstrapped_demos',
        type=int,
        default=4,
        help='Maximum number of bootstrapped demonstrations'
    )
    parser.add_argument(
        '--max_labeled_demos',
        type=int,
        default=8,
        help='Maximum number of labeled demonstrations'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=4,
        help='Number of threads for optimization'
    )

    # Data formatting arguments
    parser.add_argument(
        '--max_candidates_display',
        type=int,
        default=10,
        help='Maximum candidates to show in prompt'
    )
    parser.add_argument(
        '--max_prompt_length',
        type=int,
        default=200,
        help='Maximum length for each candidate prompt'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='router/qwen_checkpoints',
        help='Output directory for saved models'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Setup DSPy LM
    print(f"Setting up DSPy with model: {args.model}")
    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if "openai" in args.model.lower() or "gpt" in args.model.lower():
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        lm = dspy.LM(args.model, api_key=api_key)
    else:
        lm = dspy.LM(args.model)

    dspy.configure(lm=lm)
    print(f"✓ Configured DSPy LM")

    # Load data
    print(f"\nLoading training data from {args.training_data}...")
    with open(args.training_data) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} tasks")

    # Detect input field
    BENCHMARK_INPUT_FIELDS = {
        'hoverBench': 'claim',
        'HotpotQABench': 'question',
        'IFBench': 'prompt',
        'AIMEBench': 'problem',
        'Papillon': 'user_query',
    }
    input_field = BENCHMARK_INPUT_FIELDS.get(args.benchmark_name, 'claim')
    print(f"Using input field: '{input_field}'")

    # Split data
    random.shuffle(all_data)
    val_size = int(len(all_data) * args.val_split)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]

    # Limit training data if specified
    if args.max_train_samples and len(train_data) > args.max_train_samples:
        train_data = train_data[:args.max_train_samples]

    print(f"Split: {len(train_data)} train tasks, {len(val_data)} val tasks")

    # Convert to DSPy examples
    print("\nConverting to DSPy examples...")
    train_examples = [
        RouterTrainingExample(task, input_field).to_dspy_example(
            max_candidates=args.max_candidates_display,
            max_prompt_length=args.max_prompt_length
        )
        for task in tqdm(train_data, desc="Train")
    ]

    val_examples = [
        RouterTrainingExample(task, input_field).to_dspy_example(
            max_candidates=args.max_candidates_display,
            max_prompt_length=args.max_prompt_length
        )
        for task in tqdm(val_data, desc="Val")
    ]

    print(f"Created {len(train_examples)} train examples, {len(val_examples)} val examples")

    # Create predictor
    use_reasoning = not args.no_reasoning if args.no_reasoning else args.use_reasoning

    if use_reasoning:
        print("\nUsing ChainOfThought with reasoning")
        predictor = dspy.ChainOfThought(CandidateSelectionWithReasoning)
    else:
        print("\nUsing Predict without reasoning")
        predictor = dspy.Predict(CandidateSelection)

    # Evaluate before optimization
    print("\n" + "="*80)
    print("BEFORE OPTIMIZATION")
    print("="*80)

    print("\nEvaluating on validation set...")
    val_metrics_before = evaluate_on_dataset(predictor, val_examples[:50], verbose=True)
    print(f"Validation accuracy: {val_metrics_before['accuracy']:.4f}")

    # Optimize
    print("\n" + "="*80)
    print(f"OPTIMIZING WITH {args.optimizer.upper()}")
    print("="*80)

    if args.optimizer == 'bootstrap':
        optimizer = dspy.BootstrapFewShot(
            metric=metric_correct_selection,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
        )
    elif args.optimizer == 'copro':
        optimizer = dspy.COPRO(
            metric=metric_correct_selection,
            breadth=10,
            depth=3,
        )
    elif args.optimizer == 'miprov2':
        optimizer = dspy.MIPROv2(
            metric=metric_correct_selection,
            num_candidates=10,
            init_temperature=1.0,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print(f"\nCompiling predictor...")
    compiled_predictor = optimizer.compile(
        predictor,
        trainset=train_examples,
    )

    print("✓ Compilation complete")

    # Evaluate after optimization
    print("\n" + "="*80)
    print("AFTER OPTIMIZATION")
    print("="*80)

    print("\nEvaluating on validation set...")
    val_metrics_after = evaluate_on_dataset(compiled_predictor, val_examples, verbose=True)
    print(f"Validation accuracy: {val_metrics_after['accuracy']:.4f}")

    # Show improvement
    improvement = val_metrics_after['accuracy'] - val_metrics_before['accuracy']
    print(f"\nAccuracy improvement: {improvement:+.4f}")

    # Save the compiled predictor
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"compiled_predictor_{args.optimizer}.json"
    compiled_predictor.save(str(output_path))
    print(f"\n✓ Saved compiled predictor to {output_path}")

    # Save training config
    config = {
        'args': vars(args),
        'metrics_before': val_metrics_before,
        'metrics_after': val_metrics_after,
        'improvement': improvement,
    }

    config_path = output_dir / f"training_config_{args.optimizer}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Saved training config to {config_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final validation accuracy: {val_metrics_after['accuracy']:.4f}")
    print(f"Improvement: {improvement:+.4f}")


if __name__ == '__main__':
    main()
