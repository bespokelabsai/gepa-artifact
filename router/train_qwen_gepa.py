"""
Train Qwen Router using GEPA (Generalized Evolutionary Prompt Adaptation).

This script optimizes the router's prompt using DSPy's GEPA optimizer,
which iteratively improves the instruction prompts for candidate selection.
"""

import dspy
import json
import argparse
import random
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from qwen_router import CandidateSelection

# Use DSPy's built-in GEPA
from dspy.teleprompt import GEPA


def construct_training_examples(
    task_data_list: List[Dict],
    input_field: str = 'claim',
    max_candidates_display: int = 10,
) -> List[dspy.Example]:
    """
    Convert raw training data to DSPy examples.

    Args:
        task_data_list: List of task dictionaries with candidates and rewards
        input_field: Field name for task input (e.g., 'claim', 'question')
        max_candidates_display: Maximum candidates to show

    Returns:
        List of DSPy Examples
    """
    examples = []

    for task_data in task_data_list:
        task_input = task_data.get(input_field, '')
        candidates = task_data['candidates']

        # Get pareto frontier candidates
        pareto_frontier = task_data.get('pareto_frontier', [])
        if not pareto_frontier:
            # Fallback: find best candidate based on reward if no pareto frontier
            best_reward = max(c['reward'] for c in candidates)
            pareto_frontier = [c['candidate_idx'] for c in candidates if c['reward'] == best_reward]

        # Use first pareto frontier candidate as the label
        best_candidate_idx = pareto_frontier[0]

        # Format candidates for display
        candidates_str = [f"{candidate['candidate_idx']}.\n{candidate['candidate_system_prompt']}" for idx, candidate in enumerate(candidates)]

        # Create DSPy example with ground truth
        example = dspy.Example(
            task_input=task_input,
            candidates=candidates_str,
            selected_candidate_idx=str(best_candidate_idx),
            pareto_frontier=pareto_frontier,
            all_candidates=candidates  # Store for metric evaluation
        ).with_inputs("task_input", "candidates")

        examples.append(example)

    return examples


def router_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Simple metric to evaluate router selection quality.

    Returns the reward of the selected candidate.

    Args:
        example: Ground truth example with candidates and their rewards
        prediction: Model prediction
        trace: Optional trace (unused)

    Returns:
        The reward value of the selected candidate (0.0 if invalid selection)
    """
    import re

    # Extract predicted candidate index
    pred_output = prediction.selected_candidate_idx.strip()

    # Try to extract number from prediction
    pred_match = re.search(r'\d+', pred_output)

    if not pred_match:
        return 0.0

    pred_idx = int(pred_match.group())

    # Find the predicted candidate and return its reward
    all_candidates = example.all_candidates
    pred_candidate = next((c for c in all_candidates if c['candidate_idx'] == pred_idx), None)

    if pred_candidate is None:
        return 0.0

    return pred_candidate['reward']


def router_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None
) -> dspy.Prediction:
    """
    Enhanced metric with feedback for GEPA optimization.

    Evaluates predictions and generates detailed feedback for learning.

    Args:
        example: DSPy Example with ground truth pareto frontier
        prediction: DSPy Prediction with model's selected candidate
        trace: Optional trace (unused)
        pred_name: Optional prediction name (unused)
        pred_trace: Optional prediction trace (unused)

    Returns:
        DSPy Prediction with score and detailed feedback text
    """
    import re

    # Extract predicted candidate index
    try:
        pred_output = prediction.selected_candidate_idx.strip()
        pred_match = re.search(r'\d+', pred_output)

        if not pred_match:
            feedback_text = (
                f"Failed to extract a valid candidate index from '{pred_output}'. "
                f"You must output a clear integer number representing the candidate index."
            )
            return dspy.Prediction(score=0.0, feedback=feedback_text)

        pred_idx = int(pred_match.group())

    except Exception as e:
        feedback_text = f"Error parsing prediction: {str(e)}. Please output a valid candidate index."
        return dspy.Prediction(score=0.0, feedback=feedback_text)

    # Get example metadata
    pareto_frontier = example.pareto_frontier
    all_candidates = example.all_candidates

    # Find the predicted candidate
    pred_candidate = next((c for c in all_candidates if c['candidate_idx'] == pred_idx), None)

    if pred_candidate is None:
        feedback_text = (
            f"Selected invalid candidate index {pred_idx}. "
            f"Valid candidate indices are: {[c['candidate_idx'] for c in all_candidates]}. "
            f"Please select a valid candidate from the provided list."
        )
        return dspy.Prediction(score=0.0, feedback=feedback_text)

    pred_reward = pred_candidate['reward']
    score = pred_reward

    # Generate feedback based on pareto frontier
    if pred_idx in pareto_frontier:
        # Correct selection - on pareto frontier
        pareto_candidates = [c for c in all_candidates if c['candidate_idx'] in pareto_frontier]
        pareto_info = ", ".join([f"candidate {c['candidate_idx']} (reward={c['reward']:.3f})" for c in pareto_candidates[:3]])

        feedback_text = (
            f"✓ Excellent choice! You selected candidate {pred_idx} with reward {pred_reward:.3f}, "
            f"which is on the pareto frontier. "
            f"The pareto frontier includes: {pareto_info}. "
            f"These candidates represent optimal trade-offs between quality and cost."
        )
    else:
        # Sub-optimal selection - not on pareto frontier
        pareto_candidates = [c for c in all_candidates if c['candidate_idx'] in pareto_frontier]
        pareto_rewards = [c['reward'] for c in pareto_candidates]
        best_pareto_reward = max(pareto_rewards) if pareto_rewards else 0
        reward_diff = best_pareto_reward - pred_reward

        pareto_info = ", ".join([f"candidate {c['candidate_idx']} (reward={c['reward']:.3f})" for c in pareto_candidates[:3]])
        if len(pareto_candidates) > 3:
            pareto_info += f" and {len(pareto_candidates) - 3} more"

        feedback_text = (
            f"✗ You selected candidate {pred_idx} with reward {pred_reward:.3f}, "
            f"but it is NOT on the pareto frontier. "
            f"The pareto frontier candidates are: {pareto_info}. "
            f"The best pareto candidate has reward {best_pareto_reward:.3f} "
            f"(you are {reward_diff:.3f} points below). "
            f"\n\nKey insights to improve:\n"
            f"1. Look for candidates that balance performance and cost effectively\n"
            f"2. Pareto frontier candidates are non-dominated - no other candidate is better in all metrics\n"
            f"3. Consider the task requirements carefully to identify which trade-offs are most appropriate"
        )

    return dspy.Prediction(score=score, feedback=feedback_text)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen router with GEPA")

    # Data arguments
    parser.add_argument(
        '--training_data',
        type=str,
        default='router/hoverBench_val.json',
        help='Path to training data JSON'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Path to validation data JSON (if not provided, splits from training data)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio (used only if val_data is not provided)'
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=1000,
        help='Maximum number of training samples to use'
    )
    parser.add_argument(
        '--max_val_samples',
        type=int,
        default=1000,
        help='Maximum number of validation samples to use'
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default='hoverBench',
        help='Benchmark name for input field detection'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='openai/Qwen/Qwen3-30B-A3B-Instruct-2507',
        help='Model name for DSPy (student LM for inference)'
    )
    parser.add_argument(
        '--reflection_model',
        type=str,
        default='openai/gpt-5',
        help='Reflection LM for GEPA error analysis and prompt improvement'
    )

    # GEPA optimizer arguments
    parser.add_argument(
        '--auto',
        type=str,
        choices=['light', 'medium', 'heavy'],
        default='light',
        help='GEPA optimization intensity (light=fast, medium=balanced, heavy=thorough)'
    )
    parser.add_argument(
        '--max_metric_calls',
        type=int,
        default=None,
        help='Maximum total metric function calls (alternative to auto)'
    )
    parser.add_argument(
        '--reflection_minibatch_size',
        type=int,
        default=3,
        help='Number of examples per reflection step'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=8,
        help='Number of threads for parallel evaluation'
    )

    # Data formatting arguments
    parser.add_argument(
        '--max_candidates_display',
        type=int,
        default=30,
        help='Maximum candidates to show in prompt'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='router/gepa_checkpoints',
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

    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}, using system environment variables")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = str(output_dir / f"run_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("GEPA TRAINING FOR QWEN ROUTER")
    print("=" * 80)

    # Setup DSPy LMs
    print(f"\nSetting up language models...")
    api_key = os.environ.get("OPENAI_API_KEY")
    wandb_key = os.environ.get("WANDB_API_KEY")

    # Student LM (for inference)
    print(f"  Student LM: {args.model}")
    if "gpt" in args.model.lower():
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        lm = dspy.LM(args.model, api_key=api_key, temperature=1.0)
    else:
        lm = dspy.LM(args.model, api_base="http://localhost:8000/v1", api_key="api_key", temperature=1.0)

    # Reflection LM (for GEPA meta-cognitive analysis)
    print(f"  Reflection LM: {args.reflection_model}")
    if "gpt" in args.reflection_model.lower():
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found for reflection model")
        reflection_lm = dspy.LM(args.reflection_model, api_key=api_key, temperature=1.0, cache=True)
    else:
        reflection_lm = dspy.LM(args.reflection_model, api_base="http://localhost:8000/v1", api_key="api_key", temperature=1.0)

    dspy.configure(lm=lm)
    print(f"✓ Configured DSPy with student LM: {args.model}")
    print(f"✓ Configured reflection LM: {args.reflection_model}")

    # Load training data
    print(f"\nLoading training data from {args.training_data}...")
    with open(args.training_data) as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training tasks")

    # Load or split validation data
    if args.val_data is not None:
        print(f"\nLoading validation data from {args.val_data}...")
        with open(args.val_data) as f:
            val_data = json.load(f)
        print(f"Loaded {len(val_data)} validation tasks")

        # Shuffle training data only
        random.shuffle(train_data)
    else:
        print(f"\nNo separate validation data provided. Splitting from training data...")
        # Split data into train and validation
        random.shuffle(train_data)
        val_size = int(len(train_data) * args.val_split)
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        print(f"Split with ratio {args.val_split}: {len(train_data)} train, {len(val_data)} val tasks")

    # Detect input field based on benchmark
    BENCHMARK_INPUT_FIELDS = {
        'hoverBench': 'claim',
        'HotpotQABench': 'question',
        'IFBench': 'prompt',
        'AIMEBench': 'problem',
        'Papillon': 'user_query',
    }
    input_field = BENCHMARK_INPUT_FIELDS.get(args.benchmark_name, 'claim')
    print(f"Using input field: '{input_field}'")

    # Limit sample sizes
    if args.max_train_samples and len(train_data) > args.max_train_samples:
        train_data = train_data[:args.max_train_samples]
    if args.max_val_samples and len(val_data) > args.max_val_samples:
        val_data = val_data[:args.max_val_samples]

    print(f"Final sizes: {len(train_data)} train tasks, {len(val_data)} val tasks")

    # Convert to DSPy examples
    print("\nConverting to DSPy examples...")
    train_examples = construct_training_examples(
        train_data,
        input_field=input_field,
        max_candidates_display=args.max_candidates_display,
    )

    val_examples = construct_training_examples(
        val_data,
        input_field=input_field,
        max_candidates_display=args.max_candidates_display,
    )


    print(f"Created {len(train_examples)} train examples, {len(val_examples)} val examples")

    # Create base router module
    print(f"\nCreating router module")
    router_module = dspy.Predict(CandidateSelection)
    router_module.set_lm(lm)

    # Initialize GEPA optimizer
    print("\n" + "=" * 80)
    print("INITIALIZING GEPA OPTIMIZER")
    print("=" * 80)

    # GEPA configuration following the tutorial structure
    gepa_config = {
        'metric': router_metric_with_feedback,
        'reflection_lm': reflection_lm,
        'num_threads': args.num_threads,
        'track_stats': True,
        'reflection_minibatch_size': args.reflection_minibatch_size,
        'track_best_outputs': True,
        'add_format_failure_as_feedback': True,
        'use_wandb': True,
        "wandb_api_key": wandb_key,
        "log_dir": run_dir,
    }

    # Add budget control (either auto or manual)
    if args.max_metric_calls is not None:
        gepa_config['max_metric_calls'] = args.max_metric_calls
        print(f"Using manual budget: max_metric_calls={args.max_metric_calls}")
    else:
        gepa_config['auto'] = args.auto
        print(f"Using auto configuration: {args.auto}")

    print(f"GEPA config: {gepa_config}")
    # import pdb; pdb.set_trace()
    gepa_optimizer = GEPA(**gepa_config)

    print("✓ GEPA optimizer initialized")
    print(f"  - Reflection minibatch size: {args.reflection_minibatch_size}")
    print(f"  - Num threads: {args.num_threads}")
    print(f"  - Track stats: True")

    # Baseline evaluation before optimization
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (Before Optimization)")
    print("=" * 80)

    baseline_evaluator = dspy.Evaluate(
        devset=val_examples,
        metric=router_metric,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=False,
    )

    baseline_score = baseline_evaluator(router_module)

    print(f"\n✓ Baseline validation score: {baseline_score.score}")
    print(f"  (Average reward of selected candidates)")

    # Run GEPA optimization
    print("\n" + "=" * 80)
    print("STARTING GEPA OPTIMIZATION")
    print("=" * 80)
    print("\nGEPA will:")
    print("  1. Run router on training examples")
    print("  2. Analyze errors and generate feedback")
    print("  3. Use reflection LM to identify failure patterns")
    print("  4. Generate improved prompt instructions")
    print("  5. Validate improvements on validation set")
    print("  6. Iterate to find best prompt\n")

    try:
        optimized_router = gepa_optimizer.compile(
            student=router_module,
            trainset=train_examples,
            valset=val_examples,
        )

        print("\n" + "=" * 80)
        print("GEPA OPTIMIZATION COMPLETE")
        print("=" * 80)

        # Evaluate optimized router
        print("\n" + "=" * 80)
        print("FINAL EVALUATION (After Optimization)")
        print("=" * 80)

        final_evaluator = dspy.Evaluate(
            devset=val_examples,
            metric=router_metric,
            num_threads=args.num_threads,
            display_progress=True,
            display_table=False,
        )

        optimized_score = final_evaluator(optimized_router)
        print(f"\n✓ Optimized validation score: {optimized_score.score}")

        # Show improvement
        improvement = optimized_score.score - baseline_score.score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        print(f"\n{'='*80}")
        print("PERFORMANCE IMPROVEMENT")
        print(f"{'='*80}")
        print(f"  Baseline score:   {baseline_score.score}")
        print(f"  Optimized score:  {optimized_score.score}")
        print(f"  Absolute gain:    {improvement}")
        print(f"  Relative gain:    {improvement_pct}%")

        # Show optimized instructions
        print(f"\n{'='*80}")
        print("OPTIMIZED PROMPT INSTRUCTIONS")
        print(f"{'='*80}")
        try:
            optimized_instructions = optimized_router.predict.signature.instructions
            print(optimized_instructions)
        except:
            print("(Instructions not available)")

        # Save optimized router
        optimized_router.save(os.path.join(run_dir, "optimized_router.pkl"))
        print(f"\n✓ Saved optimized router to {run_dir}/optimized_router.pkl")

        # Save configuration and results
        config = {
            'args': vars(args),
            'input_field': input_field,
            'num_train_examples': len(train_examples),
            'num_val_examples': len(val_examples),
            'baseline_score': baseline_score,
            'optimized_score': optimized_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
        }

        config_path = os.path.join(run_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved training config to {config_path}")

        # Save optimized instructions to separate file
        if hasattr(optimized_router, 'predict') and hasattr(optimized_router.predict.signature, 'instructions'):
            instructions_path = os.path.join(run_dir, "optimized_instructions.txt")
            with open(instructions_path, 'w') as f:
                f.write(optimized_router.predict.signature.instructions)
            print(f"✓ Saved optimized instructions to {instructions_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nERROR during GEPA optimization: {e}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
