"""
Script to construct training dataset for prompt router.

This script creates a training dataset that maps tasks to the best system prompt candidates
from the Pareto frontier based on GEPA optimization results.
"""

import sys
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Set
import random
from datasets import load_dataset
import tqdm
import importlib

# Add gepa_artifact to path for loading pickle files
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def extract_benchmark_name(experiment_dir: str) -> str:
    """
    Extract benchmark name from experiment directory path.

    Experiment directories follow the pattern: {BenchmarkName}_{Program}_{Algorithm}_{Model}
    For example: 'hoverBench_HoverMultiHop_GEPA_Qwen3-8B' -> 'hoverBench'
    """
    dir_name = os.path.basename(experiment_dir)
    # Split by underscore and take the first part
    benchmark_name = dir_name.split('_')[0]
    return benchmark_name


def load_benchmark_dspy_dataset(benchmark_name: str = "hoverBench", seed: int = 0, split: str = "val"):
    """
    Dynamically load the appropriate benchmark dataset based on the benchmark name.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'hoverBench', 'HotpotQABench', 'IFBench')
        seed: Random seed for dataset shuffling
        split: Which split to load ('train', 'val', or 'test')

    Returns:
        DSPy dataset for the specified split
    """
    # Mapping of benchmark names to their module paths
    benchmark_modules = {
        'hoverBench': 'gepa_artifact.benchmarks.hover.hover_data',
        'HotpotQABench': 'gepa_artifact.benchmarks.hotpotQA.hotpot_data',
        'IFBench': 'gepa_artifact.benchmarks.IFBench.ifbench_data',
        'AIMEBench': 'gepa_artifact.benchmarks.AIME.AIME_data',
        'Papillon': 'gepa_artifact.benchmarks.papillon.papillon_data',
    }

    # Default to hover if benchmark not recognized
    if benchmark_name not in benchmark_modules:
        print(f"Warning: Benchmark '{benchmark_name}' not recognized. Defaulting to 'hoverBench'")
        benchmark_name = 'hoverBench'

    module_path = benchmark_modules[benchmark_name]

    # Dynamically import the benchmark module
    module = importlib.import_module(module_path)
    benchmark_class = getattr(module, benchmark_name)

    # Instantiate the benchmark (this will load and prepare the dataset)
    benchmark_instance = benchmark_class(dataset_mode="lite")

    # Get the appropriate split
    if split == "train":
        return benchmark_instance.get_train_set()
    elif split == "val":
        return benchmark_instance.get_val_set()
    elif split == "test":
        return benchmark_instance.get_test_set()
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

def count_unique_docs(example):
    """Count unique documents in supporting facts (from hover_utils.py)"""
    unique_docs = set()
    for fact in example["supporting_facts"]:
        unique_docs.add(fact["key"])
    return len(unique_docs)


def load_gepa_state(state_path: str) -> Dict:
    """Load the GEPA state file."""
    with open(state_path, 'rb') as f:
        state = pickle.load(f)
    return state


def load_hover_valset(seed: int = 0) -> List[Dict]:
    """
    Load and prepare the HoVer validation set, matching the preprocessing
    from hover_data.py.
    """
    dataset = load_dataset("hover", trust_remote_code=True)
    hf_trainset = dataset["train"]

    reformatted_hf_trainset = []

    for example in tqdm.tqdm(hf_trainset, desc="Loading HoVer dataset"):
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        label = example["label"]

        if count_unique_docs(example) == 3:  # Limit to 3 hop examples
            reformatted_hf_trainset.append(
                dict(claim=claim, supporting_facts=supporting_facts, label=label)
            )

    # Apply same shuffling as in hover_data.py
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(reformatted_hf_trainset)

    return reformatted_hf_trainset


def load_system_prompts(prog_candidates_dir: str, num_candidates: int) -> Dict[int, Dict]:
    """
    Load system prompt candidates including the actual prompt text.

    Returns dict with structure:
    {
        candidate_idx: {
            'prompt': str,  # The actual prompt text
            'exists': bool,
            'path': str,
            'dependency_versions': dict (optional)
        }
    }
    """
    system_prompts = {}

    for candidate_idx in range(num_candidates):
        prog_dir = os.path.join(prog_candidates_dir, str(candidate_idx))
        prog_path = os.path.join(prog_dir, "program.pkl")
        metadata_path = os.path.join(prog_dir, "metadata.json")

        if os.path.exists(prog_path):
            try:
                # Load the actual program to get prompt text
                with open(prog_path, 'rb') as f:
                    program = pickle.load(f)
                prompt_text = str(program)

                metadata = {
                    "exists": True,
                    "path": prog_path,
                    "prompt": prompt_text,
                    "prompt_length": len(prompt_text)
                }

                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata["dependency_versions"] = json.load(f)

                system_prompts[candidate_idx] = metadata

            except Exception as e:
                print(f"Warning: Failed to load candidate {candidate_idx}: {e}")
                system_prompts[candidate_idx] = {
                    "exists": False,
                    "error": str(e)
                }
        else:
            print(f"Warning: Program candidate {candidate_idx} not found at {prog_path}")
            system_prompts[candidate_idx] = {"exists": False}

    return system_prompts


def select_best_candidate_for_task(
    task_idx: int,
    pareto_frontier: Set[int],
    candidate_subscores: List[List],
) -> int:
    """
    Select the best candidate for a task from its Pareto frontier.

    Strategy: Among the Pareto frontier candidates, pick the one with the best score
    on this specific task.
    """
    if not pareto_frontier:
        # If no pareto frontier, return a default (shouldn't happen)
        return 0

    best_candidate = None
    best_score = -float('inf')

    for candidate_idx in pareto_frontier:
        score = candidate_subscores[candidate_idx][task_idx]
        # Convert boolean to int for comparison
        if isinstance(score, bool):
            score = int(score)

        if score > best_score:
            best_score = score
            best_candidate = candidate_idx

    return best_candidate


def construct_dataset_split(
    split_name: str,
    dataset: List,
    pareto_frontiers: List[Set[int]],
    prog_candidate_subscores: List[List],
    system_prompts: Dict[int, Dict],
    num_candidates: int
) -> List[Dict]:
    """
    Helper function to construct dataset for a specific split (train or val).

    Args:
        split_name: Name of the split ('train' or 'val')
        dataset: The dataset examples
        pareto_frontiers: List of pareto frontiers (one per task)
        prog_candidate_subscores: Candidate scores for this split
        system_prompts: Dict of system prompts
        num_candidates: Total number of candidates

    Returns:
        List of training examples for this split
    """
    training_data = []
    num_tasks = len(pareto_frontiers)

    print(f"Constructing {split_name} examples (one per task with all candidates)...")
    for task_idx in tqdm.tqdm(range(num_tasks)):
        if task_idx >= len(dataset):
            print(f"Warning: task_idx {task_idx} exceeds dataset size {len(dataset)}, skipping")
            continue

        task = dataset[task_idx]
        pareto_frontier = pareto_frontiers[task_idx]

        # Collect all candidates and their scores for this task
        candidates = []
        for candidate_idx in range(num_candidates):
            # Skip if prompt wasn't loaded successfully
            if candidate_idx not in system_prompts or not system_prompts[candidate_idx].get('exists', False):
                continue
            if 'prompt' not in system_prompts[candidate_idx]:
                continue

            reward = prog_candidate_subscores[candidate_idx][task_idx]
            # Convert boolean to float
            if isinstance(reward, bool):
                reward = float(reward)
            elif reward == 0:
                reward = 0.0
            else:
                reward = float(reward)

            candidates.append({
                "candidate_idx": candidate_idx,
                "candidate_system_prompt": system_prompts[candidate_idx]["prompt"],
                "reward": reward,
                "in_pareto_frontier": candidate_idx in pareto_frontier,
            })

        # Create one training example per task with all candidates
        if candidates:  # Only add if we have at least one valid candidate
            training_example = {
                "task_idx": task_idx,
                **task,  # Include all fields from the task
                "candidates": candidates,
                "num_candidates": len(candidates),
                "pareto_frontier": sorted(list(pareto_frontier)),
            }
            training_data.append(training_example)

    return training_data


def construct_training_data(
    experiment_dir: str,
    output_file: str,
    seed: int = 0,
    benchmark_name: str = None
):
    """
    Construct training and validation datasets for prompt router.

    Args:
        experiment_dir: Path to the experiment directory
        output_file: Base path for output files (will create _train.json and _val.json)
        seed: Random seed for dataset shuffling
        benchmark_name: Name of the benchmark to use. If None, will be extracted from experiment_dir.
                       Defaults to 'hoverBench' if not found.
    """
    print(f"Loading GEPA state from {experiment_dir}...")
    state_path = os.path.join(experiment_dir, "gepa_state.bin")
    state = load_gepa_state(state_path)

    # Extract relevant data from state for BOTH train and val
    # Training set
    pareto_frontiers_train = state.get('program_at_pareto_front', None)
    prog_candidate_train_subscores = state.get('prog_candidate_train_subscores', None)

    # Validation set
    pareto_frontiers_val = state['program_at_pareto_front_valset']
    prog_candidate_val_subscores = state['prog_candidate_val_subscores']

    num_candidates = len(prog_candidate_val_subscores)

    # Check if training data is available
    has_train_data = (pareto_frontiers_train is not None and
                     prog_candidate_train_subscores is not None)

    if has_train_data:
        num_train_tasks = len(pareto_frontiers_train)
        print(f"Found {num_candidates} candidates, {num_train_tasks} train tasks, {len(pareto_frontiers_val)} val tasks")
    else:
        print(f"Found {num_candidates} candidates, {len(pareto_frontiers_val)} val tasks (no train data in state)")

    # Determine which benchmark to use
    if benchmark_name is None:
        benchmark_name = extract_benchmark_name(experiment_dir)
        print(f"Detected benchmark: {benchmark_name}")
    else:
        print(f"Using specified benchmark: {benchmark_name}")

    # Load system prompts (including actual prompt text)
    print("Loading system prompt candidates...")
    prog_candidates_dir = os.path.join(experiment_dir, "prog_candidates")
    system_prompts = load_system_prompts(prog_candidates_dir, num_candidates)

    # Check how many loaded successfully
    loaded_count = sum(1 for p in system_prompts.values() if p.get('exists', False) and 'prompt' in p)
    print(f"Successfully loaded {loaded_count}/{num_candidates} system prompts")

    # Prepare output paths
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create separate files for train and val
    base_name = output_path.stem
    suffix = output_path.suffix
    train_file = output_dir / f"{base_name}_train{suffix}"
    val_file = output_dir / f"{base_name}_val{suffix}"

    # Construct TRAINING dataset if available
    train_data = []
    if has_train_data:
        print(f"\nLoading {benchmark_name} training set...")
        trainset = load_benchmark_dspy_dataset(benchmark_name=benchmark_name, seed=seed, split="train")

        train_data = construct_dataset_split(
            split_name="train",
            dataset=trainset,
            pareto_frontiers=pareto_frontiers_train,
            prog_candidate_subscores=prog_candidate_train_subscores,
            system_prompts=system_prompts,
            num_candidates=num_candidates
        )

        # Save training data
        print(f"\nSaving {len(train_data)} training examples to {train_file}...")
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
    else:
        print("\nSkipping training dataset (not available in GEPA state)")

    # Construct VALIDATION dataset
    print(f"\nLoading {benchmark_name} validation set...")
    valset = load_benchmark_dspy_dataset(benchmark_name=benchmark_name, seed=seed, split="val")

    val_data = construct_dataset_split(
        split_name="val",
        dataset=valset,
        pareto_frontiers=pareto_frontiers_val,
        prog_candidate_subscores=prog_candidate_val_subscores,
        system_prompts=system_prompts,
        num_candidates=num_candidates
    )

    # Save validation data
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)

    # Save system prompt metadata separately for reference (without full prompt text to save space)
    metadata_file = output_dir / "candidate_metadata.json"
    metadata_only = {
        idx: {k: v for k, v in prompt_data.items() if k != 'prompt'}
        for idx, prompt_data in system_prompts.items()
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata_only, f, indent=2)

    # Compute summary statistics for TRAINING data
    def compute_stats(dataset, split_name):
        """Helper to compute statistics for a dataset split."""
        all_rewards = []
        for ex in dataset:
            for candidate in ex["candidates"]:
                all_rewards.append(candidate["reward"])

        reward_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        reward_positive = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0
        avg_candidates = sum(ex["num_candidates"] for ex in dataset) / len(dataset) if dataset else 0

        return {
            "split": split_name,
            "num_tasks": len(dataset),
            "num_unique_candidates": num_candidates,
            "avg_candidates_per_task": avg_candidates,
            "total_task_candidate_pairs": len(all_rewards),
            "reward_statistics": {
                "mean": reward_mean,
                "positive_ratio": reward_positive,
                "total_positive": sum(1 for r in all_rewards if r > 0),
                "total_negative": sum(1 for r in all_rewards if r == 0),
            },
        }

    # Compute statistics for both splits
    train_stats = compute_stats(train_data, "train") if train_data else None
    val_stats = compute_stats(val_data, "val")

    summary = {
        "seed": seed,
        "benchmark_name": benchmark_name,
        "experiment_dir": experiment_dir,
        "num_unique_candidates": num_candidates,
        "train": train_stats,
        "val": val_stats,
    }

    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset construction complete!")
    if train_data:
        print(f"  Training data: {train_file}")
    print(f"  Validation data: {val_file}")
    print(f"  Candidate metadata: {metadata_file}")
    print(f"  Summary: {summary_file}")

    print(f"\nSummary:")
    if train_stats:
        print(f"\n  TRAINING SET:")
        print(f"    Tasks: {train_stats['num_tasks']}")
        print(f"    Avg candidates per task: {train_stats['avg_candidates_per_task']:.1f}")
        print(f"    Total task-candidate pairs: {train_stats['total_task_candidate_pairs']}")
        print(f"    Mean reward: {train_stats['reward_statistics']['mean']:.4f}")
        print(f"    Positive ratio: {train_stats['reward_statistics']['positive_ratio']:.4f}")

    print(f"\n  VALIDATION SET:")
    print(f"    Tasks: {val_stats['num_tasks']}")
    print(f"    Avg candidates per task: {val_stats['avg_candidates_per_task']:.1f}")
    print(f"    Total task-candidate pairs: {val_stats['total_task_candidate_pairs']}")
    print(f"    Mean reward: {val_stats['reward_statistics']['mean']:.4f}")
    print(f"    Positive ratio: {val_stats['reward_statistics']['positive_ratio']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construct training data for prompt router")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment_runs_data/experiment_runs/seed_1/hoverBench_HoverMultiHop_GEPA_Qwen3-8B",
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="router/training_data.json",
        help="Output file path for training data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling"
    )

    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="Name of the benchmark to use (e.g., 'hoverBench', 'HotpotQABench', 'IFBench'). "
             "If not specified, will be auto-detected from experiment_dir. Defaults to 'hoverBench'."
    )

    args = parser.parse_args()

    construct_training_data(
        experiment_dir=args.experiment_dir,
        output_file=args.output_file,
        seed=args.seed,
        benchmark_name=args.benchmark_name
    )
