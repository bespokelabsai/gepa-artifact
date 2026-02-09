"""
Script to construct training dataset for prompt router.

This script creates a training dataset that maps tasks to the best system prompt candidates
from the Pareto frontier based on GEPA optimization results.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Set
import random
from datasets import load_dataset
import tqdm
import importlib

# Add paths for importing benchmarks and gepa_artifact modules
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Add gepa_artifact directory to allow "benchmarks" to be imported as a top-level module
gepa_artifact_dir = root_dir / "gepa_artifact"
if str(gepa_artifact_dir) not in sys.path:
    sys.path.insert(0, str(gepa_artifact_dir))

# Add benchmarks directory to allow benchmark packages like "langProBe" to be imported as top-level modules
benchmarks_dir = gepa_artifact_dir / "benchmarks"
if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))

# # Add the nested langProBe directory to allow direct import of langProBe.benchmark
# langprobe_nested_dir = benchmarks_dir / "langProBe" / "langProBe"
# if str(langprobe_nested_dir) not in sys.path:
#     sys.path.insert(0, str(langprobe_nested_dir))

# from langProBe.langProBe.dspy_program import LangProBeDSPyMetaProgram
import pickle
import builtins

# Inject LangProBeDSPyMetaProgram into builtins so it's available globally
# This allows langProBe modules to find it when they're imported
# builtins.LangProBeDSPyMetaProgram = LangProBeDSPyMetaProgram


# Fix Pydantic v2 compatibility with older pickled DSPy objects
# The pickles were created with DSPy 2.6.23 which used older Pydantic
# DSPy 3.1.3 uses newer Pydantic which removed the 'exclude_if' attribute
try:
    from pydantic.fields import FieldInfo

    # Store the original __getattribute__ method
    original_getattribute = FieldInfo.__getattribute__

    def patched_getattribute(self, name):
        """Patched version that returns None for 'exclude_if'"""
        if name == 'exclude_if':
            return None
        return original_getattribute(self, name)

    # Apply the patch
    FieldInfo.__getattribute__ = patched_getattribute
    print("Applied Pydantic compatibility patch for DSPy 2.6.23 -> 3.1.3 migration")

except Exception as e:
    print(f"Warning: Could not apply Pydantic compatibility patch: {e}")


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


def get_experiment_dir_for_benchmark(benchmark_name: str, seed: int = 0) -> str:
    """
    Construct the experiment directory path for a given benchmark.

    Args:
        benchmark_name: Name of the benchmark
        seed: Random seed number (default: 0)

    Returns:
        Full path to the experiment directory
    """
    # Mapping of benchmark names to their experiment directory patterns
    benchmark_dirs = {
        'hoverBench': f'experiment_runs_data/experiment_runs/seed_{seed}/hoverBench_HoverMultiHop_GEPA_qwen3-8b',
        'HotpotQABench': f'experiment_runs_data/experiment_runs/seed_{seed}/HotpotQABench_HotpotQA_GEPA_qwen3-8b',
        'IFBench': f'experiment_runs_data/experiment_runs/seed_{seed}/IFBench_IFEval_GEPA_qwen3-8b',
        'AIMEBench': f'experiment_runs_data/experiment_runs/seed_{seed}/AIMEBench_AIME_GEPA_qwen3-8b',
        'Papillon': f'experiment_runs_data/experiment_runs/seed_{seed}/Papillon_Papillon_GEPA_qwen3-8b',
    }

    # Default to hoverBench if benchmark not recognized
    if benchmark_name not in benchmark_dirs:
        print(f"Warning: Benchmark '{benchmark_name}' not recognized. Defaulting to 'hoverBench'")
        benchmark_name = 'hoverBench'

    return benchmark_dirs[benchmark_name]


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
        'hoverBench': 'gepa_artifact.benchmarks.langProBe.langProBe.langProBe.hover.hover_data',
        'HotpotQABench': 'gepa_artifact.benchmarks.langProBe.langProBe.langProBe.hotpotQA.hotpot_data',
        'IFBench': 'gepa_artifact.benchmarks.langProBe.langProBe.langProBe.IFBench.ifbench_data',
        'AIMEBench': 'gepa_artifact.benchmarks.langProBe.langProBe.langProBe.AIME.AIME_data',
        'Papillon': 'gepa_artifact.benchmarks.langProBe.langProBe.langProBe.papillon.papillon_data',
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


def load_system_prompts(prog_candidates_dir: str, num_candidates: int, skip_first: bool = False) -> Dict[int, Dict]:
    """
    Load system prompt candidates including the actual prompt text.

    Args:
        prog_candidates_dir: Directory containing program candidates
        num_candidates: Total number of candidates (including skipped ones)
        skip_first: If True, skip candidate 0

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

    start_idx = 1 if skip_first else 0
    for candidate_idx in range(start_idx, num_candidates):
        prog_dir = os.path.join(prog_candidates_dir, str(candidate_idx))
        prog_path = os.path.join(prog_dir, "program.pkl")
        metadata_path = os.path.join(prog_dir, "metadata.json")

        if os.path.exists(prog_path):
            try:
                # Load the actual program to get prompt text
                with open(prog_path, 'rb') as f:
                    program = pickle.load(f)

                # Get string representation (Pydantic patch applied at import time)
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
                import traceback
                if candidate_idx == 0:  # Print full traceback for first candidate
                    print(f"Warning: Failed to load candidate {candidate_idx}:")
                    traceback.print_exc()
                else:
                    print(f"Warning: Failed to load candidate {candidate_idx}: {e}")
                system_prompts[candidate_idx] = {
                    "exists": False,
                    "error": str(e)
                }
        else:
            print(f"Warning: Program candidate {candidate_idx} not found at {prog_path}")
            system_prompts[candidate_idx] = {"exists": False}

    return system_prompts


def construct_dataset_split(
    split_name: str,
    dataset: List,
    prog_candidate_subscores: List[List],
    system_prompts: Dict[int, Dict],
    num_candidates: int,
    candidate_offset: int = 0
) -> List[Dict]:
    """
    Helper function to construct dataset for a specific split.

    Args:
        split_name: Name of the split ('train' or 'val')
        dataset: The dataset examples
        prog_candidate_subscores: Candidate scores for this split (programs x tasks)
        system_prompts: Dict of system prompts (keyed by original candidate indices)
        num_candidates: Total number of candidates in the scores array
        candidate_offset: Offset to add to score indices to get original candidate indices (e.g., 1 if we skipped candidate 0)

    Returns:
        List of training examples for this split
    """
    training_data = []
    num_tasks = len(prog_candidate_subscores[0]) if num_candidates > 0 else 0

    print(f"Constructing {split_name} examples (one per task with all candidates)...")
    for task_idx in tqdm.tqdm(range(num_tasks)):
        if task_idx >= len(dataset):
            print(f"Warning: task_idx {task_idx} exceeds dataset size {len(dataset)}, skipping")
            continue

        task = dataset[task_idx]

        # Collect all candidates and their scores for this task
        candidates = []
        best_score = -float('inf')
        best_candidate_idx = None

        for score_idx in range(num_candidates):
            # Map score index to original candidate index
            candidate_idx = score_idx + candidate_offset

            # Skip if prompt wasn't loaded successfully
            if candidate_idx not in system_prompts or not system_prompts[candidate_idx].get('exists', False):
                continue
            if 'prompt' not in system_prompts[candidate_idx]:
                continue

            reward = prog_candidate_subscores[score_idx][task_idx]
            # Convert boolean to float
            if isinstance(reward, bool):
                reward = float(reward)
            elif reward == 0:
                reward = 0.0
            else:
                reward = float(reward)

            # Track best candidate for this task
            if reward > best_score:
                best_score = reward
                best_candidate_idx = candidate_idx

            candidates.append({
                "candidate_idx": candidate_idx,
                "candidate_system_prompt": system_prompts[candidate_idx]["prompt"],
                "reward": reward,
            })

        # Create one training example per task with all candidates
        if candidates:  # Only add if we have at least one valid candidate
            training_example = {
                "task_idx": task_idx,
                **task,  # Include all fields from the task
                "candidates": candidates,
                "num_candidates": len(candidates),
                "best_candidate_idx": best_candidate_idx,
                "best_score": best_score,
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

    # Extract ONLY validation scores, SKIPPING candidate 0
    prog_candidate_val_subscores = state['prog_candidate_val_subscores'][1:]

    num_candidates = len(prog_candidate_val_subscores)
    num_val_tasks = len(prog_candidate_val_subscores[0]) if num_candidates > 0 else 0

    print(f"Found {num_candidates} candidates (skipped candidate 0), {num_val_tasks} val tasks")

    # Determine which benchmark to use
    if benchmark_name is None:
        benchmark_name = extract_benchmark_name(experiment_dir)
        print(f"Detected benchmark: {benchmark_name}")
    else:
        print(f"Using specified benchmark: {benchmark_name}")

    # Load system prompts (including actual prompt text), skipping candidate 0
    print("Loading system prompt candidates (skipping candidate 0)...")
    prog_candidates_dir = os.path.join(experiment_dir, "prog_candidates")
    # Pass num_candidates + 1 to account for the skipped candidate 0
    system_prompts = load_system_prompts(prog_candidates_dir, num_candidates + 1, skip_first=True)

    # Check how many loaded successfully
    loaded_count = sum(1 for p in system_prompts.values() if p.get('exists', False) and 'prompt' in p)
    print(f"Successfully loaded {loaded_count}/{num_candidates} system prompts")

    # Prepare output paths
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file with benchmark name
    suffix = output_path.suffix
    val_file = output_dir / f"{benchmark_name}_val{suffix}"

    # Construct VALIDATION dataset (only using val scores)
    print(f"\nLoading {benchmark_name} validation set...")
    valset = load_benchmark_dspy_dataset(benchmark_name=benchmark_name, seed=seed, split="val")

    val_data = construct_dataset_split(
        split_name="val",
        dataset=valset,
        prog_candidate_subscores=prog_candidate_val_subscores,
        system_prompts=system_prompts,
        num_candidates=num_candidates,
        candidate_offset=1  # We skipped candidate 0
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

    # Compute summary statistics
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

    # Compute statistics for validation split
    val_stats = compute_stats(val_data, "val")

    summary = {
        "seed": seed,
        "benchmark_name": benchmark_name,
        "experiment_dir": experiment_dir,
        "num_unique_candidates": num_candidates,
        "val": val_stats,
    }

    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset construction complete!")
    print(f"  Validation data: {val_file}")
    print(f"  Candidate metadata: {metadata_file}")
    print(f"  Summary: {summary_file}")

    print(f"\nSummary:")
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
        default=None,
        help="Path to experiment directory. If not specified, will be constructed from --benchmark_name and --seed"
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
        help="Random seed for dataset shuffling and experiment directory selection"
    )

    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="hoverBench",
        help="Name of the benchmark to use (e.g., 'hoverBench', 'HotpotQABench', 'IFBench'). "
             "If --experiment_dir is not specified, this will be used to construct the experiment path."
    )

    args = parser.parse_args()

    # If experiment_dir not provided, construct it from benchmark_name and seed
    if args.experiment_dir is None:
        args.experiment_dir = get_experiment_dir_for_benchmark(args.benchmark_name, args.seed)
        print(f"Using auto-constructed experiment directory: {args.experiment_dir}")

    construct_training_data(
        experiment_dir=args.experiment_dir,
        output_file=args.output_file,
        seed=args.seed,
        benchmark_name=args.benchmark_name
    )
