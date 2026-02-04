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

# Add gepa_artifact to path for loading pickle files
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


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


def construct_training_data(
    experiment_dir: str,
    output_file: str,
    seed: int = 0,
    strategy: str = "all_candidates"
):
    """
    Construct training dataset for prompt router.

    Args:
        experiment_dir: Path to the experiment directory
        output_file: Path to save the training data
        seed: Random seed for dataset shuffling
        strategy: Strategy for creating training examples
                 - "all_candidates": Include all (task, candidate, reward) tuples
                 - "pareto_only": Only include candidates in pareto frontier for each task
                 - "best_from_pareto": One example per task with best candidate (legacy)
    """
    print(f"Loading GEPA state from {experiment_dir}...")
    state_path = os.path.join(experiment_dir, "gepa_state.bin")
    state = load_gepa_state(state_path)

    # Extract relevant data from state
    pareto_frontiers = state['program_at_pareto_front_valset']  # List of sets
    prog_candidate_val_subscores = state['prog_candidate_val_subscores']  # List[List]
    num_candidates = len(prog_candidate_val_subscores)
    num_tasks = len(pareto_frontiers)

    print(f"Found {num_candidates} candidates and {num_tasks} tasks")

    # Load the validation set
    print("Loading HoVer validation set...")
    valset = load_hover_valset(seed=seed)

    if len(valset) != num_tasks:
        print(f"Warning: Valset size ({len(valset)}) doesn't match number of tasks ({num_tasks})")
        print(f"Using first {min(len(valset), num_tasks)} tasks")
        num_tasks = min(len(valset), num_tasks)

    # Load system prompts (including actual prompt text)
    print("Loading system prompt candidates...")
    prog_candidates_dir = os.path.join(experiment_dir, "prog_candidates")
    system_prompts = load_system_prompts(prog_candidates_dir, num_candidates)

    # Check how many loaded successfully
    loaded_count = sum(1 for p in system_prompts.values() if p.get('exists', False) and 'prompt' in p)
    print(f"Successfully loaded {loaded_count}/{num_candidates} system prompts")

    # Construct training examples as (task, candidate, reward) tuples
    training_data = []

    print("Constructing training examples...")
    for task_idx in tqdm.tqdm(range(num_tasks)):
        task = valset[task_idx]
        pareto_frontier = pareto_frontiers[task_idx]

        if strategy == "all_candidates":
            # Create one example for each (task, candidate) pair
            for candidate_idx in range(num_candidates):
                # Skip if prompt wasn't loaded successfully
                if candidate_idx not in system_prompts or not system_prompts[candidate_idx].get('exists', False):
                    continue
                if 'prompt' not in system_prompts[candidate_idx]:
                    continue

                reward = prog_candidate_val_subscores[candidate_idx][task_idx]
                # Convert boolean to float
                if isinstance(reward, bool):
                    reward = float(reward)
                elif reward == 0:
                    reward = 0.0
                else:
                    reward = float(reward)

                training_example = {
                    "task_idx": task_idx,
                    "claim": task["claim"],
                    "label": task["label"],
                    "supporting_facts": task["supporting_facts"],
                    "candidate_idx": candidate_idx,
                    "candidate_prompt": system_prompts[candidate_idx]["prompt"],
                    "reward": reward,
                    "in_pareto_frontier": candidate_idx in pareto_frontier,
                }
                training_data.append(training_example)

        elif strategy == "pareto_only":
            # Only include candidates in pareto frontier
            for candidate_idx in pareto_frontier:
                # Skip if prompt wasn't loaded successfully
                if candidate_idx not in system_prompts or not system_prompts[candidate_idx].get('exists', False):
                    continue
                if 'prompt' not in system_prompts[candidate_idx]:
                    continue

                reward = prog_candidate_val_subscores[candidate_idx][task_idx]
                if isinstance(reward, bool):
                    reward = float(reward)
                elif reward == 0:
                    reward = 0.0
                else:
                    reward = float(reward)

                training_example = {
                    "task_idx": task_idx,
                    "claim": task["claim"],
                    "label": task["label"],
                    "supporting_facts": task["supporting_facts"],
                    "candidate_idx": candidate_idx,
                    "candidate_prompt": system_prompts[candidate_idx]["prompt"],
                    "reward": reward,
                    "in_pareto_frontier": True,
                }
                training_data.append(training_example)

        elif strategy == "best_from_pareto":
            # Legacy: Select single best candidate from pareto frontier
            best_candidate_idx = select_best_candidate_for_task(
                task_idx, pareto_frontier, prog_candidate_val_subscores
            )

            training_example = {
                "task_idx": task_idx,
                "claim": task["claim"],
                "label": task["label"],
                "supporting_facts": task["supporting_facts"],
                "pareto_frontier": sorted(list(pareto_frontier)),
                "selected_candidate": best_candidate_idx,
                "selected_candidate_score": prog_candidate_val_subscores[best_candidate_idx][task_idx],
                "all_candidate_scores": {
                    cand_idx: prog_candidate_val_subscores[cand_idx][task_idx]
                    for cand_idx in pareto_frontier
                }
            }
            training_data.append(training_example)

    # Save training data
    print(f"Saving {len(training_data)} training examples to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    # Save system prompt metadata separately for reference (without full prompt text to save space)
    metadata_file = output_path.parent / "candidate_metadata.json"
    metadata_only = {
        idx: {k: v for k, v in prompt_data.items() if k != 'prompt'}
        for idx, prompt_data in system_prompts.items()
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata_only, f, indent=2)

    # Compute summary statistics
    if strategy in ["all_candidates", "pareto_only"]:
        # Compute reward statistics
        rewards = [ex["reward"] for ex in training_data]
        reward_mean = sum(rewards) / len(rewards) if rewards else 0
        reward_positive = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0

        summary = {
            "num_tasks": num_tasks,
            "num_candidates": num_candidates,
            "num_training_examples": len(training_data),
            "strategy": strategy,
            "seed": seed,
            "experiment_dir": experiment_dir,
            "reward_statistics": {
                "mean": reward_mean,
                "positive_ratio": reward_positive,
                "total_positive": sum(1 for r in rewards if r > 0),
                "total_negative": sum(1 for r in rewards if r == 0),
            },
            "examples_per_task": len(training_data) / num_tasks if num_tasks > 0 else 0,
        }
    else:
        summary = {
            "num_tasks": num_tasks,
            "num_candidates": num_candidates,
            "num_training_examples": len(training_data),
            "strategy": strategy,
            "seed": seed,
            "experiment_dir": experiment_dir,
            "pareto_frontier_sizes": {
                task_idx: len(pareto_frontiers[task_idx])
                for task_idx in range(min(10, num_tasks))  # First 10 as example
            }
        }

    summary_file = output_path.parent / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset construction complete!")
    print(f"  Training data: {output_file}")
    print(f"  Candidate metadata: {metadata_file}")
    print(f"  Summary: {summary_file}")
    print(f"\nSummary:")
    print(f"  Total tasks: {num_tasks}")
    print(f"  Total candidates: {num_candidates}")
    print(f"  Training examples: {len(training_data)}")

    if strategy in ["all_candidates", "pareto_only"]:
        print(f"  Examples per task: {len(training_data) / num_tasks:.1f}")
        print(f"  Mean reward: {summary['reward_statistics']['mean']:.4f}")
        print(f"  Positive ratio: {summary['reward_statistics']['positive_ratio']:.4f}")
    else:
        print(f"  Average pareto frontier size: {sum(len(pf) for pf in pareto_frontiers[:num_tasks]) / num_tasks:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construct training data for prompt router")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_GEPA_Qwen3-8B",
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
        "--strategy",
        type=str,
        choices=["all_candidates", "pareto_only", "best_from_pareto"],
        default="all_candidates",
        help="Strategy for creating training examples: "
             "all_candidates (all task-candidate pairs), "
             "pareto_only (only pareto frontier candidates), "
             "best_from_pareto (legacy, one per task)"
    )

    args = parser.parse_args()

    construct_training_data(
        experiment_dir=args.experiment_dir,
        output_file=args.output_file,
        seed=args.seed,
        strategy=args.strategy
    )
