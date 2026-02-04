"""
Script to analyze the training dataset for prompt router.
"""

import json
from collections import Counter, defaultdict
import numpy as np


def analyze_training_data(data_file: str = "router/training_data.json"):
    """Analyze the training dataset and print statistics."""

    with open(data_file) as f:
        data = json.load(f)

    print("=" * 80)
    print("TRAINING DATASET ANALYSIS")
    print("=" * 80)

    # Basic statistics
    print(f"\n{'Total examples:':<40} {len(data)}")

    # Label distribution
    label_counts = Counter(example['label'] for example in data)
    print(f"\n{'Label distribution:':<40}")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(data) * 100
        print(f"  Label {label}: {count:>4} ({percentage:>5.1f}%)")

    # Candidate selection frequency
    candidate_counts = Counter(example['selected_candidate'] for example in data)
    print(f"\n{'Selected candidate frequency:':<40}")
    print(f"  {'Candidate':<12} {'Count':<8} {'Percentage'}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    for candidate, count in sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(data) * 100
        print(f"  {candidate:<12} {count:<8} {percentage:>5.1f}%")

    # Pareto frontier statistics
    pareto_sizes = [len(example['pareto_frontier']) for example in data]
    print(f"\n{'Pareto frontier size statistics:':<40}")
    print(f"  Mean: {np.mean(pareto_sizes):.2f}")
    print(f"  Median: {np.median(pareto_sizes):.2f}")
    print(f"  Min: {min(pareto_sizes)}")
    print(f"  Max: {max(pareto_sizes)}")
    print(f"  Std: {np.std(pareto_sizes):.2f}")

    # Distribution of pareto frontier sizes
    size_counts = Counter(pareto_sizes)
    print(f"\n{'Pareto frontier size distribution:':<40}")
    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        percentage = count / len(data) * 100
        bar = '#' * int(percentage / 2)
        print(f"  Size {size:>2}: {count:>3} ({percentage:>5.1f}%) {bar}")

    # Score distribution
    all_scores = []
    for example in data:
        score = example['selected_candidate_score']
        if isinstance(score, bool):
            score = int(score)
        all_scores.append(score)

    score_counts = Counter(all_scores)
    print(f"\n{'Selected candidate score distribution:':<40}")
    for score, count in sorted(score_counts.items()):
        percentage = count / len(data) * 100
        print(f"  Score {score}: {count:>4} ({percentage:>5.1f}%)")

    # Candidate performance across tasks
    candidate_success_rate = defaultdict(list)
    for example in data:
        for candidate, score in example['all_candidate_scores'].items():
            if isinstance(score, bool):
                score = int(score)
            candidate_success_rate[int(candidate)].append(score)

    print(f"\n{'Candidate performance (from Pareto frontiers):':<40}")
    print(f"  {'Candidate':<12} {'Appearances':<12} {'Success Rate'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for candidate in sorted(candidate_success_rate.keys()):
        scores = candidate_success_rate[candidate]
        appearances = len(scores)
        success_rate = np.mean(scores) * 100
        print(f"  {candidate:<12} {appearances:<12} {success_rate:>5.1f}%")

    # Claim length statistics
    claim_lengths = [len(example['claim'].split()) for example in data]
    print(f"\n{'Claim length statistics (words):':<40}")
    print(f"  Mean: {np.mean(claim_lengths):.2f}")
    print(f"  Median: {np.median(claim_lengths):.2f}")
    print(f"  Min: {min(claim_lengths)}")
    print(f"  Max: {max(claim_lengths)}")

    # Supporting facts statistics
    num_supporting_facts = [len(example['supporting_facts']) for example in data]
    print(f"\n{'Number of supporting facts:':<40}")
    print(f"  Mean: {np.mean(num_supporting_facts):.2f}")
    print(f"  Min: {min(num_supporting_facts)}")
    print(f"  Max: {max(num_supporting_facts)}")

    # Correlation between label and best candidate
    print(f"\n{'Candidate selection by label:':<40}")
    label_to_candidates = defaultdict(list)
    for example in data:
        label_to_candidates[example['label']].append(example['selected_candidate'])

    for label in sorted(label_to_candidates.keys()):
        candidates = label_to_candidates[label]
        top_candidates = Counter(candidates).most_common(5)
        print(f"  Label {label} - Top candidates:")
        for candidate, count in top_candidates:
            percentage = count / len(candidates) * 100
            print(f"    Candidate {candidate}: {count:>3} ({percentage:>5.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze training data for prompt router")
    parser.add_argument(
        "--data_file",
        type=str,
        default="router/training_data.json",
        help="Path to training data JSON file"
    )

    args = parser.parse_args()
    analyze_training_data(args.data_file)
