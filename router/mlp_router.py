"""
MLP Router for prompt selection.

This router uses a trained neural network (MLP) to score candidates and
output a probability distribution over them.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union, Optional
import numpy as np
from pathlib import Path

from router_base import Router


class MLPRouter(Router):
    """
    Router that uses a trained MLP/neural network to select candidates.

    The router encodes the task input and candidate prompts, then uses
    a neural network to compute scores and select the best candidate.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_class=None,
        backbone_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        benchmark_name: str = "hoverBench",
        device: str = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize the MLP Router.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            model_class: Model class to instantiate (if None, tries to import from model.py)
            backbone_name: Name of the backbone encoder model
            benchmark_name: Name of the benchmark
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing candidates
        """
        super().__init__(benchmark_name)

        self.checkpoint_path = checkpoint_path
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load tokenizer
        print(f"Loading tokenizer: {backbone_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # Load model
        if model_class is None:
            # Try to import from model.py
            try:
                from model import PromptRouterCrossEncoder
                model_class = PromptRouterCrossEncoder
            except ImportError:
                raise ValueError(
                    "Could not import model class. Please provide model_class argument "
                    "or ensure model.py is in the same directory."
                )

        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model
        # Try to get dropout from checkpoint args, otherwise use default
        dropout = 0.1
        if 'args' in checkpoint and 'dropout' in checkpoint['args']:
            dropout = checkpoint['args']['dropout']

        self.model = model_class(
            backbone_name=backbone_name,
            dropout=dropout
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded successfully on {self.device}")

    @torch.no_grad()
    def _score_candidates(
        self,
        task_input: str,
        candidates: List[Dict]
    ) -> Dict[int, float]:
        """
        Score all candidates for a given task.

        Args:
            task_input: The task input text
            candidates: List of candidate dicts

        Returns:
            Dict mapping candidate_idx to score
        """
        scores = {}

        # Process candidates in batches for efficiency
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]

            # Prepare batch
            task_inputs = [task_input] * len(batch_candidates)
            prompts = [c['candidate_system_prompt'] for c in batch_candidates]

            # Tokenize
            encoded = self.tokenizer(
                task_inputs,
                prompts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            token_type_ids = encoded.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # Get scores from model
            batch_scores = self.model(input_ids, attention_mask, token_type_ids)

            # Apply sigmoid to get probabilities
            batch_probs = torch.sigmoid(batch_scores).cpu().numpy().flatten()

            # Store scores
            for candidate, prob in zip(batch_candidates, batch_probs):
                scores[candidate['candidate_idx']] = float(prob)

        return scores

    def _scores_to_distribution(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Convert raw scores to a probability distribution using softmax.

        Args:
            scores: Dict mapping candidate_idx to score

        Returns:
            Dict mapping candidate_idx to probability
        """
        # Convert to numpy array
        indices = list(scores.keys())
        score_values = np.array([scores[idx] for idx in indices])

        # Apply softmax to get probability distribution
        # Use temperature=1 for standard softmax
        exp_scores = np.exp(score_values - np.max(score_values))  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)

        # Create distribution dict
        distribution = {idx: float(prob) for idx, prob in zip(indices, probs)}

        return distribution

    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Select the best candidate using the MLP.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            return_scores: If True, return (candidate_idx, probability_distribution)

        Returns:
            candidate_idx or (candidate_idx, probability_distribution)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        # Score all candidates
        scores = self._score_candidates(task_input, candidates)

        # Convert to probability distribution
        distribution = self._scores_to_distribution(scores)

        # Select candidate with highest probability
        best_candidate_idx = max(distribution.items(), key=lambda x: x[1])[0]

        if return_scores:
            return best_candidate_idx, distribution

        return best_candidate_idx

    @torch.no_grad()
    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Select candidates for multiple tasks in batch.

        Args:
            task_inputs: List of input texts
            candidates_list: List of candidate lists (one per task)
            return_scores: If True, return probability distributions

        Returns:
            List of selected candidate indices or (indices, distributions)
        """
        selected_indices = []
        all_distributions = [] if return_scores else None

        for task_input, candidates in zip(task_inputs, candidates_list):
            if return_scores:
                idx, distribution = self.select_candidate(
                    task_input, candidates, return_scores=True
                )
                selected_indices.append(idx)
                all_distributions.append(distribution)
            else:
                idx = self.select_candidate(
                    task_input, candidates, return_scores=False
                )
                selected_indices.append(idx)

        if return_scores:
            return selected_indices, all_distributions

        return selected_indices

    def get_name(self) -> str:
        """Get the name of this router."""
        checkpoint_name = Path(self.checkpoint_path).stem
        return f"MLPRouter({checkpoint_name})"

    def get_probability_distribution(
        self,
        task_input: str,
        candidates: List[Dict]
    ) -> Dict[int, float]:
        """
        Get the full probability distribution over candidates for a task.

        This is a convenience method that always returns the distribution.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts

        Returns:
            Dict mapping candidate_idx to probability
        """
        _, distribution = self.select_candidate(
            task_input, candidates, return_scores=True
        )
        return distribution

    def get_top_k_candidates(
        self,
        task_input: str,
        candidates: List[Dict],
        k: int = 3
    ) -> List[tuple]:
        """
        Get the top-k candidates according to the model's probability distribution.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            k: Number of top candidates to return

        Returns:
            List of (candidate_idx, probability) tuples, sorted by probability (descending)
        """
        distribution = self.get_probability_distribution(task_input, candidates)

        # Sort by probability
        sorted_candidates = sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_candidates[:k]
