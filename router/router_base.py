"""
Base Router API for prompt routing.

This module defines the interface that all routers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import numpy as np


class Router(ABC):
    """
    Abstract base class for all routers.

    A router takes a task input and a list of candidate system prompts,
    and selects the best candidate for that task.
    """

    def __init__(self, benchmark_name: str = "hoverBench"):
        """
        Initialize the router.

        Args:
            benchmark_name: Name of the benchmark (e.g., 'hoverBench', 'HotpotQABench')
        """
        self.benchmark_name = benchmark_name

        # Mapping of benchmark names to their input field names
        self.BENCHMARK_INPUT_FIELDS = {
            'hoverBench': 'claim',
            'HotpotQABench': 'question',
            'IFBench': 'prompt',
            'AIMEBench': 'problem',
            'Papillon': 'user_query',
        }

        self.input_field = self.BENCHMARK_INPUT_FIELDS.get(benchmark_name, 'claim')

    @abstractmethod
    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Select the best candidate for a given task.

        Args:
            task_input: The input text for the task (e.g., claim, question)
            candidates: List of candidate dicts, each containing:
                - candidate_idx: int
                - candidate_system_prompt: str
                - (optional) reward: float
                - (optional) in_pareto_frontier: bool
            return_scores: If True, return (candidate_idx, scores_dict)

        Returns:
            If return_scores is False:
                candidate_idx: int - Index of the selected candidate
            If return_scores is True:
                tuple: (candidate_idx, scores_dict) where scores_dict maps candidate_idx to score
        """
        pass

    @abstractmethod
    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Select the best candidate for multiple tasks in batch.

        Args:
            task_inputs: List of input texts
            candidates_list: List of candidate lists (one per task)
            return_scores: If True, return (candidate_indices, all_scores)

        Returns:
            If return_scores is False:
                List[int] - Selected candidate indices
            If return_scores is True:
                tuple: (List[int], List[Dict]) - indices and scores for each task
        """
        pass

    def extract_task_input(self, task_data: Dict) -> str:
        """
        Extract the task input text from a task data dict.

        Args:
            task_data: Dict containing task fields

        Returns:
            str: The input text for the task
        """
        return task_data.get(self.input_field, '')

    def get_name(self) -> str:
        """Get the name of this router."""
        return self.__class__.__name__


class RandomRouter(Router):
    """
    Dummy router that randomly selects a candidate.
    Useful for baseline comparisons and testing.
    """

    def __init__(self, benchmark_name: str = "hoverBench", seed: int = 42):
        """
        Initialize the random router.

        Args:
            benchmark_name: Name of the benchmark
            seed: Random seed for reproducibility
        """
        super().__init__(benchmark_name)
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Randomly select a candidate.

        Args:
            task_input: The input text (ignored for random selection)
            candidates: List of candidate dicts
            return_scores: If True, return random scores for all candidates

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        # Randomly select a candidate
        selected_idx = self.rng.choice(len(candidates))
        selected_candidate_idx = candidates[selected_idx]['candidate_idx']

        if return_scores:
            # Generate random scores for all candidates
            scores = {}
            for candidate in candidates:
                scores[candidate['candidate_idx']] = self.rng.random()
            return selected_candidate_idx, scores

        return selected_candidate_idx

    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Randomly select candidates for multiple tasks.

        Args:
            task_inputs: List of input texts (ignored)
            candidates_list: List of candidate lists
            return_scores: If True, return random scores

        Returns:
            List of selected candidate indices or (indices, scores)
        """
        selected_indices = []
        all_scores = [] if return_scores else None

        for candidates in candidates_list:
            if return_scores:
                idx, scores = self.select_candidate("", candidates, return_scores=True)
                selected_indices.append(idx)
                all_scores.append(scores)
            else:
                idx = self.select_candidate("", candidates, return_scores=False)
                selected_indices.append(idx)

        if return_scores:
            return selected_indices, all_scores
        return selected_indices

    def get_name(self) -> str:
        """Get the name of this router."""
        return f"RandomRouter(seed={self.seed})"


class OracleRouter(Router):
    """
    Oracle router that always selects the best candidate based on ground truth rewards.
    Useful for computing upper bound performance.
    """

    def __init__(self, benchmark_name: str = "hoverBench"):
        """
        Initialize the oracle router.

        Args:
            benchmark_name: Name of the benchmark
        """
        super().__init__(benchmark_name)

    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Select the candidate with the highest reward.

        Args:
            task_input: The input text (ignored)
            candidates: List of candidate dicts with 'reward' field
            return_scores: If True, return the rewards as scores

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        # Find candidate with highest reward
        best_candidate = max(candidates, key=lambda c: c.get('reward', 0.0))
        selected_candidate_idx = best_candidate['candidate_idx']

        if return_scores:
            # Use rewards as scores
            scores = {c['candidate_idx']: c.get('reward', 0.0) for c in candidates}
            return selected_candidate_idx, scores

        return selected_candidate_idx

    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Select best candidates for multiple tasks.

        Args:
            task_inputs: List of input texts (ignored)
            candidates_list: List of candidate lists
            return_scores: If True, return rewards as scores

        Returns:
            List of selected candidate indices or (indices, scores)
        """
        selected_indices = []
        all_scores = [] if return_scores else None

        for candidates in candidates_list:
            if return_scores:
                idx, scores = self.select_candidate("", candidates, return_scores=True)
                selected_indices.append(idx)
                all_scores.append(scores)
            else:
                idx = self.select_candidate("", candidates, return_scores=False)
                selected_indices.append(idx)

        if return_scores:
            return selected_indices, all_scores
        return selected_indices
