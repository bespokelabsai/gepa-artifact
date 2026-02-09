"""
Qwen Router using DSPy Language Model.

This router uses a language model to reason about which candidate system prompt
is best suited for a given task.
"""

import dspy
from typing import List, Dict, Union
import re
import asyncio
from router_base import Router


class CandidateSelection(dspy.Signature):
    """
    Given a task and multiple candidate system prompts, select the best one.
    """

    task_input = dspy.InputField(desc="The input task (e.g., question, claim, problem)")
    candidates: List[str] = dspy.InputField(desc="List of candidate system prompts")
    selected_candidate_idx = dspy.OutputField(desc="The index of the best candidate (just the number)")




class QwenRouter(Router):
    """
    Router that uses DSPy's language model to select the best candidate.

    This router formats the task and candidates, sends them to a language model,
    and parses the model's selection.
    """

    def __init__(
        self,
        lm: dspy.LM = None,
        benchmark_name: str = "hoverBench",
        use_reasoning: bool = False
    ):
        """
        Initialize the Qwen Router.

        Args:
            lm: DSPy language model. If None, uses the currently configured default LM.
            benchmark_name: Name of the benchmark
            use_reasoning: If True, use chain-of-thought reasoning
        """
        super().__init__(benchmark_name)
        assert lm is not None, "LM must be provided"

        self.lm = lm

        # Initialize the DSPy module
        self.predictor = dspy.Predict(CandidateSelection)

        self.predictor.set_lm(lm)
        

    def _parse_selection(self, output: str, candidates: List[Dict]) -> int:
        """
        Parse the LM's output to extract the selected candidate index.

        Args:
            output: The LM's output string
            candidates: List of candidate dicts

        Returns:
            Selected candidate index
        """
        # Try to extract a number from the output
        # Look for patterns like "Candidate 5", "index 5", "5", etc.

        # First, try to find "Candidate X" or "candidate X"
        match = re.search(r'[Cc]andidate\s+(\d+)', output)
        if match:
            idx = int(match.group(1))
            # Verify this is a valid candidate index
            valid_indices = [c['candidate_idx'] for c in candidates]
            if idx in valid_indices:
                return idx

        # Try to find just a number
        numbers = re.findall(r'\b(\d+)\b', output)
        if numbers:
            # Try each number and see if it's a valid candidate index
            valid_indices = set(c['candidate_idx'] for c in candidates)
            for num_str in numbers:
                idx = int(num_str)
                if idx in valid_indices:
                    return idx

        # If we can't parse, return the first candidate as fallback
        print(f"Warning: Could not parse candidate selection from output: {output[:100]}")
        return candidates[0]['candidate_idx']

    async def _select_candidate_impl(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Internal async implementation for selecting a candidate.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            return_scores: If True, return scores

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        # Format candidates for the LM
        candidates_str = [f"{idx}.\n{candidate['candidate_system_prompt']}" for idx, candidate in enumerate(candidates)]

        # Call the LM - wrap synchronous DSPy call to make it non-blocking
        try:
            prediction = await self.predictor.acall(
                task_input=task_input,
                candidates=candidates_str
            )
            print(f"Prediction: {prediction}")
            # Parse the selection
            selected_idx = self._parse_selection(
                prediction.selected_candidate_idx,
                candidates
            )

        except Exception as e:
            print(f"Error during LM selection: {e}")
            # Fallback to first candidate
            selected_idx = candidates[0]['candidate_idx']

        if return_scores:
            # For LM-based router, we don't have numerical scores
            # Return uniform scores or None
            scores = {c['candidate_idx']: 0.5 for c in candidates}
            scores[selected_idx] = 1.0  # Selected candidate gets score of 1
            return selected_idx, scores

        return selected_idx

    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Use the LM to select the best candidate for a given task.

        This is a synchronous wrapper that runs the async implementation.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            return_scores: If True, return scores

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        # Run the async implementation synchronously
        return asyncio.run(self._select_candidate_impl(task_input, candidates, return_scores))

    async def select_candidate_async(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Async version: Use the LM to select the best candidate for a given task.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            return_scores: If True, return scores

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        return await self._select_candidate_impl(task_input, candidates, return_scores)

    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Select candidates for multiple tasks.

        This is a synchronous wrapper for compatibility with the base Router interface.
        For better performance, use batch_select_candidates_async instead.

        Args:
            task_inputs: List of input texts
            candidates_list: List of candidate lists
            return_scores: If True, return scores

        Returns:
            List of selected candidate indices or (indices, scores)
        """
        # Run the async batch method in a single event loop for efficiency
        return asyncio.run(
            self.batch_select_candidates_async(
                task_inputs, candidates_list, return_scores, max_concurrent=10
            )
        )

    async def batch_select_candidates_async(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False,
        max_concurrent: int = 10
    ) -> Union[List[int], tuple]:
        """
        Async version: Select candidates for multiple tasks in parallel.

        Args:
            task_inputs: List of input texts
            candidates_list: List of candidate lists
            return_scores: If True, return scores
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of selected candidate indices or (indices, scores)
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def select_with_semaphore(task_input, candidates):
            async with semaphore:
                return await self.select_candidate_async(task_input, candidates, return_scores)

        # Run all selections concurrently
        results = await asyncio.gather(*[
            select_with_semaphore(task_input, candidates)
            for task_input, candidates in zip(task_inputs, candidates_list)
        ])

        if return_scores:
            selected_indices = [r[0] for r in results]
            all_scores = [r[1] for r in results]
            return selected_indices, all_scores
        else:
            return results

    def get_name(self) -> str:
        """Get the name of this router."""
        reasoning_str = "with_reasoning" if self.use_reasoning else "no_reasoning"
        lm_name = self.lm.model if self.lm and hasattr(self.lm, 'model') else 'default'
        return f"QwenRouter({lm_name},{reasoning_str})"
