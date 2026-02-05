"""
Qwen Router using DSPy Language Model.

This router uses a language model to reason about which candidate system prompt
is best suited for a given task.
"""

import dspy
from typing import List, Dict, Union
import re
from router_base import Router


class CandidateSelection(dspy.Signature):
    """
    Given a task and multiple candidate system prompts, select the best one.
    """

    task_input = dspy.InputField(desc="The input task (e.g., question, claim, problem)")
    candidates_info = dspy.InputField(desc="Information about candidate system prompts with their indices")
    selected_candidate_idx = dspy.OutputField(desc="The index of the best candidate (just the number)")


class CandidateSelectionWithReasoning(dspy.Signature):
    """
    Given a task and multiple candidate system prompts, reason about and select the best one.
    """

    task_input = dspy.InputField(desc="The input task (e.g., question, claim, problem)")
    candidates_info = dspy.InputField(desc="Information about candidate system prompts with their indices")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning about which candidate is best")
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
        use_reasoning: bool = True,
        max_candidates_display: int = 10,
        summarize_prompts: bool = True,
        max_prompt_length: int = 200
    ):
        """
        Initialize the Qwen Router.

        Args:
            lm: DSPy language model. If None, uses the currently configured default LM.
            benchmark_name: Name of the benchmark
            use_reasoning: If True, use chain-of-thought reasoning
            max_candidates_display: Maximum number of candidates to show in full detail
            summarize_prompts: If True, truncate long prompts for efficiency
            max_prompt_length: Maximum length of each prompt to show
        """
        super().__init__(benchmark_name)

        self.lm = lm
        self.use_reasoning = use_reasoning
        self.max_candidates_display = max_candidates_display
        self.summarize_prompts = summarize_prompts
        self.max_prompt_length = max_prompt_length

        # Initialize the DSPy module
        if use_reasoning:
            self.predictor = dspy.ChainOfThought(CandidateSelectionWithReasoning)
        else:
            self.predictor = dspy.Predict(CandidateSelection)

        # Set the LM if provided
        if lm is not None:
            with dspy.context(lm=lm):
                pass

    def _format_candidates(self, candidates: List[Dict]) -> str:
        """
        Format candidates into a readable string for the LM.

        Args:
            candidates: List of candidate dicts

        Returns:
            Formatted string describing all candidates
        """
        formatted_lines = []

        # Limit number of candidates shown in detail
        num_to_show = min(len(candidates), self.max_candidates_display)

        for i, candidate in enumerate(candidates[:num_to_show]):
            prompt = candidate['candidate_system_prompt']

            # Optionally truncate long prompts
            if self.summarize_prompts and len(prompt) > self.max_prompt_length:
                prompt = prompt[:self.max_prompt_length] + "..."

            formatted_lines.append(
                f"Candidate {candidate['candidate_idx']}:\n{prompt}\n"
            )

        if len(candidates) > num_to_show:
            formatted_lines.append(
                f"\n... and {len(candidates) - num_to_show} more candidates (indices: "
                f"{', '.join(str(c['candidate_idx']) for c in candidates[num_to_show:])})"
            )

        return "\n".join(formatted_lines)

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

    def select_candidate(
        self,
        task_input: str,
        candidates: List[Dict],
        return_scores: bool = False
    ) -> Union[int, tuple]:
        """
        Use the LM to select the best candidate for a given task.

        Args:
            task_input: The input text for the task
            candidates: List of candidate dicts
            return_scores: If True, return scores (not implemented for LM-based router)

        Returns:
            candidate_idx or (candidate_idx, scores_dict)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        # Format candidates for the LM
        candidates_info = self._format_candidates(candidates)

        # Call the LM
        try:
            if self.lm is not None:
                with dspy.context(lm=self.lm):
                    prediction = self.predictor(
                        task_input=task_input,
                        candidates_info=candidates_info
                    )
            else:
                prediction = self.predictor(
                    task_input=task_input,
                    candidates_info=candidates_info
                )

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

    def batch_select_candidates(
        self,
        task_inputs: List[str],
        candidates_list: List[List[Dict]],
        return_scores: bool = False
    ) -> Union[List[int], tuple]:
        """
        Select candidates for multiple tasks.

        Note: Currently processes each task sequentially.
        Could be optimized with batch LM calls in the future.

        Args:
            task_inputs: List of input texts
            candidates_list: List of candidate lists
            return_scores: If True, return scores

        Returns:
            List of selected candidate indices or (indices, scores)
        """
        selected_indices = []
        all_scores = [] if return_scores else None

        for task_input, candidates in zip(task_inputs, candidates_list):
            if return_scores:
                idx, scores = self.select_candidate(task_input, candidates, return_scores=True)
                selected_indices.append(idx)
                all_scores.append(scores)
            else:
                idx = self.select_candidate(task_input, candidates, return_scores=False)
                selected_indices.append(idx)

        if return_scores:
            return selected_indices, all_scores
        return selected_indices

    def get_name(self) -> str:
        """Get the name of this router."""
        reasoning_str = "with_reasoning" if self.use_reasoning else "no_reasoning"
        lm_name = self.lm.model if self.lm and hasattr(self.lm, 'model') else 'default'
        return f"QwenRouter({lm_name},{reasoning_str})"
