"""Evaluator D implementation."""

from typing import Optional
from .types import Transcript
from .together_client import TogetherChat


class EvaluatorD:
    """Evaluator that predicts preferences from transcript answers."""

    def __init__(self, chat_client: Optional[TogetherChat] = None):
        """Initialize evaluator with chat client."""
        self.chat = chat_client or TogetherChat()

    def prob_A_over_B(
        self,
        option_a: str,
        option_b: str,
        transcript: Transcript
    ) -> float:
        """Compute P(A > B | transcript)."""
        # Format answers for evaluation
        answers_text = transcript.format_answers_for_eval()

        # Get probability using logprobs method (more efficient and accurate)
        prob = self.chat.get_eval_probability_logprobs(
            answers=answers_text,
            option_a=option_a,
            option_b=option_b
        )

        return prob

    def evaluate_pair(
        self,
        i: int,
        j: int,
        items: list,
        transcript: Transcript
    ) -> float:
        """Evaluate a specific pair given items and transcript."""
        option_a = items[i]
        option_b = items[j]
        return self.prob_A_over_B(option_a, option_b, transcript)