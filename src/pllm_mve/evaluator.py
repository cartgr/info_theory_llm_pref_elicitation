"""Evaluator D implementation."""

from typing import Optional
from .types import Transcript
from .together_client import TogetherChat


class EvaluatorD:
    """Evaluator that predicts preferences from transcript answers."""

    def __init__(self, chat_client: Optional[TogetherChat] = None):
        """Initialize evaluator with chat client."""
        self.chat = chat_client or TogetherChat()

    def prob_A_over_B(self, option_a: str, option_b: str, transcript: Transcript) -> float:
        """Compute P(A > B | transcript) with fallback if logprobs unsupported."""
        answers_text = transcript.format_answers_for_eval()
        try:
            return self.chat.get_eval_probability_logprobs(
                answers=answers_text,
                option_a=option_a,
                option_b=option_b
            )
        except Exception as e:
            print(f"[WARN] Falling back to JSON probability route: {e}")
            return self.chat.get_eval_probability(
                answers=answers_text,
                option_a=option_a,
                option_b=option_b
            )


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