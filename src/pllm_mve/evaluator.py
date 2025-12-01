"""Evaluator D implementation."""

from typing import Optional
from .types import Transcript
from .together_client import TogetherChat


class EvaluatorD:
    """Evaluator that predicts preferences from transcript answers."""

    def __init__(
        self,
        chat_client: Optional[TogetherChat] = None,
        include_questions: bool = True,
        use_logprobs: bool = False
    ):
        """Initialize evaluator with chat client.

        Args:
            chat_client: TogetherChat client for API calls
            include_questions: If True, include Q&A in evaluation context (not just answers)
            use_logprobs: If True, use logprobs for probability. If False (default), ask LLM
                         to output probability as text (more widely supported, easier to debug)
        """
        self.chat = chat_client or TogetherChat()
        self.include_questions = include_questions
        self.use_logprobs = use_logprobs

    def prob_A_over_B(self, option_a: str, option_b: str, transcript: Transcript) -> float:
        """Compute P(A > B | transcript)."""
        if self.include_questions:
            context_text = transcript.format_qa_for_eval()
        else:
            context_text = transcript.format_answers_for_eval()

        if self.use_logprobs:
            try:
                return self.chat.get_eval_probability_logprobs(
                    answers=context_text,
                    option_a=option_a,
                    option_b=option_b,
                    include_questions=self.include_questions
                )
            except Exception as e:
                print(f"[WARN] Logprobs failed, falling back to text probability: {e}")
                return self.chat.get_eval_probability(
                    answers=context_text,
                    option_a=option_a,
                    option_b=option_b,
                    include_questions=self.include_questions
                )
        else:
            # Default: ask LLM to output probability as text
            return self.chat.get_eval_probability(
                answers=context_text,
                option_a=option_a,
                option_b=option_b,
                include_questions=self.include_questions
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