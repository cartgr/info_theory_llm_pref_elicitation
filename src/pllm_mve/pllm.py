"""Participant LLM (PLLM) implementation."""

from typing import Dict, Optional, List
from .types import EpisodeEvalSet
from .together_client import TogetherChat


class PLLM:
    """Participant LLM that simulates a user with a fixed persona."""

    def __init__(self, chat_client: Optional[TogetherChat] = None):
        """Initialize PLLM with chat client."""
        self.chat = chat_client or TogetherChat()
        self._persona: Optional[str] = None
        self._eval_set: Optional[EpisodeEvalSet] = None
        self._label_cache: Dict[tuple, int] = {}

    def initialize_persona(self, persona_text: str, eval_set: EpisodeEvalSet) -> None:
        """Initialize with persona and evaluation set."""
        self._persona = persona_text
        self._eval_set = eval_set
        self._label_cache = {}  # Clear cache for new persona

    def label_eval_question(self, i: int, j: int) -> int:
        """Get preference label for pair (i, j). Returns 1 if A preferred, else 0."""
        if not self._persona or not self._eval_set:
            raise RuntimeError("PLLM not initialized with persona and eval set")

        # Check cache first
        if (i, j) in self._label_cache:
            return self._label_cache[(i, j)]

        # Get items
        item_a = self._eval_set.items[i]
        item_b = self._eval_set.items[j]

        # Get preference from Together API
        label = self.chat.get_ab_preference(
            persona=self._persona,
            option_a=item_a,
            option_b=item_b
        )

        # Cache the result
        self._label_cache[(i, j)] = label
        return label

    def answer_question(self, question: str) -> str:
        """Answer a question as the persona."""
        if not self._persona:
            raise RuntimeError("PLLM not initialized with persona")

        return self.chat.answer_question(
            persona=self._persona,
            question=question
        )

    def sample_answers(self, question: str, num_samples: int = 3) -> List[str]:
        """Sample multiple answers to a question as the persona.

        For PLLM, we call answer_question multiple times since each call
        may produce slightly different answers due to temperature.
        """
        if not self._persona:
            raise RuntimeError("PLLM not initialized with persona")

        answers = []
        for _ in range(num_samples):
            answer = self.chat.answer_question(
                persona=self._persona,
                question=question
            )
            answers.append(answer)
        return answers

    def get_all_labels(self) -> Dict[tuple, int]:
        """Get labels for all pairs in the evaluation set."""
        if not self._eval_set:
            raise RuntimeError("PLLM not initialized with eval set")

        labels = {}
        for pair in self._eval_set.pairs:
            i, j = pair
            labels[pair] = self.label_eval_question(i, j)

        return labels