"""Responder LLM for sampling hypothetical answers (distinct from PLLM)."""

from typing import Optional
from .together_client import TogetherChat

class ResponderLLM:
    """
    Separate LLM used to *sample answers* to candidate questions.
    This avoids using the same PLLM that provides ground truth labels.
    """

    def __init__(
        self,
        chat_client: Optional[TogetherChat] = None,
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature: float = 0.7,
        max_tokens: int = 64,
    ):
        self.chat = chat_client or TogetherChat(model=model, temperature=temperature, max_tokens=max_tokens)
        self._persona: Optional[str] = None

    def initialize_persona(self, persona_text: str) -> None:
        self._persona = persona_text

    def answer_question(self, question: str) -> str:
        """Sample one short, natural answer consistent with the initialized persona."""
        if not self._persona:
            raise RuntimeError("ResponderLLM not initialized with persona. Call initialize_persona(persona_text).")

        # Reuse TogetherChat.answer_question but keep this instance separate from PLLM
        return self.chat.answer_question(persona=self._persona, question=question)
