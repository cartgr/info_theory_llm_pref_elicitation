"""Responder LLM for sampling hypothetical answers (distinct from PLLM)."""

import re
from typing import Optional, List
from .together_client import TogetherChat

class ResponderLLM:
    """
    Separate LLM used to *sample answers* to candidate questions.

    IMPORTANT: This should NOT use the same persona as PLLM. The purpose is to
    generate diverse hypothetical answers that could come from various users,
    not to simulate the specific persona being evaluated.

    Supports batched sampling: generate multiple answers in one API call.
    """

    def __init__(
        self,
        chat_client: Optional[TogetherChat] = None,
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature: float = 0.8,
        max_tokens: int = 256,
        use_persona: bool = False,
    ):
        self.chat = chat_client or TogetherChat(model=model, temperature=temperature, max_tokens=max_tokens)
        self._persona: Optional[str] = None
        self._domain: Optional[str] = None
        self.use_persona = use_persona

    def initialize(self, domain: str, persona: Optional[str] = None) -> None:
        """Initialize responder with domain (and optionally persona for backward compat)."""
        self._domain = domain
        self._persona = persona

    def initialize_persona(self, persona_text: str) -> None:
        """Legacy method for backward compatibility."""
        self._persona = persona_text

    def sample_answers(self, question: str, num_samples: int = 3) -> List[str]:
        """Sample multiple diverse answers to a question in ONE API call.

        Args:
            question: The question to answer
            num_samples: Number of diverse answers to generate

        Returns:
            List of answer strings
        """
        if self._domain:
            system = (
                f"You are simulating {num_samples} different people being interviewed about {self._domain}. "
                f"Each person has different preferences and backgrounds. "
                f"Generate {num_samples} distinct, realistic answers to the question below.\n\n"
                f"Format your response EXACTLY like this:\n"
                f"[1] First person's answer here\n"
                f"[2] Second person's answer here\n"
                f"...\n\n"
                f"Each answer should be ~20 words, natural and conversational."
            )
        else:
            system = (
                f"You are simulating {num_samples} different people being interviewed. "
                f"Each person has different preferences and backgrounds. "
                f"Generate {num_samples} distinct, realistic answers to the question below.\n\n"
                f"Format your response EXACTLY like this:\n"
                f"[1] First person's answer here\n"
                f"[2] Second person's answer here\n"
                f"...\n\n"
                f"Each answer should be ~20 words, natural and conversational."
            )

        user = f"Question: {question}\n\nGenerate {num_samples} different answers:"

        response = self.chat.chat(
            system=system,
            user=user,
            temperature=0.9,  # High temperature for diversity
            max_tokens=50 * num_samples + 50  # ~50 tokens per answer + buffer
        )

        # Parse the numbered responses
        answers = self._parse_numbered_responses(response, num_samples)

        # If parsing failed, fall back to splitting by newlines
        if len(answers) < num_samples:
            answers = self._fallback_parse(response, num_samples)

        return answers

    def _parse_numbered_responses(self, response: str, num_samples: int) -> List[str]:
        """Parse responses in [1], [2], etc. format."""
        answers = []

        # Match patterns like [1], [2], (1), (2), 1., 1), etc.
        pattern = r'[\[\(]?(\d+)[\]\)\.]?\s*(.+?)(?=[\[\(]?\d+[\]\)\.]|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for num, text in matches:
            text = text.strip()
            # Clean up the text
            text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
            text = text.strip()
            if text and len(text) > 5:  # Skip empty or too-short responses
                answers.append(text)

        return answers[:num_samples]

    def _fallback_parse(self, response: str, num_samples: int) -> List[str]:
        """Fallback: split by newlines and clean up."""
        lines = response.strip().split('\n')
        answers = []

        for line in lines:
            line = line.strip()
            # Remove numbering prefixes
            line = re.sub(r'^[\[\(]?\d+[\]\)\.\:\-]?\s*', '', line)
            line = line.strip()
            if line and len(line) > 10:
                answers.append(line)

        # If we still don't have enough, duplicate the last one or use placeholder
        while len(answers) < num_samples:
            if answers:
                answers.append(answers[-1])
            else:
                answers.append("I'm not sure, it depends on what I'm looking for.")

        return answers[:num_samples]

    def answer_question(self, question: str) -> str:
        """Sample one answer (for backward compatibility).

        Note: For efficiency, prefer sample_answers() to batch multiple samples.
        """
        answers = self.sample_answers(question, num_samples=1)
        return answers[0] if answers else "I'm not sure."
