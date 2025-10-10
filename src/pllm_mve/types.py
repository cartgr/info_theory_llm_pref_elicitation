"""Core data types for the PLLM MVE."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

Item = str
Pair = Tuple[int, int]


@dataclass
class EpisodeEvalSet:
    """Evaluation set for an episode."""
    items: List[Item]
    pairs: List[Pair]
    labels: Dict[Pair, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pairs are within item bounds."""
        if self.pairs:
            max_idx = max(max(pair) for pair in self.pairs)
            if max_idx >= len(self.items):
                raise ValueError(f"Pair index {max_idx} out of bounds for {len(self.items)} items")


@dataclass
class Turn:
    """A single turn in the dialogue."""
    question: str
    answer: str


@dataclass
class Transcript:
    """Ordered list of dialogue turns."""
    turns: List[Turn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        """Add a new turn to the transcript."""
        self.turns.append(Turn(question=question, answer=answer))

    def get_answers_only(self) -> List[str]:
        """Extract only the answers from the transcript."""
        return [turn.answer for turn in self.turns]

    def format_answers_for_eval(self) -> str:
        """Format answers for evaluator consumption."""
        if not self.turns:
            return "No answers yet."
        return "\n".join([f"- {answer}" for answer in self.get_answers_only()])


@dataclass
class EpisodeConfig:
    """Configuration for an episode."""
    persona: str
    domain: str
    num_items: int = 10
    num_pairs: int = 20
    num_turns: int = 5
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    temperature: float = 0.1
    max_tokens: int = 256
    seed: Optional[int] = None