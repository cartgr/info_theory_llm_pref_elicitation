"""Online reward computation for GRPO training."""

import sys
import os
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pllm_mve.types import Transcript, Turn, EpisodeEvalSet
from src.pllm_mve.pllm import PLLM
from src.pllm_mve.evaluator import EvaluatorD
from src.pllm_mve.together_client import TogetherChat
from src.pllm_mve.eval_items import get_car_items, generate_item_pairs
from src.pllm_mve.responder import ResponderLLM


@dataclass
class RewardContext:
    """Context needed for reward computation."""
    persona: str
    domain: str
    items: List[str]
    pairs: List[tuple]
    pllm_labels: Dict[tuple, int]
    current_transcript: Transcript


class OnlineRewardCalculator:
    """Calculate information-theoretic rewards online during training."""

    def __init__(
        self,
        persona: str = "likes fast cars but is on a budget",
        domain: str = "cars",
        num_items: int = 10,
        num_pairs: int = 20,
        seed: int = 42
    ):
        """Initialize with fixed evaluation setup."""
        self.persona = persona
        self.domain = domain

        # Initialize components
        print("Initializing online reward calculator...")
        self.chat = TogetherChat(temperature=0.1)
        self.pllm = PLLM(chat_client=self.chat)
        self.evaluator = EvaluatorD(chat_client=self.chat)

        self.responder = ResponderLLM(
            chat_client=TogetherChat(temperature=0.7)  # distinct instance/temperature
        )
        self.responder.initialize_persona(persona)


        # Set up evaluation items
        if domain == "cars":
            self.items = get_car_items()[:num_items]
        else:
            raise ValueError(f"Domain {domain} not implemented")

        self.pairs = generate_item_pairs(len(self.items), num_pairs, seed)

        # Initialize PLLM with persona
        eval_set = EpisodeEvalSet(items=self.items, pairs=self.pairs)
        self.pllm.initialize_persona(persona, eval_set)

        # Collect PLLM labels once
        print("Collecting PLLM preference labels...")
        self.pllm_labels = self._collect_pllm_labels()

        # Track current transcript state
        self.current_transcript = Transcript()

    def _collect_pllm_labels(self) -> Dict[tuple, int]:
        """Collect PLLM preference labels for all pairs."""
        labels = {}
        for pair in self.pairs:
            i, j = pair
            label = self.pllm.label_eval_question(i, j)
            labels[pair] = label
        return labels

    def compute_log_score(self, transcript: Transcript) -> float:
        """Compute log-likelihood score F(S) for a transcript."""
        if not transcript.turns:
            return 0.0

        log_score = 0.0
        for pair in self.pairs:
            i, j = pair
            # Get evaluator prediction
            prob_a = self.evaluator.evaluate_pair(i, j, self.items, transcript)
            # Get true label
            true_label = self.pllm_labels[pair]
            # Compute log likelihood
            if true_label == 1:  # A preferred
                log_score += np.log(max(prob_a, 1e-10))
            else:  # B preferred
                log_score += np.log(max(1 - prob_a, 1e-10))

        return log_score / len(self.pairs)  # Normalize

    def compute_rewards_for_questions(
        self,
        questions: List[str],
        base_transcript: Optional[Transcript] = None
    ) -> List[float]:
        """
        Compute rewards for generated questions.

        This is called by GRPO during training for each batch of generated questions.
        """
        if base_transcript is None:
            base_transcript = self.current_transcript

        rewards = []
        base_score = self.compute_log_score(base_transcript)

        for i, question in enumerate(questions):
            try:
                # Clean the question (remove prompt if still there)
                question = self._clean_generated_question(question)

                print(f"Computing reward for question {i+1}/{len(questions)}: {question[:50]}...")

                # Get PLLM answer
                answer = self.responder.answer_question(question)

                # Create new transcript with this Q&A
                new_transcript = Transcript(turns=base_transcript.turns.copy())
                new_transcript.add_turn(question, answer)

                # Compute information gain
                new_score = self.compute_log_score(new_transcript)
                reward = new_score - base_score

                rewards.append(reward)

                print(f"  Answer: {answer[:50]}...")
                print(f"  Reward: {reward:.4f}")

            except Exception as e:
                print(f"Error computing reward for question {i}: {e}")
                rewards.append(0.0)  # Default to neutral reward on error

        return rewards

    def _clean_generated_question(self, text: str) -> str:
        """Extract just the question from generated text."""
        # Remove common prompt prefixes
        prefixes = [
            "Context:",
            "Task:",
            "Question:",
            "Q:",
        ]

        for prefix in prefixes:
            if prefix in text:
                # Take everything after the last occurrence
                parts = text.split(prefix)
                text = parts[-1]

        # Clean up
        text = text.strip()

        # If it's multiline, take first question
        if '\n' in text:
            lines = text.split('\n')
            for line in lines:
                if '?' in line:
                    return line.strip()
            # If no question mark, return first non-empty line
            for line in lines:
                if line.strip():
                    return line.strip()

        return text

    def update_context(self, transcript: Transcript):
        """Update the current transcript context."""
        self.current_transcript = transcript

    def create_reward_function(self, context: Optional[Transcript] = None):
        """
        Create a reward function closure for GRPO trainer.

        Returns a function that can be passed to GRPOTrainer.
        """
        def reward_func(completions, **kwargs):
            """Reward function for GRPO that uses online computation."""
            # Use provided context or current
            base_transcript = context if context is not None else self.current_transcript

            # Compute rewards for all completions
            rewards = self.compute_rewards_for_questions(completions, base_transcript)

            return rewards

        return reward_func


# Global instance for use in training
ONLINE_CALCULATOR = None


def get_online_reward_calculator(
    persona: str = "likes fast cars but is on a budget",
    domain: str = "cars"
) -> OnlineRewardCalculator:
    """Get or create global calculator instance."""
    global ONLINE_CALCULATOR
    if ONLINE_CALCULATOR is None:
        ONLINE_CALCULATOR = OnlineRewardCalculator(persona=persona, domain=domain)
    return ONLINE_CALCULATOR


def create_online_reward_function(
    persona: str = "likes fast cars but is on a budget",
    domain: str = "cars",
    context: Optional[Transcript] = None
):
    """
    Create an online reward function for GRPO training.

    This is the main entry point for the training script.
    """
    calculator = get_online_reward_calculator(persona, domain)
    return calculator.create_reward_function(context)


if __name__ == "__main__":
    # Test the online reward calculator
    print("Testing online reward calculator...")

    calculator = OnlineRewardCalculator()

    # Test questions
    test_questions = [
        "What's your budget for a car?",
        "Do you prefer performance or fuel efficiency?",
        "How important is reliability to you?"
    ]

    print(f"\nTesting with {len(test_questions)} questions...")
    rewards = calculator.compute_rewards_for_questions(test_questions)

    print("\nResults:")
    for q, r in zip(test_questions, rewards):
        print(f"  Q: {q}")
        print(f"  Reward: {r:.4f}")

    print("\nOnline reward calculator test complete!")