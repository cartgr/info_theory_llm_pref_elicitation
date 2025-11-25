"""GRPO data collection with group rewards and tree structure."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pllm_mve.types import Transcript, Turn, EpisodeEvalSet, Pair
from src.pllm_mve.pllm import PLLM
from src.pllm_mve.evaluator import EvaluatorD
from src.pllm_mve.together_client import TogetherChat
from src.pllm_mve.eval_items import generate_item_pairs
from question_generator import QuestionGenerator


@dataclass
class GRPODataPoint:
    """Single GRPO training data point."""
    prompt: str  # Context + task instruction
    completions: List[str]  # Generated questions
    rewards: List[float]  # Rewards for each question
    advantages: List[float]  # Group-relative advantages
    metadata: Dict = field(default_factory=dict)


@dataclass
class TreeNode:
    """Node in the dialogue tree."""
    question: str
    answer: str
    transcript: Transcript
    depth: int
    reward: float
    children: List['TreeNode'] = field(default_factory=list)


class GRPOCollector:
    """Collect GRPO training data with tree-structured dialogues."""

    def __init__(
        self,
        persona: str,
        domain: str,
        items: List[str],
        pairs: List[Pair],
        num_generations: int = 8,
        max_depth: int = 3,
        beginning_prompt: str = "Discover the user's preferences",
        chat_client: Optional[TogetherChat] = None
    ):
        """Initialize GRPO collector."""
        self.persona = persona
        self.domain = domain
        self.items = items
        self.pairs = pairs
        self.num_generations = num_generations
        self.max_depth = max_depth
        self.beginning_prompt = beginning_prompt

        # Initialize components
        self.chat = chat_client or TogetherChat()
        self.pllm = PLLM(chat_client=self.chat)
        self.eval_set = EpisodeEvalSet(items=items, pairs=pairs)
        self.pllm.initialize_persona(persona, self.eval_set)
        self.evaluator = EvaluatorD(chat_client=self.chat)
        self.question_gen = QuestionGenerator(chat_client=self.chat)
        self.responder = responder or QuestionGenerator(chat_client=self.chat)

        # Collect PLLM labels once
        self.pllm_labels = self._collect_pllm_labels()

    def _collect_pllm_labels(self) -> Dict[Pair, int]:
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

        return log_score / len(self.pairs)  # Normalize by number of pairs

    def compute_rewards_for_questions(
        self,
        questions: List[str],
        base_transcript: Transcript
    ) -> Tuple[List[float], List[str]]:
        """Compute rewards for a group of questions."""
        rewards = []
        answers = []

        base_score = self.compute_log_score(base_transcript)

        for question in questions:
            # Get PLLM answer
            answer = self.responder.answer_question(question)
            answers.append(answer)

            # Create new transcript with this Q&A
            new_transcript = Transcript(turns=base_transcript.turns.copy())
            new_transcript.add_turn(question, answer)

            # Compute information gain
            new_score = self.compute_log_score(new_transcript)
            reward = new_score - base_score
            rewards.append(reward)

        return rewards, answers

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages from rewards."""
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)

        if std_reward < 1e-8:  # Avoid division by zero
            return [0.0] * len(rewards)

        advantages = (rewards_array - mean_reward) / std_reward
        return advantages.tolist()

    def collect_initial_data(self) -> GRPODataPoint:
        """Collect GRPO data for initial questions."""
        # Generate diverse initial questions
        questions = self.question_gen.generate_initial_questions(
            self.beginning_prompt,
            self.domain,
            self.num_generations
        )

        # Compute rewards
        empty_transcript = Transcript()
        rewards, answers = self.compute_rewards_for_questions(questions, empty_transcript)

        # Compute advantages
        advantages = self.compute_advantages(rewards)

        # Format prompt
        prompt = self.question_gen.format_as_prompt(empty_transcript, self.domain)

        return GRPODataPoint(
            prompt=prompt,
            completions=questions,
            rewards=rewards,
            advantages=advantages,
            metadata={
                "persona": self.persona,
                "domain": self.domain,
                "depth": 0,
                "answers": answers
            }
        )

    def collect_followup_data(self, transcript: Transcript, depth: int) -> GRPODataPoint:
        """Collect GRPO data for follow-up questions."""
        # Generate follow-up questions
        questions = self.question_gen.generate_followup_questions(
            transcript,
            self.domain,
            self.num_generations
        )

        # Compute rewards
        rewards, answers = self.compute_rewards_for_questions(questions, transcript)

        # Compute advantages
        advantages = self.compute_advantages(rewards)

        # Format prompt
        prompt = self.question_gen.format_as_prompt(transcript, self.domain)

        return GRPODataPoint(
            prompt=prompt,
            completions=questions,
            rewards=rewards,
            advantages=advantages,
            metadata={
                "persona": self.persona,
                "domain": self.domain,
                "depth": depth,
                "answers": answers
            }
        )

    def build_tree_dataset(self, branch_factor: int = 2) -> List[GRPODataPoint]:
        """Build tree-structured dataset with selective branching."""
        dataset = []

        # Collect initial data
        print("Collecting initial questions...")
        initial_data = self.collect_initial_data()
        dataset.append(initial_data)

        # Select top questions to branch from (based on advantages)
        top_indices = np.argsort(initial_data.advantages)[-branch_factor:]

        # Build tree from selected initial questions
        for idx in top_indices:
            question = initial_data.completions[idx]
            answer = initial_data.metadata["answers"][idx]

            # Create transcript for this branch
            transcript = Transcript()
            transcript.add_turn(question, answer)

            # Recursively build tree
            self._build_tree_recursive(
                transcript,
                depth=1,
                dataset=dataset,
                branch_factor=branch_factor
            )

        return dataset

    def _build_tree_recursive(
        self,
        transcript: Transcript,
        depth: int,
        dataset: List[GRPODataPoint],
        branch_factor: int
    ):
        """Recursively build tree and collect data."""
        if depth >= self.max_depth:
            return

        # Collect follow-up data
        print(f"Collecting data at depth {depth}, transcript length: {len(transcript.turns)}")
        data = self.collect_followup_data(transcript, depth)
        dataset.append(data)

        # Select top questions to continue branching
        if depth < self.max_depth - 1:
            top_indices = np.argsort(data.advantages)[-branch_factor:]

            for idx in top_indices:
                question = data.completions[idx]
                answer = data.metadata["answers"][idx]

                # Create new transcript with this Q&A
                new_transcript = Transcript(turns=transcript.turns.copy())
                new_transcript.add_turn(question, answer)

                # Recurse
                self._build_tree_recursive(
                    new_transcript,
                    depth + 1,
                    dataset,
                    branch_factor
                )

    def collect_full_dataset(self, branch_factor: int = 2) -> List[GRPODataPoint]:
        """Collect full GRPO dataset with tree structure."""
        print(f"Starting GRPO data collection for persona: {self.persona}")
        print(f"Domain: {self.domain}")
        print(f"Generations per turn: {self.num_generations}")
        print(f"Max depth: {self.max_depth}")
        print(f"Branch factor: {branch_factor}")

        dataset = self.build_tree_dataset(branch_factor)

        print(f"\nCollected {len(dataset)} GRPO data points")

        # Print statistics
        depths = [d.metadata["depth"] for d in dataset]
        for depth in range(self.max_depth):
            count = sum(1 for d in depths if d == depth)
            print(f"  Depth {depth}: {count} data points")

        return dataset