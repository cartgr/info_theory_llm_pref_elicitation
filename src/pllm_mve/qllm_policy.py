"""Question generation and selection policy."""

import random
from typing import List, Tuple, Optional
from .types import EpisodeEvalSet, Transcript, Turn
from .pllm import PLLM
from .evaluator import EvaluatorD
from .scoring import log_score


def generate_candidate_questions(
    domain: str,
    items: List[str],
    transcript: Transcript,
    pool_size: int = 10
) -> List[str]:
    """Generate a pool of candidate questions."""
    questions = []

    # Generate comparison questions
    for _ in range(pool_size // 2):
        i, j = random.sample(range(len(items)), 2)
        questions.append(f"Which would you prefer: {items[i]} or {items[j]}? Why?")

    # Generate feature-based questions
    feature_questions = [
        f"What's most important to you when choosing {domain}?",
        f"What features do you absolutely need in {domain}?",
        f"What's your typical budget for {domain}?",
        f"How important is reliability versus performance for you?",
        f"Do you prefer newer or more established options?",
        f"What past experiences have shaped your preferences?",
        f"What would be a deal-breaker for you?",
        f"How do you typically use or interact with {domain}?",
    ]

    # Add random feature questions
    remaining = pool_size - len(questions)
    questions.extend(random.sample(feature_questions, min(remaining, len(feature_questions))))

    # If still need more, generate specific item questions
    while len(questions) < pool_size:
        item = random.choice(items)
        questions.append(f"What do you think about {item}?")

    return questions[:pool_size]


def select_greedy_question(
    questions: List[str],
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    transcript: Transcript
) -> Tuple[str, str, float]:
    """
    Select the question that maximizes expected information gain.

    Returns:
        (best_question, best_answer, best_gain)
    """
    # Compute baseline score
    baseline_score = log_score(eval_set, transcript, evaluator)

    best_question = None
    best_answer = None
    best_gain = float("-inf")

    for question in questions:
        # Get hypothetical answer from PLLM
        answer = pllm.answer_question(question)

        # Create trial transcript
        trial_transcript = Transcript(turns=transcript.turns.copy())
        trial_transcript.add_turn(question, answer)

        # Compute new score
        new_score = log_score(eval_set, trial_transcript, evaluator)
        gain = new_score - baseline_score

        # Track best
        if gain > best_gain:
            best_question = question
            best_answer = answer
            best_gain = gain

    return best_question, best_answer, best_gain


def select_random_question(
    questions: List[str],
    pllm: PLLM
) -> Tuple[str, str]:
    """
    Select a random question (baseline).

    Returns:
        (question, answer)
    """
    question = random.choice(questions)
    answer = pllm.answer_question(question)
    return question, answer