"""Question generation and selection policy."""

import random
import numpy as np
from typing import List, Tuple, Optional
from .types import EpisodeEvalSet, Transcript, Turn
from .pllm import PLLM
from .evaluator import EvaluatorD
from .scoring import log_score
from .together_client import TogetherChat


def generate_candidate_questions(
    domain: str,
    items: List[str],
    transcript: Transcript,
    pool_size: int = 10,
    client: Optional[TogetherChat] = None
) -> List[str]:
    """Generate a pool of candidate questions using an LLM.

    Args:
        domain: Domain for the questions (e.g., "cars")
        items: List of items in the domain
        transcript: Current conversation transcript
        pool_size: Number of questions to generate
        client: Together API client (creates new one if None)

    Returns:
        List of candidate questions
    """
    if client is None:
        client = TogetherChat()

    # Build context from transcript
    context = ""
    if transcript.turns:
        context = "Previous conversation:\n"
        for i, turn in enumerate(transcript.turns):
            context += f"Q{i+1}: {turn.question}\n"
            context += f"A{i+1}: {turn.answer}\n\n"

    # Build system prompt
    system = (
        f"You are an expert interviewer trying to learn someone's preferences about {domain}. "
        f"Generate informative questions that will help uncover their preferences.\n\n"
        f"The items being considered are:\n" + "\n".join(f"- {item}" for item in items[:10]) +
        (f"\n...and {len(items) - 10} more" if len(items) > 10 else "")
    )

    # Build user prompt
    user = (
        f"{context}\n"
        f"Generate {pool_size} diverse questions to ask next. Each question should:\n"
        f"- Be specific and targeted\n"
        f"- Help reveal preferences about {domain}\n"
        f"- Be natural and conversational\n"
        f"- Be different from the others (diverse approaches)\n\n"
        f"Return ONLY the questions, one per line, numbered 1-{pool_size}."
    )

    # Generate questions
    print(f"  Generating {pool_size} candidate questions using LLM...")
    response = client.chat(
        system=system,
        user=user,
        temperature=0.8,  # Higher temperature for diversity
        max_tokens=500
    )

    # Parse questions from response
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Remove numbering (e.g., "1. ", "1) ", etc.)
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Remove quotes if present
        line = line.strip('"\'')
        # Add if it looks like a question
        if line and (line.endswith('?') or len(line) > 10):
            questions.append(line)

    # If we didn't get enough questions, generate simple fallbacks
    while len(questions) < pool_size:
        questions.append(f"What's most important to you when choosing {domain}?")

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


def select_bestofn_question(
    questions: List[str],
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    transcript: Transcript,
    num_samples: int = 3
) -> Tuple[str, str, float]:
    """
    Select the question that maximizes EXPECTED information gain.

    For each candidate question:
    - Sample num_samples hypothetical answers
    - Compute information gain for each answer
    - Average the gains to get expected gain

    Then select the question with highest expected gain and ask it for real.

    Args:
        questions: Pool of candidate questions
        pllm: PLLM to answer questions
        evaluator: Evaluator to score transcripts
        eval_set: Fixed evaluation set of pairs
        transcript: Current conversation transcript
        num_samples: Number of hypothetical answers to sample per question (t)

    Returns:
        (best_question, actual_answer, expected_gain)
    """
    from tqdm import tqdm

    # Compute baseline score
    baseline_score = log_score(eval_set, transcript, evaluator)

    best_question = None
    best_expected_gain = float("-inf")
    question_gains = {}  # Store for debugging

    print(f"\n  Evaluating {len(questions)} candidate questions with {num_samples} samples each...")

    for q_idx, question in enumerate(tqdm(questions, desc="  Candidates", leave=False)):
        gains = []

        # Sample multiple hypothetical answers
        for sample_idx in range(num_samples):
            # Get hypothetical answer from PLLM
            answer = pllm.answer_question(question)

            # Create trial transcript
            trial_transcript = Transcript(turns=transcript.turns.copy())
            trial_transcript.add_turn(question, answer)

            # Compute new score and gain
            new_score = log_score(eval_set, trial_transcript, evaluator)
            gain = new_score - baseline_score
            gains.append(gain)

        # Calculate expected gain (mean)
        expected_gain = np.mean(gains)
        question_gains[question] = expected_gain

        # Print details for this question
        print(f"\n    Q{q_idx+1}: {question[:80]}...")
        print(f"         Gains: {[f'{g:.3f}' for g in gains]} → Mean: {expected_gain:.3f}")

        # Track best
        if expected_gain > best_expected_gain:
            best_question = question
            best_expected_gain = expected_gain

    # Print selection result
    print(f"\n  ✓ Selected question with expected gain: {best_expected_gain:.3f}")
    print(f"    Question: {best_question}")

    # Now actually ask the best question
    print(f"\n  Asking PLLM the selected question...")
    actual_answer = pllm.answer_question(best_question)
    print(f"  Answer: {actual_answer}")

    return best_question, actual_answer, best_expected_gain


def select_direct_question(
    domain: str,
    items: List[str],
    transcript: Transcript,
    pllm: PLLM,
    client: Optional[TogetherChat] = None
) -> Tuple[str, str]:
    """
    Generate a single maximally informative question directly (baseline).

    Uses an LLM to generate a question with instruction to be maximally informative,
    without any evaluation or best-of-n selection.

    Args:
        domain: Domain for the question (e.g., "cars")
        items: List of items in the domain
        transcript: Current conversation transcript
        pllm: PLLM to answer the question
        client: Together API client (creates new one if None)

    Returns:
        (question, answer)
    """
    if client is None:
        client = TogetherChat()

    # Build context from transcript
    context = ""
    if transcript.turns:
        context = "Previous conversation:\n"
        for i, turn in enumerate(transcript.turns):
            context += f"Q{i+1}: {turn.question}\n"
            context += f"A{i+1}: {turn.answer}\n\n"

    # Build system prompt (same as best-of-n for fair comparison)
    system = (
        f"You are an expert interviewer trying to learn someone's preferences about {domain}. "
        f"Generate informative questions that will help uncover their preferences.\n\n"
        f"The items being considered are:\n" + "\n".join(f"- {item}" for item in items[:10]) +
        (f"\n...and {len(items) - 10} more" if len(items) > 10 else "")
    )

    # Build user prompt (same structure as best-of-n)
    user = (
        f"{context}\n"
        f"Generate 1 question to ask next. The question should:\n"
        f"- Be specific and targeted\n"
        f"- Help reveal preferences about {domain}\n"
        f"- Be natural and conversational\n\n"
        f"Return ONLY the question, nothing else."
    )

    # Generate question
    print(f"\n  Generating 1 question using LLM...")
    question = client.chat(
        system=system,
        user=user,
        temperature=0.8,  # Same temperature as best-of-n for fair comparison
        max_tokens=100
    )
    print(f"  Question: {question}")

    # Get answer from PLLM
    print(f"\n  Asking PLLM...")
    answer = pllm.answer_question(question)
    print(f"  Answer: {answer}")

    return question, answer