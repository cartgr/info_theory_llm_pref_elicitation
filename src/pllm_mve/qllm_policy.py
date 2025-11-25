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
    responder,  # ResponderLLM
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    transcript: Transcript
) -> Tuple[str, str, float]:
    """Select the question that maximizes information gain (using ResponderLLM to sample the *actual* answer)."""
    from .scoring import log_score
    baseline_score = log_score(eval_set, transcript, evaluator)

    best_question = None
    best_answer = None
    best_gain = float("-inf")

    for question in questions:
        # Hypothetical: use responder to sample the answer
        answer = responder.answer_question(question)
        trial = Transcript(turns=transcript.turns.copy())
        trial.add_turn(question, answer)

        new_score = log_score(eval_set, trial, evaluator)
        gain = new_score - baseline_score

        if gain > best_gain:
            best_gain = gain
            best_question = question
            best_answer = answer

    return best_question, best_answer, best_gain


def select_random_question(
    questions: List[str],
    responder  # ResponderLLM
) -> Tuple[str, str]:
    """Random baseline using the responder for the answer."""
    import random
    question = random.choice(questions)
    answer = responder.answer_question(question)
    return question, answer


def select_bestofn_question(
    questions: List[str],
    responder,  # ResponderLLM
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    transcript: Transcript,
    num_samples: int = 3
) -> Tuple[str, str, float]:
    """
    Best-of-N selection using expected information gain.
    For each candidate: sample 'num_samples' answers from the *responder* (not PLLM),
    compute mean gain, pick the best, then get one more actual answer for the chosen Q.
    """
    from .scoring import log_score
    import numpy as np

    baseline_score = log_score(eval_set, transcript, evaluator)
    best_question = None
    best_expected_gain = float("-inf")

    for question in questions:
        gains = []
        for _ in range(num_samples):
            a = responder.answer_question(question)
            trial = Transcript(turns=transcript.turns.copy())
            trial.add_turn(question, a)
            new_score = log_score(eval_set, trial, evaluator)
            gains.append(new_score - baseline_score)

        expected_gain = float(np.mean(gains)) if gains else 0.0
        if expected_gain > best_expected_gain:
            best_expected_gain = expected_gain
            best_question = question

    # Ask once "for real" (also via responder — same distribution)
    actual_answer = responder.answer_question(best_question)
    return best_question, actual_answer, best_expected_gain


def select_direct_question(
    domain: str,
    items: List[str],
    transcript: Transcript,
    responder,                      # ResponderLLM-like object with .answer_question()
    client: Optional[TogetherChat] = None,
) -> Tuple[str, str]:
    """
    Generate a single question directly with an LLM and answer it with the responder.

    This is the 'direct' baseline: no best-of-n search, just:
      1) Use client (question LLM) to generate one informative question.
      2) Use responder to answer that question given the persona.
    """
    if client is None:
        client = TogetherChat()

    # Build conversation context
    context = ""
    if transcript.turns:
        context = "Previous conversation:\n"
        for i, turn in enumerate(transcript.turns):
            context += f"Q{i+1}: {turn.question}\n"
            context += f"A{i+1}: {turn.answer}\n\n"

    # System prompt: interviewer that knows the domain & items
    system = (
        f"You are an expert interviewer trying to learn someone's preferences about {domain}. "
        f"Generate informative questions that will help uncover their preferences.\n\n"
        f"The items being considered are:\n"
        + "\n".join(f"- {item}" for item in items[:10])
        + (f"\n...and {len(items) - 10} more" if len(items) > 10 else "")
    )

    # User prompt: ask for one targeted, conversational question
    user = (
        f"{context}\n"
        f"Generate 1 question to ask next. The question should:\n"
        f"- Be specific and targeted\n"
        f"- Help reveal preferences about {domain}\n"
        f"- Be natural and conversational\n\n"
        f"Return ONLY the question text, with no numbering or extra commentary."
    )

    question = client.chat(
        system=system,
        user=user,
        temperature=0.8,
        max_tokens=100,
    ).strip()

    # Answer is produced by the responder LLM (separate from PLLM/evaluator)
    answer = responder.answer_question(question)

    return question, answer
