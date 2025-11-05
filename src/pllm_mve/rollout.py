"""Episode rollout logic."""

from typing import Dict, List, Tuple
from tqdm import tqdm

from .types import EpisodeConfig, EpisodeEvalSet, Transcript, Turn
from .pllm import PLLM
from .evaluator import EvaluatorD
from .scoring import log_score, compute_reward
from .qllm_policy import (
    generate_candidate_questions,
    select_greedy_question,
    select_random_question
)
from .io_utils import EpisodeLogger


def run_greedy_episode(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger
) -> Tuple[Transcript, Dict]:
    """
    Run a greedy episode selecting questions to maximize information gain.

    Returns:
        (final_transcript, summary_stats)
    """
    transcript = Transcript()
    f_prev = log_score(eval_set, transcript, evaluator)

    print(f"Starting greedy episode with {config.num_turns} turns...")
    print(f"Initial score: {f_prev:.4f}")

    for turn in tqdm(range(config.num_turns), desc="Greedy turns"):
        # Generate candidate questions
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pool_size=config.num_items,  # Use num_items as pool size
            client=pllm.chat  # Use same client as PLLM
        )

        # Select best question greedily
        best_q, best_a, best_gain = select_greedy_question(
            questions=questions,
            pllm=pllm,
            evaluator=evaluator,
            eval_set=eval_set,
            transcript=transcript
        )

        # Add to transcript
        transcript.add_turn(best_q, best_a)

        # Compute new score
        f_new = log_score(eval_set, transcript, evaluator)
        reward = f_new - f_prev

        # Log turn
        logger.log_turn(
            turn=turn,
            question=best_q,
            answer=best_a,
            score=f_new,
            reward=reward,
            expected_gain=best_gain,
            policy="greedy"
        )

        print(f"  Turn {turn}: score={f_new:.4f}, reward={reward:.4f}, expected_gain={best_gain:.4f}")

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def run_random_episode(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger
) -> Tuple[Transcript, Dict]:
    """
    Run a random baseline episode.

    Returns:
        (final_transcript, summary_stats)
    """
    transcript = Transcript()
    f_prev = log_score(eval_set, transcript, evaluator)

    print(f"Starting random episode with {config.num_turns} turns...")
    print(f"Initial score: {f_prev:.4f}")

    for turn in tqdm(range(config.num_turns), desc="Random turns"):
        # Generate candidate questions
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pool_size=config.num_items,
            client=pllm.chat  # Use same client as PLLM
        )

        # Select random question
        question, answer = select_random_question(questions, pllm)

        # Add to transcript
        transcript.add_turn(question, answer)

        # Compute new score
        f_new = log_score(eval_set, transcript, evaluator)
        reward = f_new - f_prev

        # Log turn
        logger.log_turn(
            turn=turn,
            question=question,
            answer=answer,
            score=f_new,
            reward=reward,
            policy="random"
        )

        print(f"  Turn {turn}: score={f_new:.4f}, reward={reward:.4f}")

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def compare_policies(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    output_dir
) -> Dict:
    """Run both greedy and random episodes and compare."""
    # Run greedy episode
    greedy_logger = EpisodeLogger(output_dir)
    greedy_transcript, greedy_summary = run_greedy_episode(
        config, pllm, evaluator, eval_set, greedy_logger
    )
    greedy_logger.save(suffix="_greedy")

    # Run random baseline
    random_logger = EpisodeLogger(output_dir)
    random_transcript, random_summary = run_random_episode(
        config, pllm, evaluator, eval_set, random_logger
    )
    random_logger.save(suffix="_random")

    # Compare results
    comparison = {
        "greedy": greedy_summary,
        "random": random_summary,
        "improvement": {
            "total_reward": greedy_summary["total_reward"] - random_summary["total_reward"],
            "final_score": greedy_summary["final_score"] - random_summary["final_score"]
        }
    }

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Greedy total reward: {greedy_summary['total_reward']:.4f}")
    print(f"Random total reward: {random_summary['total_reward']:.4f}")
    print(f"Improvement: {comparison['improvement']['total_reward']:.4f}")
    print(f"Success: {comparison['improvement']['total_reward'] > 0}")

    return comparison