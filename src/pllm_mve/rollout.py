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
    responder,  # ResponderLLM
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger
) -> Tuple[Transcript, Dict]:
    """Greedy selection using the responder to sample answers."""
    from .scoring import log_score
    transcript = Transcript()
    f_prev = log_score(eval_set, transcript, evaluator)

    print(f"Starting greedy episode with {config.num_turns} turns...")
    print(f"Initial score: {f_prev:.4f}")

    from .qllm_policy import generate_candidate_questions, select_greedy_question

    for turn in range(config.num_turns):
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pool_size=config.num_items,
            client=evaluator.chat  # reuse same Together client for generation
        )

        best_q, best_a, best_gain = select_greedy_question(
            questions=questions,
            responder=responder,
            evaluator=evaluator,
            eval_set=eval_set,
            transcript=transcript
        )

        transcript.add_turn(best_q, best_a)
        f_new = log_score(eval_set, transcript, evaluator)
        reward = f_new - f_prev
        logger.log_turn(
            turn=turn, question=best_q, answer=best_a,
            score=f_new, reward=reward, expected_gain=best_gain, policy="greedy"
        )
        print(f" Turn {turn}: score={f_new:.4f}, reward={reward:.4f}, expected_gain={best_gain:.4f}")
        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def run_random_episode(
    config: EpisodeConfig,
    responder,  # ResponderLLM
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger
) -> Tuple[Transcript, Dict]:
    """Random baseline using the responder to sample answers."""
    from .scoring import log_score
    transcript = Transcript()
    f_prev = log_score(eval_set, transcript, evaluator)

    print(f"Starting random episode with {config.num_turns} turns...")
    print(f"Initial score: {f_prev:.4f}")

    from .qllm_policy import generate_candidate_questions, select_random_question

    for turn in range(config.num_turns):
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pool_size=config.num_items,
            client=evaluator.chat
        )

        question, answer = select_random_question(questions, responder)
        transcript.add_turn(question, answer)

        f_new = log_score(eval_set, transcript, evaluator)
        reward = f_new - f_prev
        logger.log_turn(
            turn=turn, question=question, answer=answer,
            score=f_new, reward=reward, policy="random"
        )
        print(f" Turn {turn}: score={f_new:.4f}, reward={reward:.4f}")
        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def compare_policies(
    config: EpisodeConfig,
    responder,             # ResponderLLM
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    output_dir
) -> Dict:
    """Run greedy vs random using the responder for answer sampling."""
    greedy_logger = EpisodeLogger(output_dir)
    greedy_transcript, greedy_summary = run_greedy_episode(
        config, responder, evaluator, eval_set, greedy_logger
    )
    greedy_logger.save(suffix="_greedy")

    random_logger = EpisodeLogger(output_dir)
    random_transcript, random_summary = run_random_episode(
        config, responder, evaluator, eval_set, random_logger
    )
    random_logger.save(suffix="_random")

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
