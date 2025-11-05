#!/usr/bin/env python3
"""Best-of-N experiment: Compare best-of-n selection vs direct baseline."""

import argparse
import random
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pllm_mve.config import load_config, save_config
from pllm_mve.types import EpisodeConfig, EpisodeEvalSet, Transcript
from pllm_mve.together_client import TogetherChat
from pllm_mve.eval_items import generate_items_for_domain, generate_item_pairs
from pllm_mve.pllm import PLLM
from pllm_mve.evaluator import EvaluatorD
from pllm_mve.scoring import log_score
from pllm_mve.qllm_policy import (
    generate_candidate_questions,
    select_bestofn_question,
    select_direct_question
)
from pllm_mve.io_utils import create_experiment_dir, EpisodeLogger, save_json


def run_bestofn_episode(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger,
    num_candidates: int = 5,
    num_samples: int = 3
) -> Tuple[Transcript, Dict]:
    """
    Run a best-of-n episode.

    For each turn:
    - Generate num_candidates questions
    - For each, sample num_samples hypothetical answers and compute expected gain
    - Select question with highest expected gain
    - Actually ask it and get real answer

    Args:
        config: Episode configuration
        pllm: PLLM to answer questions
        evaluator: Evaluator to score transcripts
        eval_set: Fixed evaluation set
        logger: Episode logger
        num_candidates: Number of candidate questions (k)
        num_samples: Number of samples per candidate (t)

    Returns:
        (final_transcript, summary_stats)
    """
    transcript = Transcript()

    print(f"Starting best-of-n episode with {config.num_turns} turns...")
    print(f"  Candidates per turn (k): {num_candidates}")
    print(f"  Samples per candidate (t): {num_samples}")

    # Show initial baseline predictions
    print(f"\n{'='*70}")
    print("INITIAL BASELINE (no conversation yet)")
    print(f"{'='*70}")
    f_prev = log_score(eval_set, transcript, evaluator, verbose=True)

    for turn in tqdm(range(config.num_turns), desc="Best-of-n turns"):
        # Generate candidate questions using LLM
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pool_size=num_candidates,
            client=pllm.chat  # Use same client as PLLM
        )

        # Select best question using best-of-n with sampling
        best_q, actual_a, expected_gain = select_bestofn_question(
            questions=questions,
            pllm=pllm,
            evaluator=evaluator,
            eval_set=eval_set,
            transcript=transcript,
            num_samples=num_samples
        )

        # Add to transcript
        transcript.add_turn(best_q, actual_a)

        # Compute new score and show predictions
        print(f"\n  {'='*70}")
        print(f"  AFTER TURN {turn} - Updated predictions:")
        print(f"  {'='*70}")
        f_new = log_score(eval_set, transcript, evaluator, verbose=True)
        reward = f_new - f_prev

        # Log turn
        logger.log_turn(
            turn=turn,
            question=best_q,
            answer=actual_a,
            score=f_new,
            reward=reward,
            expected_gain=expected_gain,
            policy="bestofn"
        )

        print(f"  Turn {turn} Summary: score={f_new:.4f}, reward={reward:.4f}, expected_gain={expected_gain:.4f}")

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def run_direct_episode(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    logger: EpisodeLogger,
    client: TogetherChat
) -> Tuple[Transcript, Dict]:
    """
    Run a direct baseline episode.

    For each turn:
    - Use LLM to directly generate a maximally informative question
    - No evaluation or best-of-n selection
    - Ask PLLM and get answer

    Args:
        config: Episode configuration
        pllm: PLLM to answer questions
        evaluator: Evaluator to score transcripts
        eval_set: Fixed evaluation set
        logger: Episode logger
        client: Together API client for question generation

    Returns:
        (final_transcript, summary_stats)
    """
    transcript = Transcript()

    print(f"Starting direct baseline episode with {config.num_turns} turns...")

    # Show initial baseline predictions
    print(f"\n{'='*70}")
    print("INITIAL BASELINE (no conversation yet)")
    print(f"{'='*70}")
    f_prev = log_score(eval_set, transcript, evaluator, verbose=True)

    for turn in tqdm(range(config.num_turns), desc="Direct turns"):
        # Generate question directly
        question, answer = select_direct_question(
            domain=config.domain,
            items=eval_set.items,
            transcript=transcript,
            pllm=pllm,
            client=client
        )

        # Add to transcript
        transcript.add_turn(question, answer)

        # Compute new score and show predictions
        print(f"\n  {'='*70}")
        print(f"  AFTER TURN {turn} - Updated predictions:")
        print(f"  {'='*70}")
        f_new = log_score(eval_set, transcript, evaluator, verbose=True)
        reward = f_new - f_prev

        # Log turn
        logger.log_turn(
            turn=turn,
            question=question,
            answer=answer,
            score=f_new,
            reward=reward,
            policy="direct"
        )

        print(f"  Turn {turn} Summary: score={f_new:.4f}, reward={reward:.4f}")

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary


def compare_policies(
    config: EpisodeConfig,
    pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set: EpisodeEvalSet,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3,
    client: TogetherChat = None
) -> Dict:
    """Run both best-of-n and direct episodes and compare."""
    if client is None:
        client = TogetherChat()

    # Run best-of-n episode
    print("\n" + "=" * 60)
    print("RUNNING BEST-OF-N POLICY")
    print("=" * 60)
    bestofn_logger = EpisodeLogger(output_dir)
    bestofn_transcript, bestofn_summary = run_bestofn_episode(
        config, pllm, evaluator, eval_set, bestofn_logger,
        num_candidates=num_candidates,
        num_samples=num_samples
    )
    bestofn_logger.save(suffix="_bestofn")

    # Run direct baseline
    print("\n" + "=" * 60)
    print("RUNNING DIRECT BASELINE")
    print("=" * 60)
    direct_logger = EpisodeLogger(output_dir)
    direct_transcript, direct_summary = run_direct_episode(
        config, pllm, evaluator, eval_set, direct_logger, client
    )
    direct_logger.save(suffix="_direct")

    # Compare results
    comparison = {
        "bestofn": bestofn_summary,
        "direct": direct_summary,
        "improvement": {
            "total_reward": bestofn_summary["total_reward"] - direct_summary["total_reward"],
            "final_score": bestofn_summary["final_score"] - direct_summary["final_score"]
        },
        "config": {
            "num_candidates": num_candidates,
            "num_samples": num_samples
        }
    }

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Best-of-N total reward: {bestofn_summary['total_reward']:.4f}")
    print(f"Direct total reward: {direct_summary['total_reward']:.4f}")
    print(f"Improvement: {comparison['improvement']['total_reward']:.4f}")
    print(f"Success: {comparison['improvement']['total_reward'] > 0}")
    print(f"\nBest-of-N final score: {bestofn_summary['final_score']:.4f}")
    print(f"Direct final score: {direct_summary['final_score']:.4f}")

    return comparison


def setup_episode(config: EpisodeConfig, seed: int = None) -> EpisodeEvalSet:
    """Setup an episode with items and pairs."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\nSetting up episode (seed={seed})...")

    # Initialize chat client
    chat = TogetherChat(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature
    )

    # Generate items for the domain
    items = generate_items_for_domain(
        domain=config.domain,
        persona=config.persona,
        num_items=config.num_items,
        chat_client=chat
    )
    print(f"Generated {len(items)} items for domain '{config.domain}'")

    # Generate pairs for comparison
    pairs = generate_item_pairs(
        num_items=config.num_items,
        num_pairs=config.num_pairs,
        seed=seed
    )
    print(f"Generated {len(pairs)} pairs for comparison")

    # Create evaluation set
    eval_set = EpisodeEvalSet(
        items=items,
        pairs=pairs,
        labels={}
    )

    # Get labels from PLLM
    print("Getting preference labels from PLLM...")
    pllm = PLLM(chat)
    pllm.initialize_persona(config.persona, eval_set)

    for i, pair in enumerate(tqdm(pairs, desc="Labeling pairs")):
        label = pllm.label_eval_question(pair[0], pair[1])
        eval_set.labels[pair] = label

    print(f"Labeled all {len(pairs)} pairs")

    return eval_set


def run_single_seed(
    config,
    seed: int,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3
) -> dict:
    """Run experiment for a single seed."""
    print(f"\n{'='*60}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*60}")

    # Setup episode
    eval_set = setup_episode(config.episode, seed)

    # Save episode data
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    episode_data = {
        "seed": seed,
        "items": eval_set.items,
        "pairs": eval_set.pairs,
        "labels": {str(k): v for k, v in eval_set.labels.items()}
    }
    save_json(episode_data, seed_dir / "episode_data.json")

    # Initialize PLLM and Evaluator for the episode
    chat = TogetherChat(
        model=config.episode.model,
        max_tokens=config.episode.max_tokens,
        temperature=config.episode.temperature
    )
    pllm = PLLM(chat)
    pllm.initialize_persona(config.episode.persona, eval_set)

    evaluator = EvaluatorD(chat)

    # Run comparison
    comparison = compare_policies(
        config=config.episode,
        pllm=pllm,
        evaluator=evaluator,
        eval_set=eval_set,
        output_dir=seed_dir,
        num_candidates=num_candidates,
        num_samples=num_samples,
        client=chat
    )

    # Save comparison results
    save_json(comparison, seed_dir / "comparison.json")

    return comparison


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="Run Best-of-N experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/mve_one_persona.yml"),
        help="Path to experiment config file"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidate questions (k) for best-of-n"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples per candidate (t) for expected gain estimation"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Override number of seeds from config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with single seed"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.num_seeds:
        config.num_seeds = args.num_seeds

    if args.debug:
        config.num_seeds = 1
        print("Debug mode: Running single seed only")

    # Create experiment directory
    exp_dir = create_experiment_dir(
        base_dir=Path(config.output_dir).parent,
        experiment_name=f"bestofn_{config.name}"
    )
    config.output_dir = exp_dir

    # Save config
    save_config(config, exp_dir / "config.yml")

    print(f"\nExperiment: Best-of-N vs Direct Baseline - {config.name}")
    print(f"Output directory: {exp_dir}")
    print(f"Candidates (k): {args.num_candidates}")
    print(f"Samples (t): {args.num_samples}")
    print(f"Running {config.num_seeds} seeds...")

    # Track all results
    all_results = []

    for seed in range(config.num_seeds):
        try:
            comparison = run_single_seed(
                config, seed, exp_dir,
                num_candidates=args.num_candidates,
                num_samples=args.num_samples
            )
            all_results.append(comparison)

        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            if args.debug:
                raise

    # Compute aggregate statistics
    if all_results:
        bestofn_rewards = [r["bestofn"]["total_reward"] for r in all_results]
        direct_rewards = [r["direct"]["total_reward"] for r in all_results]
        improvements = [r["improvement"]["total_reward"] for r in all_results]

        aggregate = {
            "num_seeds": len(all_results),
            "bestofn": {
                "mean_reward": np.mean(bestofn_rewards),
                "std_reward": np.std(bestofn_rewards),
                "all_rewards": bestofn_rewards
            },
            "direct": {
                "mean_reward": np.mean(direct_rewards),
                "std_reward": np.std(direct_rewards),
                "all_rewards": direct_rewards
            },
            "improvement": {
                "mean": np.mean(improvements),
                "std": np.std(improvements),
                "all": improvements,
                "success_rate": sum(1 for x in improvements if x > 0) / len(improvements)
            },
            "config": {
                "num_candidates": args.num_candidates,
                "num_samples": args.num_samples
            }
        }

        save_json(aggregate, exp_dir / "aggregate_results.json")

        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"Seeds completed: {len(all_results)}")
        print(f"\nBest-of-N: {aggregate['bestofn']['mean_reward']:.4f} ± {aggregate['bestofn']['std_reward']:.4f}")
        print(f"Direct: {aggregate['direct']['mean_reward']:.4f} ± {aggregate['direct']['std_reward']:.4f}")
        print(f"Improvement: {aggregate['improvement']['mean']:.4f} ± {aggregate['improvement']['std']:.4f}")
        print(f"Success rate: {aggregate['improvement']['success_rate']:.2%}")

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
