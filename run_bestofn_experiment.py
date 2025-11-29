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
from pllm_mve.responder import ResponderLLM

def run_bestofn_episode(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    responder: ResponderLLM,          # proposal model P
    evaluator: EvaluatorD,            # judge
    eval_set_E: EpisodeEvalSet,       # only E used for rewards
    logger: EpisodeLogger,
    num_candidates: int = 5,
    num_samples: int = 3,
) -> Tuple[Transcript, Dict]:
    """
    Best-of-N episode:

      - Rewards/log-scores are computed ONLY on E.
      - Hypothetical answers for scoring questions come from ResponderLLM (P).
      - Real transcript answers come from Persona LLM (PLLM).
    """

    transcript = Transcript()

    print(f"Starting best-of-n episode with {config.num_turns} turns...")
    print(f" Candidates per turn (k): {num_candidates}")
    print(f" Samples per candidate (t): {num_samples}")

    print("\n" + "=" * 70)
    print("INITIAL BASELINE ON E (no conversation yet)")
    print("=" * 70)

    # Use E for scoring
    f_prev = log_score(eval_set_E, transcript, evaluator, verbose=True)

    for turn in tqdm(range(config.num_turns), desc="Best-of-n turns"):
        # 1) Generate candidate questions
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set_E.items,
            transcript=transcript,
            pool_size=num_candidates,
            client=evaluator.chat,  # reuse same Together client
        )

        baseline_score = f_prev
        best_q = None
        best_gain_stat = float("-inf")

        # 2) For each question, sample num_samples hypothetical answers from P
        for q in questions:
            gains = []
            for _ in range(num_samples):
                # Hypothetical answer from proposal model P (ResponderLLM)
                a_sample = responder.answer_question(q)

                # Trial transcript
                trial = Transcript(turns=transcript.turns.copy())
                trial.add_turn(q, a_sample)

                # Judge on E
                new_score = log_score(eval_set_E, trial, evaluator, verbose=False)
                gains.append(new_score - baseline_score)

            # Aggregate: max gain (as you already switched)
            gain_stat = float(np.max(gains)) if gains else 0.0

            print("\n" + "-" * 60)
            print(f"Candidate: {q[:80]}...")
            print(f"Gains on E: {[f'{g:.3f}' for g in gains]} → Max = {gain_stat:.3f}")

            if gain_stat > best_gain_stat:
                best_gain_stat = gain_stat
                best_q = q

        # 3) Ask the best question to the Persona LLM for the *real* transcript
        real_answer = persona_pllm.answer_question(best_q)
        transcript.add_turn(best_q, real_answer)

        print("\n" + "=" * 70)
        print(f" AFTER TURN {turn} - Updated predictions on E:")
        print("=" * 70)

        f_new = log_score(eval_set_E, transcript, evaluator, verbose=True)
        reward = f_new - f_prev

        logger.log_turn(
            turn=turn,
            question=best_q,
            answer=real_answer,
            score=f_new,
            reward=reward,
            expected_gain=best_gain_stat,
            policy="bestofn",
        )

        print(
            f" Turn {turn}: score_E={f_new:.4f}, "
            f"reward={reward:.4f}, "
            f"max_gain={best_gain_stat:.4f}"
        )

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary




def run_direct_episode(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    evaluator: EvaluatorD,
    eval_set_E: EpisodeEvalSet,
    logger: EpisodeLogger,
    client: TogetherChat,
    num_candidates: int = 5,
) -> Tuple[Transcript, Dict]:
    """
    Direct baseline episode.
    """

    transcript = Transcript()

    print(f"Starting direct baseline episode with {config.num_turns} turns...")
    print(f" Direct baseline num_candidates per turn: {num_candidates}")

    print("\n" + "=" * 70)
    print("INITIAL BASELINE ON E (no conversation yet)")
    print("=" * 70)

    # Score on E with empty transcript
    f_prev = log_score(eval_set_E, transcript, evaluator, verbose=True)

    for turn in tqdm(range(config.num_turns), desc="Direct turns"):
        # 1) Generate same candidate pool as Best-of-N
        questions = generate_candidate_questions(
            domain=config.domain,
            items=eval_set_E.items,
            transcript=transcript,
            pool_size=num_candidates,
            client=client,   # same TogetherChat used elsewhere
        )

        if not questions:
            print("WARNING: generate_candidate_questions returned no questions; skipping turn.")
            break

        # 2) Pick one at random (no selection logic)
        question = random.choice(questions)
        print("\n" + "-" * 60)
        print(f"Direct baseline picked question (random from {num_candidates}):")
        print(question)

        # 3) Ask Persona PLLM for the real answer
        answer = persona_pllm.answer_question(question)
        transcript.add_turn(question, answer)

        print("\n" + "=" * 70)
        print(f" AFTER TURN {turn} - Updated predictions on E:")
        print("=" * 70)

        # 4) Recompute score on E
        f_new = log_score(eval_set_E, transcript, evaluator, verbose=True)
        reward = f_new - f_prev

        logger.log_turn(
            turn=turn,
            question=question,
            answer=answer,
            score=f_new,
            reward=reward,
            policy="direct",
        )

        print(f" Turn {turn}: score_E={f_new:.4f}, reward={reward:.4f}")

        f_prev = f_new

    summary = logger.get_summary()
    return transcript, summary




def compare_policies(
    config: EpisodeConfig,
    persona_pllm: PLLM,
    responder: ResponderLLM,
    evaluator: EvaluatorD,
    eval_set_E: EpisodeEvalSet,
    eval_set_T: EpisodeEvalSet,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3,
    client: TogetherChat = None,
) -> Dict:
    """
    Run Best-of-N and Direct baseline:

      - Both use SAME question generator (generate_candidate_questions).
      - Best-of-N scores the candidates using hypothetical answers and the judge on E.
      - Direct picks one candidate at random (no scoring).
      - Both use Persona PLLM for real answers.
      - Final performance is evaluated on T.
    """

    if client is None:
        client = TogetherChat(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    # ------------------- Best-of-N -------------------
    print("\n" + "=" * 60)
    print("RUNNING BEST-OF-N POLICY")
    print("=" * 60)

    bestofn_logger = EpisodeLogger(output_dir)
    bestofn_transcript, bestofn_summary = run_bestofn_episode(
        config=config,
        persona_pllm=persona_pllm,
        responder=responder,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        logger=bestofn_logger,
        num_candidates=num_candidates,
        num_samples=num_samples,
    )
    bestofn_logger.save(suffix="_bestofn")

    # ------------------- Direct baseline -------------------
    print("\n" + "=" * 60)
    print("RUNNING DIRECT BASELINE")
    print("=" * 60)

    direct_logger = EpisodeLogger(output_dir)
    direct_transcript, direct_summary = run_direct_episode(
        config=config,
        persona_pllm=persona_pllm,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        logger=direct_logger,
        client=client,
        num_candidates=num_candidates,  # match Best-of-N
    )
    direct_logger.save(suffix="_direct")

    # ------------------- Final evaluation on T -------------------
    bestofn_final_T = log_score(eval_set_T, bestofn_transcript, evaluator, verbose=False)
    direct_final_T = log_score(eval_set_T, direct_transcript, evaluator, verbose=False)

    # Maintain backward-compatible keys so aggregate code still works
    total_reward_diff = bestofn_summary["total_reward"] - direct_summary["total_reward"]
    final_score_diff = bestofn_final_T - direct_final_T

    comparison = {
        "bestofn": {
            **bestofn_summary,
            "final_logscore_T": bestofn_final_T,
        },
        "direct": {
            **direct_summary,
            "final_logscore_T": direct_final_T,
        },
        "improvement": {
            # old keys expected by aggregate code:
            "total_reward": total_reward_diff,
            "final_score": final_score_diff,
            # more explicit keys (optional, for your own inspection):
            "total_reward_E": total_reward_diff,
            "final_logscore_T": final_score_diff,
        },
        "config": {
            "num_candidates": num_candidates,
            "num_samples": num_samples,
        },
    }

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Best-of-N total reward on E: {bestofn_summary['total_reward']:.4f}")
    print(f"Direct   total reward on E: {direct_summary['total_reward']:.4f}")
    print(f"Improvement (reward E):    {total_reward_diff:.4f}")
    print(f"\nBest-of-N final logscore on T: {bestofn_final_T:.4f}")
    print(f"Direct   final logscore on T: {direct_final_T:.4f}")
    print(f"Improvement (logscore T):      {final_score_diff:.4f}")

    return comparison



def setup_episode(config: EpisodeConfig, seed: int = None) -> Tuple[EpisodeEvalSet, EpisodeEvalSet]:
    """
    Setup an episode with:
      - items
      - E: a subset of pairs for reward computation
      - T: the full set of pairs for final evaluation

    Returns:
        eval_set_E, eval_set_T
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"\nSetting up episode (seed={seed})...")

    # 1) Build items using a temporary Together client
    temp_chat = TogetherChat(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    items = generate_items_for_domain(
        domain=config.domain,
        persona=config.persona,
        num_items=config.num_items,
        chat_client=temp_chat,
    )

    print(f"Generated {len(items)} items for domain '{config.domain}'")

    # 2) Build full pair set T
    all_pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]

    print(f"Full pair set T has {len(all_pairs)} pairs")

    # 3) Sample E ⊂ T for rewards
    num_pairs_E = min(config.num_pairs, len(all_pairs))
    pairs_E = random.sample(all_pairs, num_pairs_E)

    print(f"Eval subset E has {len(pairs_E)} pairs")

    eval_set_E = EpisodeEvalSet(items=items, pairs=pairs_E, labels={})
    eval_set_T = EpisodeEvalSet(items=items, pairs=all_pairs, labels={})

    return eval_set_E, eval_set_T



def run_single_seed(
    config,
    seed: int,
    output_dir: Path,
    num_candidates: int = 5,
    num_samples: int = 3,
) -> dict:
    """Run Best-of-N vs Direct for a single seed."""

    print(f"\n{'=' * 60}")
    print(f"RUNNING SEED {seed}")
    print(f"{'=' * 60}")

    # 1) Setup episode -> E and T
    eval_set_E, eval_set_T = setup_episode(config.episode, seed)

    # 2) Save episode metadata
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    episode_data = {
        "seed": seed,
        "items": eval_set_T.items,
        "pairs_E": eval_set_E.pairs,
        "pairs_T": eval_set_T.pairs,
    }
    save_json(episode_data, seed_dir / "episode_data.json")

    # 3) Persona PLLM + labels
    chat_persona = TogetherChat(
        model=config.episode.model,
        max_tokens=config.episode.max_tokens,
        temperature=config.episode.temperature,
    )
    persona_pllm = PLLM(chat_persona)
    persona_pllm.initialize_persona(config.episode.persona, eval_set_T)

    print("Labeling E and T with Persona PLLM...")

    for pair in eval_set_E.pairs:
        label = persona_pllm.label_eval_question(pair[0], pair[1])
        eval_set_E.labels[pair] = label

    for pair in eval_set_T.pairs:
        label = persona_pllm.label_eval_question(pair[0], pair[1])
        eval_set_T.labels[pair] = label

    # 4) Judge (EvaluatorD)
    evaluator = EvaluatorD(chat_persona)

    # 5) ResponderLLM (proposal model P) 
    chat_responder = TogetherChat(
        model=config.episode.model,
        max_tokens=config.episode.max_tokens,
        temperature=config.episode.temperature,
    )
    responder = ResponderLLM(chat_responder)
    responder.initialize_persona(config.episode.persona)

    # 6) Run comparison (Best-of-N vs Direct)
    comparison = compare_policies(
        config=config.episode,
        persona_pllm=persona_pllm,
        responder=responder,
        evaluator=evaluator,
        eval_set_E=eval_set_E,
        eval_set_T=eval_set_T,
        output_dir=seed_dir,
        num_candidates=num_candidates,
        num_samples=num_samples,
        client=chat_persona,  # also used to generate questions
    )

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
