#!/usr/bin/env python3
"""Main experiment runner for PLLM MVE."""

import argparse
import random
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pllm_mve.config import load_config, save_config
from pllm_mve.types import EpisodeConfig, EpisodeEvalSet
from pllm_mve.together_client import TogetherChat
from pllm_mve.eval_items import generate_items_for_domain, generate_item_pairs
from pllm_mve.pllm import PLLM
from pllm_mve.evaluator import EvaluatorD
from pllm_mve.rollout import compare_policies
from pllm_mve.io_utils import create_experiment_dir, ExperimentLogger, save_json


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
    output_dir: Path
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
        output_dir=seed_dir
    )

    # Save comparison results
    save_json(comparison, seed_dir / "comparison.json")

    return comparison


def main():
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="Run PLLM MVE experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/mve_one_persona.yml"),
        help="Path to experiment config file"
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
        experiment_name=config.name
    )
    config.output_dir = exp_dir

    # Save config
    save_config(config, exp_dir / "config.yml")

    print(f"\nExperiment: {config.name}")
    print(f"Output directory: {exp_dir}")
    print(f"Running {config.num_seeds} seeds...")

    # Run experiments across seeds
    experiment_logger = ExperimentLogger(exp_dir)

    for seed in range(config.num_seeds):
        try:
            comparison = run_single_seed(config, seed, exp_dir)

            # Add to experiment logger
            summary = comparison["greedy"]
            summary["improvement_over_random"] = comparison["improvement"]["total_reward"]
            experiment_logger.add_episode(seed, summary)

        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            if args.debug:
                raise

    # Save experiment summary
    experiment_logger.save_summary()

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()