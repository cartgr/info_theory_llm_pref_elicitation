#!/usr/bin/env python
"""Main script to run GRPO data collection."""

import argparse
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pllm_mve.eval_items import get_car_items, generate_item_pairs
from src.pllm_mve.together_client import TogetherChat
from grpo_collector import GRPOCollector
from export_grpo_dataset import export_all_formats


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    default_config = script_dir / "configs" / "grpo_config.yml"

    parser = argparse.ArgumentParser(description="Run GRPO data collection")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--persona",
        help="Override persona from config"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        help="Override number of generations per turn"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Override maximum tree depth"
    )
    parser.add_argument(
        "--branch-factor",
        type=int,
        help="Override branch factor"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    grpo_config = config['grpo_collection']

    # Override with command-line arguments
    if args.persona:
        grpo_config['persona'] = args.persona
    if args.output_dir:
        grpo_config['output_dir'] = args.output_dir
    if args.num_generations:
        grpo_config['num_generations'] = args.num_generations
    if args.max_depth:
        grpo_config['max_depth'] = args.max_depth
    if args.branch_factor:
        grpo_config['branch_factor'] = args.branch_factor

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(grpo_config['output_dir']) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    with open(output_dir / "config.yml", 'w') as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("GRPO Data Collection")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Persona: {grpo_config['persona']}")
    print(f"  Domain: {grpo_config['domain']}")
    print(f"  Generations per turn: {grpo_config['num_generations']}")
    print(f"  Max depth: {grpo_config['max_depth']}")
    print(f"  Branch factor: {grpo_config['branch_factor']}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    # Initialize chat client
    print("\nInitializing Together API client...")
    chat_client = TogetherChat(
        model=grpo_config['model'],
        temperature=grpo_config['temperature_answer']
    )

    # Get items and pairs
    print("Setting up evaluation items...")
    if grpo_config['domain'] == 'cars':
        items = get_car_items()
    else:
        raise ValueError(f"Domain {grpo_config['domain']} not yet implemented")

    # Ensure we have the right number of items
    items = items[:grpo_config['num_items']]

    # Generate pairs
    pairs = generate_item_pairs(
        num_items=len(items),
        num_pairs=grpo_config['num_pairs'],
        seed=args.seed
    )

    print(f"  - {len(items)} items")
    print(f"  - {len(pairs)} comparison pairs")

    # Initialize collector
    print("\nInitializing GRPO collector...")
    collector = GRPOCollector(
        persona=grpo_config['persona'],
        domain=grpo_config['domain'],
        items=items,
        pairs=pairs,
        num_generations=grpo_config['num_generations'],
        max_depth=grpo_config['max_depth'],
        beginning_prompt=grpo_config['beginning_prompt'],
        chat_client=chat_client
    )

    # Collect data
    print("\nStarting data collection...")
    print("-" * 40)
    dataset = collector.collect_full_dataset(
        branch_factor=grpo_config['branch_factor']
    )

    # Export dataset
    print("\n" + "=" * 60)
    print("Exporting dataset...")
    export_all_formats(dataset, str(output_dir))

    # Print summary
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print(f"Total data points collected: {len(dataset)}")
    print(f"Output saved to: {output_dir}")

    # Print sample data point
    if dataset:
        print("\nSample data point (first entry):")
        sample = dataset[0]
        print(f"  Prompt: {sample.prompt[:100]}...")
        print(f"  Number of completions: {len(sample.completions)}")
        if sample.completions:
            print(f"  First completion: {sample.completions[0]}")
            print(f"  Reward: {sample.rewards[0]:.4f}")
            print(f"  Advantage: {sample.advantages[0]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()