#!/usr/bin/env python
"""Collect diverse training data with multiple personas and evaluation sets."""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from question_generator import QuestionGenerator
from src.pllm_mve.types import Transcript

# Diverse personas for car preferences
PERSONAS = [
    "likes fast cars but is on a budget",
    "prioritizes safety and reliability over speed",
    "wants a luxury SUV for family trips",
    "prefers eco-friendly electric vehicles",
    "looking for a practical commuter car with good gas mileage",
    "enthusiast who wants a manual transmission sports car",
    "needs a truck for work and hauling",
    "wants a compact car for city driving and easy parking",
    "interested in hybrid vehicles for fuel efficiency",
    "looking for a vintage/classic car to restore",
    "needs an all-wheel drive vehicle for winter weather",
    "wants a convertible for weekend drives",
    "prioritizes technology and modern features",
    "looking for a reliable used car under $15k",
    "wants a minivan for large family",
    "interested in performance sedans with good handling",
    "needs a fuel-efficient car for long commutes",
    "wants off-road capable vehicle for outdoor adventures",
    "looking for a quiet, comfortable luxury sedan",
    "prefers American-made vehicles",
]

# Different conversation starters and scenarios
SCENARIOS = [
    "discover what they're looking for",
    "understand their budget constraints",
    "learn about their daily driving needs",
    "find out their must-have features",
    "understand their lifestyle and how they'll use the car",
    "learn about their previous car experiences",
    "discover their priorities (speed, safety, efficiency, etc.)",
    "understand their family size and passenger needs",
    "learn about their typical driving conditions",
    "find out their timeline for purchasing",
]


def generate_diverse_prompts(
    question_gen: QuestionGenerator,
    domain: str,
    num_prompts_per_persona: int,
    max_depth: int,
    questions_per_prompt: int
):
    """Generate diverse prompts across multiple personas and scenarios."""
    all_prompts = []

    for persona_idx, persona in enumerate(PERSONAS, 1):
        print(f"\n{'='*60}")
        print(f"Persona {persona_idx}/{len(PERSONAS)}: {persona}")
        print(f"{'='*60}")

        # For each persona, generate prompts at different depths
        for depth in range(max_depth + 1):
            print(f"\n  Depth {depth}: Generating {num_prompts_per_persona} prompts...")

            for prompt_idx in range(num_prompts_per_persona):
                # Build a transcript to this depth
                transcript = Transcript()

                # Randomly select a scenario for variety
                scenario = SCENARIOS[prompt_idx % len(SCENARIOS)]

                # Build conversation to desired depth
                for d in range(depth):
                    if d == 0:
                        # Initial question
                        questions = question_gen.generate_initial_questions(
                            beginning_prompt=scenario,
                            domain=domain,
                            num_questions=1
                        )
                    else:
                        # Follow-up question
                        questions = question_gen.generate_followup_questions(
                            transcript=transcript,
                            domain=domain,
                            num_questions=1
                        )

                    if questions:
                        # Simulate adding to transcript (without actual PLLM call)
                        question = questions[0]
                        # Use persona as answer context (simplified)
                        answer = f"[Answer based on: {persona}]"
                        transcript.add_turn(question, answer)

                # Generate candidate questions at this depth
                if depth == 0:
                    questions = question_gen.generate_initial_questions(
                        beginning_prompt=scenario,
                        domain=domain,
                        num_questions=questions_per_prompt
                    )
                else:
                    questions = question_gen.generate_followup_questions(
                        transcript=transcript,
                        domain=domain,
                        num_questions=questions_per_prompt
                    )

                # Format as training prompt
                prompt = question_gen.format_as_prompt(transcript, domain)

                prompt_data = {
                    "prompt": prompt,
                    "persona": persona,
                    "scenario": scenario,
                    "depth": depth,
                    "candidate_questions": questions,
                    "transcript_length": len(transcript.turns)
                }

                all_prompts.append(prompt_data)

            print(f"    Generated {num_prompts_per_persona} prompts at depth {depth}")

    return all_prompts


def main():
    parser = argparse.ArgumentParser(description="Collect diverse training data")
    parser.add_argument(
        "--domain",
        default="cars",
        help="Domain for questions"
    )
    parser.add_argument(
        "--prompts-per-persona",
        type=int,
        default=5,
        help="Number of prompts to generate per persona at each depth"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum conversation depth"
    )
    parser.add_argument(
        "--questions-per-prompt",
        type=int,
        default=8,
        help="Number of candidate questions per prompt"
    )
    parser.add_argument(
        "--output-dir",
        default="grpo_data_collection/outputs/diverse_collection",
        help="Output directory"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DIVERSE DATA COLLECTION FOR GRPO TRAINING")
    print("=" * 60)
    print(f"Domain: {args.domain}")
    print(f"Personas: {len(PERSONAS)}")
    print(f"Prompts per persona per depth: {args.prompts_per_persona}")
    print(f"Max depth: {args.max_depth}")
    print(f"Questions per prompt: {args.questions_per_prompt}")
    print(f"Total prompts: {len(PERSONAS) * args.prompts_per_persona * (args.max_depth + 1)}")
    print("=" * 60)

    # Initialize question generator
    question_gen = QuestionGenerator()

    # Generate diverse prompts
    all_prompts = generate_diverse_prompts(
        question_gen=question_gen,
        domain=args.domain,
        num_prompts_per_persona=args.prompts_per_persona,
        max_depth=args.max_depth,
        questions_per_prompt=args.questions_per_prompt
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"collected_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save in TRL format (just prompts)
    trl_dir = output_dir / "trl_format"
    trl_dir.mkdir(exist_ok=True)

    prompts_only = [p["prompt"] for p in all_prompts]
    trl_data = {"prompt": prompts_only}

    with open(trl_dir / "train.json", 'w') as f:
        json.dump(trl_data, f, indent=2)

    # Save full metadata
    with open(output_dir / "prompts_with_metadata.json", 'w') as f:
        json.dump(all_prompts, f, indent=2)

    # Save statistics
    stats = {
        "total_prompts": len(all_prompts),
        "personas": len(PERSONAS),
        "prompts_per_persona": args.prompts_per_persona,
        "max_depth": args.max_depth,
        "questions_per_prompt": args.questions_per_prompt,
        "personas_used": PERSONAS,
        "scenarios_used": SCENARIOS,
        "timestamp": timestamp
    }

    with open(output_dir / "collection_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total prompts collected: {len(all_prompts)}")
    print(f"Personas covered: {len(PERSONAS)}")
    print(f"Depth range: 0-{args.max_depth}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Training file: {trl_dir / 'train.json'}")
    print(f"Metadata: {output_dir / 'prompts_with_metadata.json'}")
    print("\n" + "=" * 60)
    print("\nTo train on this data:")
    print(f"  export PYTORCH_ENABLE_MPS_FALLBACK=1")
    print(f"  python grpo_data_collection/train_with_online_rewards.py \\")
    print(f"    --prompts {trl_dir / 'train.json'} \\")
    print(f"    --model Qwen/Qwen2.5-1.5B-Instruct \\")
    print(f"    --output grpo_trained_models/diverse_personas \\")
    print(f"    --num-generations 3 \\")
    print(f"    --epochs 2")
    print("=" * 60)


if __name__ == "__main__":
    main()
