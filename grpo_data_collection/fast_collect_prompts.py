#!/usr/bin/env python
"""Fast prompt collection - NO reward computation, just generate prompts."""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pllm_mve.together_client import TogetherChat
from src.pllm_mve.types import Transcript
from question_generator import QuestionGenerator


def generate_prompts_at_depth(
    question_gen: QuestionGenerator,
    domain: str,
    depth: int,
    num_prompts: int,
    num_questions_per_prompt: int
):
    """Generate prompts at a specific dialogue depth."""
    prompts_data = []

    print(f"\nGenerating prompts at depth {depth}...")

    for i in range(num_prompts):
        # Create a transcript with 'depth' turns
        transcript = Transcript()

        if depth > 0:
            # Generate a sample conversation to this depth
            print(f"  Creating sample dialogue {i+1}/{num_prompts} (depth {depth})...")

            # First question
            initial_questions = question_gen.generate_initial_questions(
                f"Discover preferences about {domain}",
                domain,
                num_questions=1
            )
            first_q = initial_questions[0] if initial_questions else f"What {domain} do you prefer?"
            first_a = f"Sample answer about {domain} preferences"
            transcript.add_turn(first_q, first_a)

            # Add follow-up turns to reach desired depth
            for turn_num in range(1, depth):
                followup_qs = question_gen.generate_followup_questions(
                    transcript, domain, num_questions=1
                )
                q = followup_qs[0] if followup_qs else f"Follow-up question {turn_num}"
                a = f"Sample answer {turn_num}"
                transcript.add_turn(q, a)

        # Now generate the candidate questions for this prompt
        if depth == 0:
            questions = question_gen.generate_initial_questions(
                f"Discover preferences about {domain}",
                domain,
                num_questions=num_questions_per_prompt
            )
        else:
            questions = question_gen.generate_followup_questions(
                transcript,
                domain,
                num_questions=num_questions_per_prompt
            )

        # Format as training prompt
        prompt = question_gen.format_as_prompt(transcript, domain)

        prompts_data.append({
            "prompt": prompt,
            "completions": questions,
            "depth": depth,
            "transcript_length": len(transcript.turns)
        })

    return prompts_data


def main():
    parser = argparse.ArgumentParser(description="Fast prompt collection (no rewards)")
    parser.add_argument("--domain", default="cars", help="Domain")
    parser.add_argument("--max-depth", type=int, default=3, help="Max dialogue depth")
    parser.add_argument("--prompts-per-depth", type=int, default=10, help="Prompts per depth")
    parser.add_argument("--questions-per-prompt", type=int, default=8, help="Questions per prompt")
    parser.add_argument("--output-dir", default="grpo_data_collection/outputs", help="Output directory")

    args = parser.parse_args()

    print("=" * 60)
    print("FAST PROMPT COLLECTION (No Reward Computation)")
    print("=" * 60)
    print(f"Domain: {args.domain}")
    print(f"Max depth: {args.max_depth}")
    print(f"Prompts per depth: {args.prompts_per_depth}")
    print(f"Questions per prompt: {args.questions_per_prompt}")
    print("=" * 60)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"fast_collect_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize question generator
    print("\nInitializing question generator...")
    chat_client = TogetherChat(temperature=0.8)
    question_gen = QuestionGenerator(chat_client=chat_client)

    # Collect prompts at each depth
    all_prompts = []

    for depth in range(args.max_depth + 1):
        depth_prompts = generate_prompts_at_depth(
            question_gen=question_gen,
            domain=args.domain,
            depth=depth,
            num_prompts=args.prompts_per_depth,
            num_questions_per_prompt=args.questions_per_prompt
        )
        all_prompts.extend(depth_prompts)

    print(f"\n{'=' * 60}")
    print(f"Collection complete! Generated {len(all_prompts)} prompts")
    print(f"{'=' * 60}")

    # Print statistics
    depths = [p["depth"] for p in all_prompts]
    for d in range(args.max_depth + 1):
        count = sum(1 for depth in depths if depth == d)
        print(f"  Depth {d}: {count} prompts")

    # Export in TRL format
    print(f"\nExporting to {output_dir}...")

    # TRL format (just prompts - no pre-computed rewards)
    trl_dir = output_dir / "trl_format"
    trl_dir.mkdir(exist_ok=True)

    trl_data = {
        "prompt": [p["prompt"] for p in all_prompts]
    }

    with open(trl_dir / "train.json", 'w') as f:
        json.dump(trl_data, f, indent=2)

    # Also save full data with completions for reference
    with open(output_dir / "full_data.json", 'w') as f:
        json.dump(all_prompts, f, indent=2)

    # Save config
    config = {
        "domain": args.domain,
        "max_depth": args.max_depth,
        "prompts_per_depth": args.prompts_per_depth,
        "questions_per_prompt": args.questions_per_prompt,
        "total_prompts": len(all_prompts),
        "timestamp": timestamp
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nFiles created:")
    print(f"  - {trl_dir / 'train.json'} (for training)")
    print(f"  - {output_dir / 'full_data.json'} (reference)")
    print(f"  - {output_dir / 'config.json'} (config)")

    print(f"\n{'=' * 60}")
    print("Ready for training!")
    print(f"{'=' * 60}")
    print(f"\nTo train:")
    print(f"  python grpo_data_collection/train_with_online_rewards.py \\")
    print(f"    --prompts {trl_dir / 'train.json'} \\")
    print(f"    --model Qwen/Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    main()
