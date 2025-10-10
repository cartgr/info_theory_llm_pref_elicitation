#!/usr/bin/env python
"""Extract prompts from existing datasets and combine them."""

import json
import argparse
from pathlib import Path
from datetime import datetime


def extract_prompts_from_dataset(dataset_path):
    """Extract prompts from a dataset.json file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    if isinstance(data, list):
        # Format: [{"prompt": ..., "completions": ..., ...}, ...]
        for item in data:
            if "prompt" in item:
                prompts.append(item["prompt"])
    elif isinstance(data, dict) and "prompt" in data:
        # Format: {"prompt": [...], "completions": [...]}
        if isinstance(data["prompt"], list):
            prompts.extend(data["prompt"])
        else:
            prompts.append(data["prompt"])

    return prompts


def find_all_datasets(base_dir):
    """Find all dataset.json files in output directories."""
    base_path = Path(base_dir)
    dataset_files = list(base_path.glob("**/dataset.json"))
    return dataset_files


def main():
    parser = argparse.ArgumentParser(description="Extract and combine prompts from existing datasets")
    parser.add_argument(
        "--base-dir",
        default="grpo_data_collection/outputs",
        help="Base directory to search for datasets"
    )
    parser.add_argument(
        "--output",
        default="grpo_data_collection/outputs/combined_prompts",
        help="Output directory"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Remove duplicate prompts"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EXTRACTING AND COMBINING PROMPTS")
    print("=" * 60)

    # Find all existing datasets
    print(f"\nSearching for datasets in: {args.base_dir}")
    dataset_files = find_all_datasets(args.base_dir)

    print(f"Found {len(dataset_files)} dataset files:")
    for f in dataset_files:
        print(f"  - {f}")

    # Extract prompts from each
    all_prompts = []
    for dataset_file in dataset_files:
        print(f"\nExtracting prompts from: {dataset_file.name}")
        try:
            prompts = extract_prompts_from_dataset(dataset_file)
            print(f"  Extracted {len(prompts)} prompts")
            all_prompts.extend(prompts)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Total prompts extracted: {len(all_prompts)}")

    # Deduplicate
    if args.deduplicate:
        unique_prompts = list(dict.fromkeys(all_prompts))  # Preserves order
        duplicates_removed = len(all_prompts) - len(unique_prompts)
        print(f"Removed {duplicates_removed} duplicates")
        print(f"Unique prompts: {len(unique_prompts)}")
        all_prompts = unique_prompts

    # Create output directory
    output_dir = Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save in TRL format
    trl_dir = output_dir / "trl_format"
    trl_dir.mkdir(exist_ok=True)

    trl_data = {"prompt": all_prompts}

    with open(trl_dir / "train.json", 'w') as f:
        json.dump(trl_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Saved to: {trl_dir / 'train.json'}")
    print(f"Total prompts: {len(all_prompts)}")

    # Save metadata
    metadata = {
        "source_files": [str(f) for f in dataset_files],
        "total_prompts": len(all_prompts),
        "deduplicated": args.deduplicate,
        "timestamp": timestamp
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print sample prompts
    print(f"\nSample prompts:")
    for i, prompt in enumerate(all_prompts[:3], 1):
        preview = prompt[:100].replace('\n', ' ')
        print(f"  {i}. {preview}...")

    print(f"\n{'=' * 60}")
    print("Ready for training!")
    print(f"{'=' * 60}")
    print(f"\nTo train:")
    print(f"  export PYTORCH_ENABLE_MPS_FALLBACK=1")
    print(f"  python grpo_data_collection/train_with_online_rewards.py \\")
    print(f"    --prompts {trl_dir / 'train.json'} \\")
    print(f"    --model Qwen/Qwen2.5-1.5B-Instruct \\")
    print(f"    --output grpo_trained_models/from_combined")


if __name__ == "__main__":
    main()
