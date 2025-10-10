"""Export GRPO data to HuggingFace datasets format."""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from grpo_collector import GRPODataPoint


def export_to_json(dataset: List[GRPODataPoint], output_path: str):
    """Export dataset to JSON format."""
    data = []
    for point in dataset:
        data.append({
            "prompt": point.prompt,
            "completions": point.completions,
            "rewards": point.rewards,
            "advantages": point.advantages,
            "metadata": point.metadata
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(data)} data points to {output_path}")


def export_to_jsonl(dataset: List[GRPODataPoint], output_path: str):
    """Export dataset to JSONL format (one JSON object per line)."""
    with open(output_path, 'w') as f:
        for point in dataset:
            line = json.dumps({
                "prompt": point.prompt,
                "completions": point.completions,
                "rewards": point.rewards,
                "advantages": point.advantages,
                "metadata": point.metadata
            })
            f.write(line + '\n')

    print(f"Exported {len(dataset)} data points to {output_path}")


def export_for_trl(dataset: List[GRPODataPoint], output_dir: str):
    """Export dataset in format ready for TRL GRPOTrainer."""
    os.makedirs(output_dir, exist_ok=True)

    # Flatten the dataset for TRL
    prompts = []
    all_completions = []
    all_rewards = []

    for point in dataset:
        prompts.append(point.prompt)
        all_completions.append(point.completions)
        all_rewards.append(point.rewards)

    # Create the dataset dictionary
    trl_dataset = {
        "prompt": prompts,
        "completions": all_completions,
        "rewards": all_rewards
    }

    # Save as JSON
    with open(os.path.join(output_dir, "train.json"), 'w') as f:
        json.dump(trl_dataset, f, indent=2)

    # Also save as separate files for easy loading
    pd.DataFrame({"prompt": prompts}).to_csv(
        os.path.join(output_dir, "prompts.csv"), index=False
    )

    # Save completions and rewards as numpy-friendly format
    import numpy as np
    np.save(os.path.join(output_dir, "completions.npy"), all_completions)
    np.save(os.path.join(output_dir, "rewards.npy"), all_rewards)

    print(f"Exported TRL-formatted dataset to {output_dir}")
    print(f"  - {len(prompts)} prompts")
    print(f"  - {sum(len(c) for c in all_completions)} total completions")


def export_for_analysis(dataset: List[GRPODataPoint], output_path: str):
    """Export dataset with analysis-friendly structure."""
    rows = []

    for point_idx, point in enumerate(dataset):
        for comp_idx, (question, reward, advantage) in enumerate(
            zip(point.completions, point.rewards, point.advantages)
        ):
            row = {
                "data_point_id": point_idx,
                "depth": point.metadata.get("depth", 0),
                "persona": point.metadata.get("persona", ""),
                "domain": point.metadata.get("domain", ""),
                "question": question,
                "reward": reward,
                "advantage": advantage,
                "question_index": comp_idx,
                "prompt_preview": point.prompt[:100] + "..." if len(point.prompt) > 100 else point.prompt
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Exported analysis dataset to {output_path}")
    print(f"Dataset statistics:")
    print(f"  - Average reward: {df['reward'].mean():.4f}")
    print(f"  - Std reward: {df['reward'].std():.4f}")
    print(f"  - Min reward: {df['reward'].min():.4f}")
    print(f"  - Max reward: {df['reward'].max():.4f}")

    # Group by depth
    depth_stats = df.groupby('depth')['reward'].agg(['mean', 'std', 'min', 'max'])
    print("\nRewards by depth:")
    print(depth_stats)


def create_train_test_split(
    dataset: List[GRPODataPoint],
    output_dir: str,
    test_ratio: float = 0.1
):
    """Create train/test split of the dataset."""
    import random
    random.seed(42)

    # Shuffle and split
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    # Export train and test sets
    os.makedirs(output_dir, exist_ok=True)
    export_to_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    export_to_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))

    print(f"Created train/test split:")
    print(f"  - Train: {len(train_data)} examples")
    print(f"  - Test: {len(test_data)} examples")


def export_all_formats(dataset: List[GRPODataPoint], base_output_dir: str):
    """Export dataset in all useful formats."""
    os.makedirs(base_output_dir, exist_ok=True)

    # JSON formats
    export_to_json(dataset, os.path.join(base_output_dir, "dataset.json"))
    export_to_jsonl(dataset, os.path.join(base_output_dir, "dataset.jsonl"))

    # TRL format
    export_for_trl(dataset, os.path.join(base_output_dir, "trl_format"))

    # Analysis format
    export_for_analysis(dataset, os.path.join(base_output_dir, "analysis.csv"))

    # Train/test split
    create_train_test_split(dataset, os.path.join(base_output_dir, "splits"))

    print(f"\nAll exports completed to {base_output_dir}")