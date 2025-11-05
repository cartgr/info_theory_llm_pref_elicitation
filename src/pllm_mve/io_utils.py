"""I/O utilities for logging and data management."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd


def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load data from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Save data as pickle."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load data from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


class EpisodeLogger:
    """Logger for episode data."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs: List[Dict] = []

    def log_turn(
        self,
        turn: int,
        question: str,
        answer: str,
        score: float,
        reward: float,
        **kwargs
    ) -> None:
        """Log a single turn."""
        log_entry = {
            "turn": turn,
            "question": question,
            "answer": answer,
            "score": score,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.logs.append(log_entry)

    def save(self, seed: Optional[int] = None, suffix: Optional[str] = None) -> None:
        """Save logs to file."""
        if suffix is not None:
            file_suffix = suffix
        elif seed is not None:
            file_suffix = f"_seed{seed}"
        else:
            file_suffix = ""

        json_path = self.output_dir / f"episode_logs{file_suffix}.json"
        save_json(self.logs, json_path)

        # Also save as CSV for easy analysis
        if self.logs:
            df = pd.DataFrame(self.logs)
            csv_path = self.output_dir / f"episode_logs{file_suffix}.csv"
            df.to_csv(csv_path, index=False)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.logs:
            return {}

        scores = [log["score"] for log in self.logs]
        rewards = [log["reward"] for log in self.logs]

        return {
            "total_turns": len(self.logs),
            "final_score": scores[-1] if scores else 0,
            "initial_score": scores[0] if scores else 0,
            "total_reward": sum(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "score_trajectory": scores,
            "reward_trajectory": rewards
        }


class ExperimentLogger:
    """Logger for full experiment across multiple seeds."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.episode_summaries: List[Dict] = []

    def add_episode(self, seed: int, summary: Dict) -> None:
        """Add an episode summary."""
        summary["seed"] = seed
        self.episode_summaries.append(summary)

    def save_summary(self) -> None:
        """Save experiment summary."""
        if not self.episode_summaries:
            return

        # Save raw data
        json_path = self.output_dir / "experiment_summary.json"
        save_json(self.episode_summaries, json_path)

        # Save aggregated statistics
        df = pd.DataFrame(self.episode_summaries)
        stats = {
            "num_episodes": len(self.episode_summaries),
            "mean_total_reward": df["total_reward"].mean(),
            "std_total_reward": df["total_reward"].std(),
            "mean_final_score": df["final_score"].mean(),
            "std_final_score": df["final_score"].std(),
            "success_rate": (df["total_reward"] > 0).mean()
        }

        stats_path = self.output_dir / "experiment_stats.json"
        save_json(stats, stats_path)

        # Save as CSV
        csv_path = self.output_dir / "experiment_summary.csv"
        df.to_csv(csv_path, index=False)

        print(f"\nExperiment Summary:")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  Mean Total Reward: {stats['mean_total_reward']:.4f} ± {stats['std_total_reward']:.4f}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")