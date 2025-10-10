"""Configuration management for experiments."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from .types import EpisodeConfig

load_dotenv()


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str
    output_dir: Path
    num_seeds: int = 10
    episode: EpisodeConfig = None
    question_pool_size: int = 10
    baseline_type: str = "random"  # "random" or "greedy"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.episode is None:
            self.episode = EpisodeConfig(
                persona="likes fast cars but is on a budget",
                domain="cars"
            )


def load_config(config_path: Optional[Path] = None) -> ExperimentConfig:
    """Load configuration from YAML file or use defaults."""
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested episode config
        if 'episode' in config_dict:
            episode_dict = config_dict.pop('episode')
            episode_config = EpisodeConfig(**episode_dict)
        else:
            episode_config = None

        return ExperimentConfig(
            episode=episode_config,
            **config_dict
        )
    else:
        # Default configuration
        return ExperimentConfig(
            name="mve_default",
            output_dir=Path("experiments/outputs/runs/default"),
            num_seeds=10,
            episode=EpisodeConfig(
                persona="likes fast cars but is on a budget",
                domain="cars",
                num_items=10,
                num_pairs=20,
                num_turns=5
            ),
            question_pool_size=10
        )


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Save configuration to YAML file."""
    config_dict = asdict(config)
    # Convert Path to string
    config_dict['output_dir'] = str(config_dict['output_dir'])

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_api_key() -> str:
    """Get Together API key from environment."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables")
    return api_key