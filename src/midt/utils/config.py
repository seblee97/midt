"""Configuration dataclasses and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class DQNConfig(BaseModel):
    """Configuration for DQN training."""

    # Environment
    env_layout_path: str = Field(description="Path to gridworld layout file")
    obs_mode: str = Field(default="symbolic_minimal", description="Observation mode")
    max_episode_steps: Optional[int] = Field(default=200, description="Max steps per episode")

    # Pixel preprocessing (only used when obs_mode="pixels")
    obs_resize: Optional[list[int]] = Field(default=None, description="Resize pixel obs to [H, W] before replay buffer")
    obs_grayscale: bool = Field(default=False, description="Convert pixel obs to grayscale (1 channel)")
    frame_stack: int = Field(default=1, description="Number of frames to stack along the channel axis")

    # DQN hyperparameters
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    buffer_size: int = Field(default=100_000, description="Replay buffer size")
    learning_starts: int = Field(default=1000, description="Steps before learning")
    batch_size: int = Field(default=32, description="Batch size for training")
    gamma: float = Field(default=0.99, description="Discount factor")
    exploration_fraction: float = Field(default=0.1, description="Fraction of training for exploration")
    exploration_initial_eps: float = Field(default=1.0, description="Initial exploration epsilon")
    exploration_final_eps: float = Field(default=0.05, description="Final exploration epsilon")
    target_update_interval: int = Field(default=1000, description="Target network update frequency")
    train_freq: int = Field(default=4, description="Training frequency")

    # Training
    total_timesteps: int = Field(default=100_000, description="Total training timesteps")
    seed: int = Field(default=42, description="Random seed")

    # Logging
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str = Field(default="midt-dqn", description="W&B project name")
    run_name: Optional[str] = Field(default=None, description="Run name")
    log_interval: int = Field(default=100, description="Logging interval")

    # Video recording
    record_video: bool = Field(default=False, description="Record training and test videos")
    video_freq: int = Field(default=10, description="Record videos every N episodes")
    video_fps: int = Field(default=10, description="Video playback frames per second")

    # Output
    output_dir: str = Field(default="outputs", description="Output directory")


class DTConfig(BaseModel):
    """Configuration for Decision Transformer model."""

    # Model architecture
    embed_dim: int = Field(default=128, description="Embedding dimension")
    num_layers: int = Field(default=3, description="Number of transformer layers")
    num_heads: int = Field(default=4, description="Number of attention heads")
    dropout: float = Field(default=0.1, description="Dropout rate")
    max_timestep: int = Field(default=1000, description="Maximum timestep for embedding")

    # Input dimensions (inferred from data if None)
    state_dim: Optional[int] = Field(default=None, description="State dimension")
    action_dim: int = Field(default=4, description="Number of actions")

    # Observation type
    obs_type: str = Field(default="symbolic", description="Observation type (symbolic or pixels)")


class TrainingConfig(BaseModel):
    """Configuration for Decision Transformer training."""

    # Data
    data_path: str = Field(description="Path to transition data")
    context_length: int = Field(default=20, description="Context length K")
    rtg_scale: float = Field(default=1.0, description="Return-to-go scaling factor")
    train_split: float = Field(default=0.9, description="Train/eval split ratio")

    # Optimization
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    batch_size: int = Field(default=64, description="Batch size")
    max_steps: int = Field(default=100_000, description="Maximum training steps")
    warmup_steps: int = Field(default=1000, description="Warmup steps")
    scheduler_type: str = Field(default="cosine", description="LR scheduler type")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping")

    # Evaluation
    eval_every: int = Field(default=1000, description="Evaluation interval")
    save_every: int = Field(default=5000, description="Checkpoint interval")

    # Logging
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str = Field(default="midt-dt", description="W&B project name")
    run_name: Optional[str] = Field(default=None, description="Run name")

    # Reproducibility
    seed: int = Field(default=42, description="Random seed")

    # Output
    output_dir: str = Field(default="outputs", description="Output directory")


class EvalConfig(BaseModel):
    """Configuration for Decision Transformer evaluation."""

    # Model
    checkpoint_path: str = Field(description="Path to model checkpoint")

    # Environment
    env_layout_path: str = Field(description="Path to gridworld layout file")
    obs_mode: str = Field(default="symbolic_minimal", description="Observation mode")
    max_episode_steps: int = Field(default=200, description="Max steps per episode")

    # Evaluation settings
    target_returns: list[float] = Field(default=[0.5, 0.8, 1.0], description="Target returns to condition on")
    num_episodes: int = Field(default=20, description="Episodes per target return")
    deterministic: bool = Field(default=True, description="Use deterministic actions")
    render: bool = Field(default=False, description="Render episodes")

    # Output
    output_dir: str = Field(default="outputs/eval", description="Output directory")
    seed: int = Field(default=42, description="Random seed")


class InterpConfig(BaseModel):
    """Configuration for interpretability experiments."""

    # Model
    checkpoint_path: str = Field(description="Path to model checkpoint")

    # Data
    data_path: str = Field(description="Path to transition data")
    num_samples: int = Field(default=1000, description="Number of samples for analysis")

    # Probing
    probe_targets: list[str] = Field(
        default=["position", "held_key", "posner_cue"],
        description="Features to probe for",
    )
    probe_epochs: int = Field(default=100, description="Epochs for probe training")
    probe_lr: float = Field(default=1e-3, description="Probe learning rate")

    # Attention analysis
    analyze_heads: bool = Field(default=True, description="Analyze attention heads")

    # Patching
    run_patching: bool = Field(default=True, description="Run activation patching")
    patching_samples: int = Field(default=100, description="Samples for patching")

    # Output
    output_dir: str = Field(default="outputs/interp", description="Output directory")
    seed: int = Field(default=42, description="Random seed")


@dataclass
class ExperimentConfig:
    """Combined configuration for a full experiment."""

    dqn: Optional[DQNConfig] = None
    dt: Optional[DTConfig] = None
    training: Optional[TrainingConfig] = None
    eval: Optional[EvalConfig] = None
    interp: Optional[InterpConfig] = None


def load_config(path: str | Path, config_class: type[BaseModel]) -> BaseModel:
    """Load a configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.
        config_class: The Pydantic model class to validate against.

    Returns:
        Validated configuration object.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def save_config(config: BaseModel, path: str | Path) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: Configuration object to save.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
