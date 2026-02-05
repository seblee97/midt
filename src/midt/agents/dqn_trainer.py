"""DQN training with integrated data collection."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from midt.agents.callbacks import EpisodeLoggerCallback, TransitionCollectorCallback, VideoRecorderCallback
from midt.data.storage import TransitionStorage
from midt.utils.config import DQNConfig


class DQNTrainer:
    """Wrapper around SB3 DQN with integrated transition collection.

    This trainer captures ALL transitions that occur during training
    for use in offline RL (Decision Transformer training).
    """

    def __init__(
        self,
        env: gym.Env,
        config: DQNConfig,
        output_dir: str | Path,
        eval_env: Optional[gym.Env] = None,
        video_env: Optional[gym.Env] = None,
    ):
        """Initialize the trainer.

        Args:
            env: Training environment.
            config: DQN configuration.
            output_dir: Directory for outputs (data, logs, checkpoints).
            eval_env: Optional separate environment for evaluation.
            video_env: Optional environment for video recording (must have render_mode="rgb_array").
        """
        self.env = env
        self.config = config
        self.eval_env = eval_env
        self.video_env = video_env

        # Set wandb mode based on config
        if config.use_wandb:
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "disabled"

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config.run_name}" if config.run_name else timestamp
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Initialize storage
        self.storage = TransitionStorage(
            self.output_dir / "data" / "transitions.h5",
            mode="w",
        )

        # Select policy based on observation mode
        policy = "CnnPolicy" if config.obs_mode in ("pixels", "pixel") else "MlpPolicy"

        # Initialize DQN model
        self.model = DQN(
            policy=policy,
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            gamma=config.gamma,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            target_update_interval=config.target_update_interval,
            train_freq=config.train_freq,
            verbose=1,
            tensorboard_log=str(self.output_dir / "logs" / "tb_logs"),
            seed=config.seed,
        )

        # Initialize callbacks
        self.collector_callback = TransitionCollectorCallback(
            storage=self.storage,
            save_freq=10000,
            verbose=1,
        )
        self.logger_callback = EpisodeLoggerCallback(verbose=0)

        # Optional WandB logging
        self._wandb_run = None
        if config.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name,
                config=self.config.model_dump(),
                dir=str(self.output_dir / "logs"),
            )
        except ImportError:
            print("wandb not installed, skipping W&B logging")

    def train(self) -> dict[str, Any]:
        """Train the DQN agent and collect all transitions.

        Returns:
            Dictionary with training statistics.
        """
        callbacks = [self.collector_callback, self.logger_callback]

        # Add evaluation callback if eval env provided
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.output_dir / "checkpoints"),
                log_path=str(self.output_dir / "logs"),
                eval_freq=5000,
                n_eval_episodes=10,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Add video recording callback if enabled
        if self.config.record_video and self.video_env is not None:
            video_callback = VideoRecorderCallback(
                eval_env=self.video_env,
                video_dir=self.output_dir / "videos",
                record_freq=self.config.video_freq,
                log_to_wandb=self.config.use_wandb,
                fps=self.config.video_fps,
                verbose=1,
            )
            callbacks.append(video_callback)

        # Train
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        # Final save
        self.storage.save()

        # Compute and save statistics
        stats = self.storage.compute_and_save_statistics()

        # Save model
        self.model.save(self.output_dir / "checkpoints" / "final_model")

        # Get training statistics
        episode_stats = self.logger_callback.get_statistics()
        storage_meta = self.storage.get_metadata()

        result = {
            "num_episodes": storage_meta["num_episodes"],
            "total_transitions": storage_meta.get("total_transitions", 0),
            "state_dim": storage_meta.get("state_dim"),
            **episode_stats,
            **{f"data_{k}": v.tolist() if isinstance(v, np.ndarray) else v
               for k, v in stats.items()},
        }

        # Log to WandB
        if self._wandb_run is not None:
            import wandb
            wandb.log(result)
            wandb.finish()

        return result

    def save_config(self) -> None:
        """Save configuration to output directory."""
        import yaml

        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.model_dump(), f)


def create_gridworld_env(
    layout_path: str | Path,
    obs_mode: str = "symbolic_minimal",
    max_steps: Optional[int] = 200,
    posner_mode: bool = False,
    render_mode: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    """Create a GridWorld environment.

    Args:
        layout_path: Path to layout file.
        obs_mode: Observation mode.
        max_steps: Maximum steps per episode.
        posner_mode: Enable Posner cueing.
        render_mode: Render mode (e.g., "rgb_array" for video recording).
        **kwargs: Additional environment arguments.

    Returns:
        Gymnasium environment.
    """
    # Import gridworld_env (assuming it's installed or in path)
    import sys
    gridworld_path = Path("/Users/sebastianlee/Dropbox/Documents/Research/Projects/gridworld_env/src")
    if str(gridworld_path) not in sys.path:
        sys.path.insert(0, str(gridworld_path))

    from gridworld_env import GridWorldEnv

    # Only flatten for symbolic observations, not pixels
    flatten = obs_mode not in ("pixels")

    env = GridWorldEnv(
        layout=layout_path,
        obs_mode=obs_mode,
        max_steps=max_steps,
        posner_mode=posner_mode,
        flatten_obs=flatten,
        render_mode=render_mode,
        **kwargs,
    )

    return env
