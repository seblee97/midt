"""Train DQN and collect trajectory data."""

from __future__ import annotations

# Disable wandb by default (must be before any imports that might trigger it)
import os
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
from pathlib import Path

from midt.agents.dqn_trainer import DQNTrainer, create_gridworld_env
from midt.utils.config import DQNConfig, load_config
from midt.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train DQN and collect data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to DQN config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, DQNConfig)

    # Override from command line
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    # Set seed
    set_seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pixel preprocessing kwargs (passed to both training and video envs)
    pixel_kwargs = dict(
        obs_resize=config.obs_resize,
        obs_grayscale=config.obs_grayscale,
        frame_stack=config.frame_stack,
    )

    # Create environment (with render support if recording videos)
    env = create_gridworld_env(
        layout_path=config.env_layout_path,
        obs_mode=config.obs_mode,
        max_steps=config.max_episode_steps,
        render_mode="rgb_array" if config.record_video else None,
        **pixel_kwargs,
    )

    # Create separate eval environment for test rollouts if recording
    video_env = None
    if config.record_video:
        video_env = create_gridworld_env(
            layout_path=config.env_layout_path,
            obs_mode=config.obs_mode,
            max_steps=config.max_episode_steps,
            render_mode="rgb_array",
            **pixel_kwargs,
        )

    # Create trainer
    trainer = DQNTrainer(
        env=env,
        config=config,
        output_dir=output_dir,
        video_env=video_env,
    )

    # Save config
    trainer.save_config()

    # Train
    print(f"Training DQN for {config.total_timesteps} timesteps...")
    results = trainer.train()

    print("\nTraining complete!")
    print(f"Episodes collected: {results['num_episodes']}")
    print(f"Total transitions: {results['total_transitions']}")
    print(f"Mean episode reward: {results.get('mean_reward', 'N/A'):.4f}")
    print(f"Output saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
