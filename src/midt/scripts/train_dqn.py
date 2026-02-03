"""Train DQN and collect trajectory data."""

from __future__ import annotations

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

    # Create environment
    env = create_gridworld_env(
        layout_path=config.env_layout_path,
        obs_mode=config.obs_mode,
        max_steps=config.max_episode_steps,
    )

    # Create trainer
    trainer = DQNTrainer(
        env=env,
        config=config,
        output_dir=output_dir,
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
    print(f"Data saved to: {output_dir / 'data' / 'transitions.h5'}")


if __name__ == "__main__":
    main()
