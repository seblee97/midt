"""Evaluate a trained Decision Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from midt.agents.dqn_trainer import create_gridworld_env
from midt.evaluation.rollout import create_rollout_from_checkpoint
from midt.utils.config import EvalConfig, load_config
from midt.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate Decision Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to eval config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to transition data (for normalization stats)",
    )
    parser.add_argument(
        "--env-layout",
        type=str,
        required=True,
        help="Path to environment layout file",
    )
    parser.add_argument(
        "--target-returns",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0],
        help="Target returns to evaluate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Episodes per target return",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes",
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = create_gridworld_env(
        layout_path=args.env_layout,
        obs_mode="symbolic_minimal",
        max_steps=200,
    )

    # Create rollout handler
    print(f"Loading model from {args.checkpoint}...")
    rollout = create_rollout_from_checkpoint(
        checkpoint_path=args.checkpoint,
        env=env,
        data_path=args.data_path,
    )

    # Evaluate
    print(f"\nEvaluating with target returns: {args.target_returns}")
    print(f"Episodes per target: {args.num_episodes}")

    results = rollout.evaluate(
        target_returns=args.target_returns,
        num_episodes=args.num_episodes,
        deterministic=True,
    )

    # Save results
    results_path = output_dir / "eval_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n=== Evaluation Summary ===")
    for tr in args.target_returns:
        subset = results[results["target_return"] == tr]
        print(f"\nTarget Return: {tr}")
        print(f"  Mean Return: {subset['actual_return'].mean():.4f} ± {subset['actual_return'].std():.4f}")
        print(f"  Mean Length: {subset['length'].mean():.1f}")

    print(f"\nOverall Mean Return: {results['actual_return'].mean():.4f}")


if __name__ == "__main__":
    main()
