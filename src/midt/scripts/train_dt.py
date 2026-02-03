"""Train Decision Transformer on collected data."""

from __future__ import annotations

import argparse
from pathlib import Path

from midt.data.dataset import DecisionTransformerDataset
from midt.data.storage import TransitionStorage
from midt.models.decision_transformer import DecisionTransformer
from midt.training.trainer import DecisionTransformerTrainer
from midt.utils.config import DTConfig, TrainingConfig, load_config
from midt.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML file (optional, uses defaults if not provided)",
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

    # Load configs
    train_config = load_config(args.config, TrainingConfig)

    if args.model_config:
        model_config = load_config(args.model_config, DTConfig)
    else:
        model_config = DTConfig()

    # Override from command line
    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.seed is not None:
        train_config.seed = args.seed

    # Set seed
    set_seed(train_config.seed)

    # Create output directory
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {train_config.data_path}...")
    storage = TransitionStorage.load(train_config.data_path)
    metadata = storage.get_metadata()
    print(f"Loaded {metadata['num_episodes']} episodes, {metadata['total_transitions']} transitions")

    # Create dataset
    dataset = DecisionTransformerDataset(
        storage=storage,
        context_length=train_config.context_length,
        rtg_scale=train_config.rtg_scale,
    )

    # Split into train/eval
    train_dataset, eval_dataset = dataset.split(
        train_ratio=train_config.train_split,
        seed=train_config.seed,
    )
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Infer state dimension
    state_dim = model_config.state_dim or dataset.state_dim

    # Create model
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=model_config.action_dim,
        embed_dim=model_config.embed_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        context_length=train_config.context_length,
        max_timestep=model_config.max_timestep,
        dropout=model_config.dropout,
        obs_type=model_config.obs_type,
    )
    print(f"\nModel: {model}")

    # Create trainer
    trainer = DecisionTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
        output_dir=output_dir,
        eval_dataset=eval_dataset,
    )

    # Train
    print(f"\nTraining for {train_config.max_steps} steps...")
    metrics = trainer.train()

    print("\nTraining complete!")
    print(f"Final train loss: {metrics['train_loss']:.4f}")
    print(f"Final train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
