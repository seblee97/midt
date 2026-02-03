"""Run interpretability experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from midt.data.dataset import DecisionTransformerDataset, create_dataloader
from midt.data.storage import TransitionStorage
from midt.interp.attention import AttentionAnalyzer
from midt.interp.probing import ProbingExperiment, plot_probing_results
from midt.training.trainer import load_model_from_checkpoint
from midt.utils.config import InterpConfig, load_config
from midt.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run interpretability experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to interp config YAML file",
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
        help="Path to transition data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/interp",
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--run-attention",
        action="store_true",
        help="Run attention analysis",
    )
    parser.add_argument(
        "--run-probing",
        action="store_true",
        help="Run probing experiments",
    )
    parser.add_argument(
        "--probe-target",
        type=str,
        default="action",
        choices=["action", "rtg", "timestep"],
        help="Target for probing",
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint)
    print(f"Model: {model}")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    storage = TransitionStorage.load(args.data_path)

    dataset = DecisionTransformerDataset(
        storage=storage,
        context_length=model.context_length,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    # Run attention analysis
    if args.run_attention:
        print("\n=== Attention Analysis ===")
        analyzer = AttentionAnalyzer(model)

        # Get a sample batch
        batch = next(iter(dataloader))

        # Get attention weights for each layer
        for layer in range(model.num_layers):
            attn = analyzer.get_attention_weights(
                states=batch["states"],
                actions=batch["actions"],
                returns_to_go=batch["returns_to_go"],
                timesteps=batch["timesteps"],
                attention_mask=batch["attention_mask"],
                layer=layer,
            )

            # Plot attention heatmap for first head
            fig = analyzer.plot_attention_heatmap(attn, layer=layer, head=0)
            fig.savefig(output_dir / f"attention_layer{layer}_head0.png", dpi=150)
            plt.close(fig)

            # Plot modality attention matrix
            fig = analyzer.plot_modality_attention_matrix(attn, layer=layer)
            fig.savefig(output_dir / f"modality_attention_layer{layer}.png", dpi=150)
            plt.close(fig)

            # Print summary
            summary = analyzer.compute_modality_attention_summary(attn)
            print(f"\nLayer {layer} Modality Attention:")
            print(summary.to_string(index=False))

        # Analyze head specialization
        print("\nAnalyzing head specialization...")
        head_stats = analyzer.analyze_head_specialization(dataloader, num_batches=10)
        head_stats.to_csv(output_dir / "head_specialization.csv", index=False)
        print(f"Head specialization saved to {output_dir / 'head_specialization.csv'}")

    # Run probing experiments
    if args.run_probing:
        print("\n=== Probing Experiments ===")
        prober = ProbingExperiment(model)

        num_batches = args.num_samples // 32

        print(f"Probing for '{args.probe_target}' across all layers...")
        results = prober.probe_all_layers(
            dataloader=dataloader,
            target=args.probe_target,
            modality="state",
            num_batches=num_batches,
            num_epochs=50,
        )

        # Save results
        results.to_csv(output_dir / f"probing_{args.probe_target}.csv", index=False)
        print(f"\nProbing results saved to {output_dir / f'probing_{args.probe_target}.csv'}")

        # Plot results
        metric = "accuracy" if args.probe_target != "rtg" else "mse"
        fig = plot_probing_results(
            results,
            metric=metric,
            title=f"Probing for '{args.probe_target}' across layers",
        )
        fig.savefig(output_dir / f"probing_{args.probe_target}.png", dpi=150)
        plt.close(fig)

        print("\nProbing Results:")
        print(results.to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
