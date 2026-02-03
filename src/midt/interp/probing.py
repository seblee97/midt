"""Linear probing for feature detection in Decision Transformer."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from midt.interp.cache import ActivationCache
from midt.models.decision_transformer import DecisionTransformer


class LinearProbe(nn.Module):
    """Linear probe for detecting features in representations.

    Used to answer questions like:
    - Does the model represent the agent's position?
    - Does it encode which key is held?
    - Does it track the return-to-go?
    """

    def __init__(self, input_dim: int, num_classes: int):
        """Initialize the probe.

        Args:
            input_dim: Dimension of input representations.
            num_classes: Number of output classes (or 1 for regression).
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Output tensor (..., num_classes)
        """
        return self.linear(x)


class ProbingExperiment:
    """Run probing experiments on Decision Transformer activations.

    Supports probing for various features:
    - position: Agent's (row, col) position
    - held_key: Which key is held (none/red/blue)
    - posner_cue: Current Posner cue
    - action: Action that was taken
    - return: Future return (regression)
    """

    def __init__(
        self,
        model: DecisionTransformer,
        device: str = "cpu",
    ):
        """Initialize the probing experiment.

        Args:
            model: Decision Transformer model.
            device: Device to run on.
        """
        self.model = model
        self.device = device
        self.cache = ActivationCache(model)

    def collect_activations(
        self,
        dataloader: DataLoader,
        layer: int,
        modality: str = "state",
        num_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Collect activations and corresponding labels from data.

        Args:
            dataloader: DataLoader with samples.
            layer: Layer to extract activations from.
            modality: Token type ("return", "state", "action").
            num_batches: Number of batches to collect (None for all).

        Returns:
            Tuple of (activations, labels_dict) where labels_dict contains
            various potential probe targets.
        """
        self.model.eval()

        all_activations = []
        all_actions = []
        all_rtg = []
        all_timesteps = []

        modality_offset = {"return": 0, "state": 1, "action": 2}[modality]

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
                if num_batches is not None and i >= num_batches:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Run forward pass with caching
                self.cache.run_with_cache(
                    states=batch["states"],
                    actions=batch["actions"],
                    returns_to_go=batch["returns_to_go"],
                    timesteps=batch["timesteps"],
                    attention_mask=batch["attention_mask"],
                )

                # Get hidden states
                hidden = self.cache.get_hidden(layer)
                if hidden is None:
                    continue

                # Extract modality-specific positions
                K = batch["states"].shape[1]
                indices = torch.arange(modality_offset, 3 * K, 3, device=self.device)
                modality_hidden = hidden[:, indices, :]  # (B, K, D)

                # Flatten batch and sequence dimensions
                B = modality_hidden.shape[0]
                modality_hidden = modality_hidden.reshape(-1, modality_hidden.shape[-1])

                # Get corresponding labels
                mask = batch["attention_mask"].bool()
                mask_flat = mask.reshape(-1)

                # Only keep valid (non-padded) positions
                all_activations.append(modality_hidden[mask_flat].cpu())
                all_actions.append(batch["actions"].reshape(-1)[mask_flat].cpu())
                all_rtg.append(batch["returns_to_go"].reshape(-1)[mask_flat].cpu())
                all_timesteps.append(batch["timesteps"].reshape(-1)[mask_flat].cpu())

        activations = torch.cat(all_activations, dim=0)
        labels = {
            "action": torch.cat(all_actions, dim=0),
            "rtg": torch.cat(all_rtg, dim=0),
            "timestep": torch.cat(all_timesteps, dim=0),
        }

        return activations, labels

    def train_probe(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        num_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_split: float = 0.1,
        regression: bool = False,
    ) -> Tuple[LinearProbe, Dict[str, float]]:
        """Train a linear probe.

        Args:
            activations: Activation tensor (N, embed_dim).
            labels: Label tensor (N,) for classification or (N, 1) for regression.
            num_classes: Number of classes (1 for regression).
            num_epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            val_split: Validation split fraction.
            regression: If True, treat as regression task.

        Returns:
            Tuple of (trained probe, metrics dict).
        """
        # Split data
        N = len(activations)
        perm = torch.randperm(N)
        val_size = int(N * val_split)

        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        train_acts = activations[train_idx].to(self.device)
        train_labels = labels[train_idx].to(self.device)
        val_acts = activations[val_idx].to(self.device)
        val_labels = labels[val_idx].to(self.device)

        # Create probe
        probe = LinearProbe(activations.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        # Training loop
        train_dataset = TensorDataset(train_acts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_metric = -float("inf") if not regression else float("inf")
        best_probe_state = None

        for epoch in range(num_epochs):
            probe.train()
            for batch_acts, batch_labels in train_loader:
                optimizer.zero_grad()

                logits = probe(batch_acts)

                if regression:
                    loss = F.mse_loss(logits.squeeze(-1), batch_labels.float())
                else:
                    loss = F.cross_entropy(logits, batch_labels.long())

                loss.backward()
                optimizer.step()

            # Validation
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_acts)

                if regression:
                    val_loss = F.mse_loss(val_logits.squeeze(-1), val_labels.float()).item()
                    val_metric = -val_loss  # Negative for "higher is better"
                else:
                    val_loss = F.cross_entropy(val_logits, val_labels.long()).item()
                    val_preds = val_logits.argmax(dim=-1)
                    val_acc = (val_preds == val_labels).float().mean().item()
                    val_metric = val_acc

                if (not regression and val_metric > best_val_metric) or \
                   (regression and val_metric > best_val_metric):
                    best_val_metric = val_metric
                    best_probe_state = probe.state_dict().copy()

        # Load best probe
        probe.load_state_dict(best_probe_state)

        # Final evaluation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_acts)

            if regression:
                metrics = {
                    "mse": F.mse_loss(val_logits.squeeze(-1), val_labels.float()).item(),
                    "mae": F.l1_loss(val_logits.squeeze(-1), val_labels.float()).item(),
                }
            else:
                val_preds = val_logits.argmax(dim=-1)
                metrics = {
                    "accuracy": (val_preds == val_labels).float().mean().item(),
                    "loss": F.cross_entropy(val_logits, val_labels.long()).item(),
                }

        return probe, metrics

    def probe_all_layers(
        self,
        dataloader: DataLoader,
        target: str,
        modality: str = "state",
        num_batches: Optional[int] = None,
        num_epochs: int = 100,
        **probe_kwargs,
    ) -> pd.DataFrame:
        """Train probes at every layer and return accuracy per layer.

        Args:
            dataloader: Data loader.
            target: Target to probe ("action", "rtg", "timestep").
            modality: Token modality to probe.
            num_batches: Batches to use.
            num_epochs: Probe training epochs.
            **probe_kwargs: Additional arguments for train_probe.

        Returns:
            DataFrame with layer-wise probe results.
        """
        results = []

        # Include input embeddings (layer -1), all hidden layers, and final
        layers_to_probe = list(range(self.model.num_layers)) + [-1]

        for layer in layers_to_probe:
            print(f"Probing layer {layer}...")

            # Collect activations
            activations, labels = self.collect_activations(
                dataloader, layer, modality, num_batches
            )

            # Get target labels
            if target not in labels:
                raise ValueError(f"Unknown target: {target}")

            target_labels = labels[target]

            # Determine number of classes
            if target == "action":
                num_classes = self.model.action_dim
                regression = False
            elif target == "rtg":
                num_classes = 1
                regression = True
            elif target == "timestep":
                num_classes = min(int(target_labels.max().item()) + 1, 100)
                regression = False
            else:
                raise ValueError(f"Unknown target: {target}")

            # Train probe
            probe, metrics = self.train_probe(
                activations,
                target_labels,
                num_classes,
                num_epochs=num_epochs,
                regression=regression,
                **probe_kwargs,
            )

            result = {
                "layer": layer,
                "target": target,
                "modality": modality,
                **metrics,
            }
            results.append(result)

        return pd.DataFrame(results)


def plot_probing_results(
    results: pd.DataFrame,
    metric: str = "accuracy",
    title: Optional[str] = None,
) -> "plt.Figure":
    """Plot probing results across layers.

    Args:
        results: DataFrame from probe_all_layers.
        metric: Metric to plot.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    layers = results["layer"].values
    values = results[metric].values

    ax.plot(layers, values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Probing {results['target'].iloc[0]} across layers")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
