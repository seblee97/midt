"""Activation patching for causal interventions."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from midt.interp.cache import ActivationCache
from midt.models.decision_transformer import DecisionTransformer


class ActivationPatcher:
    """Causal intervention via activation patching.

    Allows answering counterfactual questions like:
    - What if the model had seen a different RTG?
    - What if state representations were swapped between inputs?
    - Which components are causally important for a behavior?
    """

    def __init__(
        self,
        model: DecisionTransformer,
        device: str = "cpu",
    ):
        """Initialize the patcher.

        Args:
            model: Decision Transformer model.
            device: Device to run on.
        """
        self.model = model
        self.device = device
        self.cache = ActivationCache(model)

    def get_clean_logits(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run model and return clean (unpatched) logits.

        Args:
            states, actions, returns_to_go, timesteps: Model inputs.
            attention_mask: Optional attention mask.

        Returns:
            Action logits (batch, K, action_dim)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(
                states, actions, returns_to_go, timesteps, attention_mask
            )

    def patch_hidden_states(
        self,
        clean_inputs: Dict[str, torch.Tensor],
        patch_inputs: Dict[str, torch.Tensor],
        layer: int,
        positions: List[int],
    ) -> torch.Tensor:
        """Run model with hidden states patched from another input.

        Args:
            clean_inputs: Dictionary with clean input tensors.
            patch_inputs: Dictionary with input tensors to get patch from.
            layer: Layer at which to patch.
            positions: Token positions to patch (in 3K sequence).

        Returns:
            Action logits after patching.
        """
        self.model.eval()

        # First, get patch activations
        self.cache.run_with_cache(
            patch_inputs["states"],
            patch_inputs["actions"],
            patch_inputs["returns_to_go"],
            patch_inputs["timesteps"],
            patch_inputs.get("attention_mask"),
        )
        patch_hidden = self.cache.get_hidden(layer).clone()

        # Define patching hook
        def patch_hook(module, input, output):
            # Patch specified positions
            patched = output.clone()
            for pos in positions:
                patched[:, pos, :] = patch_hidden[:, pos, :]
            return patched

        # Register hook
        target_block = self.model.blocks[layer]
        handle = target_block.register_forward_hook(patch_hook)

        try:
            # Run with patching
            with torch.no_grad():
                patched_logits = self.model(
                    clean_inputs["states"],
                    clean_inputs["actions"],
                    clean_inputs["returns_to_go"],
                    clean_inputs["timesteps"],
                    clean_inputs.get("attention_mask"),
                )
        finally:
            handle.remove()

        return patched_logits

    def compute_patching_effect(
        self,
        clean_inputs: Dict[str, torch.Tensor],
        patch_inputs: Dict[str, torch.Tensor],
        metric_fn: Callable[[torch.Tensor], float],
        layers: Optional[List[int]] = None,
        positions: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute effect of patching at each (layer, position).

        Args:
            clean_inputs: Clean input tensors.
            patch_inputs: Tensors to get patch activations from.
            metric_fn: Function that takes logits and returns a scalar metric.
            layers: Layers to test (default: all).
            positions: Positions to test (default: all).

        Returns:
            Array of shape (num_layers, num_positions) with patching effects.
        """
        num_layers = self.model.num_layers
        K = clean_inputs["states"].shape[1]
        total_positions = 3 * K

        if layers is None:
            layers = list(range(num_layers))
        if positions is None:
            positions = list(range(total_positions))

        # Get clean metric
        clean_logits = self.get_clean_logits(
            clean_inputs["states"],
            clean_inputs["actions"],
            clean_inputs["returns_to_go"],
            clean_inputs["timesteps"],
            clean_inputs.get("attention_mask"),
        )
        clean_metric = metric_fn(clean_logits)

        # Compute patching effects
        effects = np.zeros((len(layers), len(positions)))

        for i, layer in enumerate(tqdm(layers, desc="Patching layers")):
            for j, pos in enumerate(positions):
                patched_logits = self.patch_hidden_states(
                    clean_inputs, patch_inputs, layer, [pos]
                )
                patched_metric = metric_fn(patched_logits)
                effects[i, j] = patched_metric - clean_metric

        return effects

    def find_critical_components(
        self,
        dataloader: torch.utils.data.DataLoader,
        comparison_fn: Callable[[Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        num_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Identify which components are most important for behavior.

        Args:
            dataloader: DataLoader with samples.
            comparison_fn: Function that takes a batch and returns
                (clean_inputs, patch_inputs) pair.
            metric_fn: Function that takes (clean_logits, patched_logits)
                and returns a scalar measuring the change.
            num_samples: Number of samples to average over.

        Returns:
            Dictionary with aggregated patching effects.
        """
        num_layers = self.model.num_layers

        all_effects = []
        samples_collected = 0

        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Get clean and patch inputs
            clean_inputs, patch_inputs = comparison_fn(batch)

            # Get clean logits
            clean_logits = self.get_clean_logits(
                clean_inputs["states"],
                clean_inputs["actions"],
                clean_inputs["returns_to_go"],
                clean_inputs["timesteps"],
                clean_inputs.get("attention_mask"),
            )

            K = clean_inputs["states"].shape[1]
            total_positions = 3 * K

            # Test each layer and position
            sample_effects = np.zeros((num_layers, total_positions))

            for layer in range(num_layers):
                for pos in range(total_positions):
                    patched_logits = self.patch_hidden_states(
                        clean_inputs, patch_inputs, layer, [pos]
                    )
                    sample_effects[layer, pos] = metric_fn(clean_logits, patched_logits)

            all_effects.append(sample_effects)
            samples_collected += clean_inputs["states"].shape[0]

        # Average effects
        mean_effects = np.mean(all_effects, axis=0)
        std_effects = np.std(all_effects, axis=0)

        return {
            "mean_effect": mean_effects,
            "std_effect": std_effects,
            "num_samples": samples_collected,
        }


def create_rtg_patching_comparison(
    high_rtg: float = 1.0,
    low_rtg: float = 0.0,
) -> Callable:
    """Create a comparison function for RTG patching.

    Returns a function that takes a batch and returns clean (high RTG)
    and patch (low RTG) inputs.
    """
    def comparison_fn(batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        clean = batch.copy()
        patch = batch.copy()

        # Modify RTG
        clean["returns_to_go"] = torch.full_like(batch["returns_to_go"], high_rtg)
        patch["returns_to_go"] = torch.full_like(batch["returns_to_go"], low_rtg)

        return clean, patch

    return comparison_fn


def action_probability_metric(action_idx: int) -> Callable:
    """Create a metric that measures probability of a specific action.

    Args:
        action_idx: Action index to measure.

    Returns:
        Metric function.
    """
    def metric_fn(logits: torch.Tensor) -> float:
        # Get probability of action at last position
        probs = F.softmax(logits[:, -1, :], dim=-1)
        return probs[:, action_idx].mean().item()

    return metric_fn


def action_change_metric() -> Callable:
    """Create a metric that measures how much the action distribution changed."""
    def metric_fn(clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
        clean_probs = F.softmax(clean_logits[:, -1, :], dim=-1)
        patched_probs = F.softmax(patched_logits[:, -1, :], dim=-1)

        # KL divergence
        kl = (clean_probs * (clean_probs.log() - patched_probs.log())).sum(dim=-1)
        return kl.mean().item()

    return metric_fn


def plot_patching_heatmap(
    effects: np.ndarray,
    title: str = "Activation Patching Effects",
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Plot patching effects as a heatmap.

    Args:
        effects: Array of shape (num_layers, num_positions).
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(effects, aspect="auto", cmap="RdBu_r", origin="lower")

    ax.set_xlabel("Position (RTG-State-Action interleaved)")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Effect")

    # Add modality separators
    K = effects.shape[1] // 3
    for i in range(1, 3):
        ax.axvline(x=i * K - 0.5, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig
