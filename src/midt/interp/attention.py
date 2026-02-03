"""Attention pattern analysis for Decision Transformer."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from midt.interp.cache import ActivationCache, BatchedActivationCache
from midt.models.decision_transformer import DecisionTransformer


class AttentionAnalyzer:
    """Tools for analyzing attention patterns in Decision Transformer.

    Provides methods for:
    - Decomposing attention by modality (RTG/state/action)
    - Analyzing temporal attention patterns
    - Identifying head specialization
    - Visualizing attention patterns
    """

    def __init__(self, model: DecisionTransformer, device: str = "cpu"):
        """Initialize the analyzer.

        Args:
            model: Decision Transformer model.
            device: Device to run analysis on.
        """
        self.model = model
        self.device = device
        self.cache = ActivationCache(model)

    def get_attention_weights(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """Get attention weights for a batch of inputs.

        Args:
            states, actions, returns_to_go, timesteps: Model inputs.
            attention_mask: Optional attention mask.
            layer: Layer to get attention from.

        Returns:
            Attention weights (batch, num_heads, 3K, 3K)
        """
        self.cache.run_with_cache(
            states, actions, returns_to_go, timesteps, attention_mask
        )
        return self.cache.get_attention(layer)

    def decompose_by_modality(
        self,
        attn_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decompose attention weights by source/target modality.

        In the DT sequence [R_0, s_0, a_0, R_1, s_1, a_1, ...],
        we separate attention between different token types.

        Args:
            attn_weights: Attention weights (batch, heads, 3K, 3K)

        Returns:
            Dictionary with keys like "state_to_state", "state_to_return", etc.
            Each value has shape (batch, heads, K, K)
        """
        batch_size, num_heads, total_len, _ = attn_weights.shape
        K = total_len // 3

        modalities = ["return", "state", "action"]
        result = {}

        for src_idx, src_mod in enumerate(modalities):
            for tgt_idx, tgt_mod in enumerate(modalities):
                # Source indices: positions where src_mod tokens appear
                src_positions = torch.arange(src_idx, total_len, 3, device=attn_weights.device)
                # Target indices: positions where tgt_mod tokens appear
                tgt_positions = torch.arange(tgt_idx, total_len, 3, device=attn_weights.device)

                # Extract attention from src to tgt
                extracted = attn_weights[:, :, src_positions, :][:, :, :, tgt_positions]
                result[f"{src_mod}_to_{tgt_mod}"] = extracted

        return result

    def compute_modality_attention_summary(
        self,
        attn_weights: torch.Tensor,
    ) -> pd.DataFrame:
        """Compute summary statistics for attention between modalities.

        Args:
            attn_weights: Attention weights (batch, heads, 3K, 3K)

        Returns:
            DataFrame with attention statistics.
        """
        decomposed = self.decompose_by_modality(attn_weights)

        rows = []
        for key, attn in decomposed.items():
            src, tgt = key.split("_to_")

            # Average attention (mean over batch, heads, positions)
            mean_attn = attn.mean().item()

            # Max attention
            max_attn = attn.max().item()

            # Per-head means
            head_means = attn.mean(dim=(0, 2, 3))  # (num_heads,)

            rows.append({
                "source": src,
                "target": tgt,
                "mean_attention": mean_attn,
                "max_attention": max_attn,
                "head_variance": head_means.var().item(),
            })

        return pd.DataFrame(rows)

    def compute_temporal_attention(
        self,
        attn_weights: torch.Tensor,
        modality: str = "state",
    ) -> torch.Tensor:
        """Analyze temporal attention patterns within a modality.

        Computes how much each position attends to past positions.

        Args:
            attn_weights: Attention weights (batch, heads, 3K, 3K)
            modality: Which modality to analyze ("return", "state", "action")

        Returns:
            Tensor (K, K) showing average attention from position i to j
            (lower triangular due to causal masking)
        """
        decomposed = self.decompose_by_modality(attn_weights)
        key = f"{modality}_to_{modality}"
        attn = decomposed[key]  # (batch, heads, K, K)

        # Average over batch and heads
        return attn.mean(dim=(0, 1))

    def analyze_head_specialization(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
        layer: int = -1,
    ) -> pd.DataFrame:
        """Analyze what each attention head specializes in.

        Args:
            dataloader: DataLoader for samples.
            num_batches: Number of batches to analyze.
            layer: Layer to analyze.

        Returns:
            DataFrame with per-head statistics.
        """
        num_heads = self.model.num_heads
        all_decomposed = {key: [] for key in [
            "return_to_return", "return_to_state", "return_to_action",
            "state_to_return", "state_to_state", "state_to_action",
            "action_to_return", "action_to_state", "action_to_action",
        ]}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                attn = self.get_attention_weights(
                    states=batch["states"],
                    actions=batch["actions"],
                    returns_to_go=batch["returns_to_go"],
                    timesteps=batch["timesteps"],
                    attention_mask=batch["attention_mask"],
                    layer=layer,
                )

                decomposed = self.decompose_by_modality(attn)
                for key, val in decomposed.items():
                    all_decomposed[key].append(val.cpu())

        # Concatenate across batches
        rows = []
        for head_idx in range(num_heads):
            head_stats = {"head": head_idx}

            for key in all_decomposed:
                if all_decomposed[key]:
                    stacked = torch.cat(all_decomposed[key], dim=0)  # (N, H, K, K)
                    head_attn = stacked[:, head_idx]  # (N, K, K)
                    head_stats[f"{key}_mean"] = head_attn.mean().item()
                    head_stats[f"{key}_max"] = head_attn.max().item()

            rows.append(head_stats)

        return pd.DataFrame(rows)

    def plot_attention_heatmap(
        self,
        attn_weights: torch.Tensor,
        layer: int = 0,
        head: int = 0,
        sample_idx: int = 0,
        figsize: tuple[int, int] = (10, 8),
        show_modality_separators: bool = True,
    ) -> plt.Figure:
        """Plot attention pattern as a heatmap.

        Args:
            attn_weights: Attention weights (batch, heads, 3K, 3K)
            layer: Layer index (for title).
            head: Head index.
            sample_idx: Which sample in batch to plot.
            figsize: Figure size.
            show_modality_separators: Whether to show lines separating modalities.

        Returns:
            Matplotlib figure.
        """
        attn = attn_weights[sample_idx, head].cpu().numpy()
        K = attn.shape[0] // 3

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn, cmap="Blues", aspect="auto")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Attention Weight")

        # Add modality separators
        if show_modality_separators:
            for i in range(1, 3):
                ax.axhline(y=i * K - 0.5, color="red", linestyle="--", alpha=0.5)
                ax.axvline(x=i * K - 0.5, color="red", linestyle="--", alpha=0.5)

        # Labels
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title(f"Layer {layer}, Head {head} Attention")

        # Add modality annotations
        for i, mod in enumerate(["R", "s", "a"]):
            mid = i * K + K // 2
            ax.text(-K * 0.1, mid, mod, ha="center", va="center", fontsize=12)
            ax.text(mid, -K * 0.1, mod, ha="center", va="center", fontsize=12)

        plt.tight_layout()
        return fig

    def plot_modality_attention_matrix(
        self,
        attn_weights: torch.Tensor,
        layer: int = 0,
        figsize: tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot average attention between modalities as a 3x3 matrix.

        Args:
            attn_weights: Attention weights.
            layer: Layer index (for title).
            figsize: Figure size.

        Returns:
            Matplotlib figure.
        """
        summary = self.compute_modality_attention_summary(attn_weights)

        # Create 3x3 matrix
        modalities = ["return", "state", "action"]
        matrix = np.zeros((3, 3))

        for _, row in summary.iterrows():
            src_idx = modalities.index(row["source"])
            tgt_idx = modalities.index(row["target"])
            matrix[src_idx, tgt_idx] = row["mean_attention"]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="Blues")

        # Add text annotations
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                       ha="center", va="center", fontsize=12)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["RTG", "State", "Action"])
        ax.set_yticklabels(["RTG", "State", "Action"])
        ax.set_xlabel("Attends To")
        ax.set_ylabel("Query From")
        ax.set_title(f"Layer {layer} Modality Attention")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
