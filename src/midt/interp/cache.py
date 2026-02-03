"""Activation caching utilities for interpretability."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from midt.models.decision_transformer import DecisionTransformer


class ActivationCache:
    """Manages activation caching for interpretability experiments.

    Provides a clean interface for running forward passes with
    activation caching and retrieving specific activations.
    """

    def __init__(self, model: DecisionTransformer):
        """Initialize the cache.

        Args:
            model: Decision Transformer model.
        """
        self.model = model
        self._cache: Dict[str, torch.Tensor] = {}

    def run_with_cache(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass and cache all intermediate activations.

        Args:
            states: States tensor (batch, K, state_dim)
            actions: Actions tensor (batch, K)
            returns_to_go: RTG tensor (batch, K, 1)
            timesteps: Timesteps tensor (batch, K)
            attention_mask: Optional attention mask (batch, K)

        Returns:
            Tuple of (output logits, activation cache dict)
        """
        self.model.clear_cache()

        with torch.no_grad():
            output = self.model(
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                cache_activations=True,
            )

        self._cache = self.model.get_cache()
        self.model.clear_cache()

        return output, self._cache

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached activation by key.

        Args:
            key: Cache key (e.g., "block_0_attn_weights", "final_hidden")

        Returns:
            Cached tensor or None if not found.
        """
        return self._cache.get(key)

    def get_hidden(self, layer: int) -> Optional[torch.Tensor]:
        """Get hidden states from a specific layer.

        Args:
            layer: Layer index (0-indexed, or -1 for final).

        Returns:
            Hidden states tensor (batch, 3K, embed_dim)
        """
        if layer == -1:
            return self._cache.get("final_hidden")
        return self._cache.get(f"block_{layer}_hidden")

    def get_attention(self, layer: int) -> Optional[torch.Tensor]:
        """Get attention weights from a specific layer.

        Args:
            layer: Layer index.

        Returns:
            Attention weights (batch, num_heads, 3K, 3K)
        """
        num_layers = self.model.num_layers
        if layer < 0:
            layer = num_layers + layer
        return self._cache.get(f"block_{layer}_attn_weights")

    def get_mlp_output(self, layer: int) -> Optional[torch.Tensor]:
        """Get MLP output from a specific layer.

        Args:
            layer: Layer index.

        Returns:
            MLP output tensor.
        """
        num_layers = self.model.num_layers
        if layer < 0:
            layer = num_layers + layer
        return self._cache.get(f"block_{layer}_mlp_output")

    def get_attn_output(self, layer: int) -> Optional[torch.Tensor]:
        """Get attention output from a specific layer.

        Args:
            layer: Layer index.

        Returns:
            Attention output tensor.
        """
        num_layers = self.model.num_layers
        if layer < 0:
            layer = num_layers + layer
        return self._cache.get(f"block_{layer}_attn_output")

    def keys(self) -> list[str]:
        """List available cached activations.

        Returns:
            List of cache keys.
        """
        return list(self._cache.keys())

    def clear(self) -> None:
        """Clear the activation cache."""
        self._cache.clear()

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.model.num_layers

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self.model.embed_dim

    @property
    def context_length(self) -> int:
        """Context length K."""
        return self.model.context_length


class BatchedActivationCache:
    """Cache activations across multiple samples efficiently."""

    def __init__(self, model: DecisionTransformer, device: str = "cpu"):
        """Initialize the batched cache.

        Args:
            model: Decision Transformer model.
            device: Device to store cached tensors.
        """
        self.model = model
        self.device = device
        self.cache = ActivationCache(model)

        self._batched_cache: Dict[str, list[torch.Tensor]] = {}

    def add_sample(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add a sample to the batched cache.

        Args:
            states, actions, returns_to_go, timesteps: Model inputs.
            attention_mask: Optional attention mask.

        Returns:
            Model output for this sample.
        """
        output, sample_cache = self.cache.run_with_cache(
            states, actions, returns_to_go, timesteps, attention_mask
        )

        for key, value in sample_cache.items():
            if key not in self._batched_cache:
                self._batched_cache[key] = []
            self._batched_cache[key].append(value.to(self.device))

        return output

    def get_stacked(self, key: str) -> Optional[torch.Tensor]:
        """Get stacked activations for a key.

        Args:
            key: Cache key.

        Returns:
            Stacked tensor (num_samples, ...)
        """
        if key not in self._batched_cache:
            return None
        return torch.cat(self._batched_cache[key], dim=0)

    def get_all_hidden(self, layer: int) -> Optional[torch.Tensor]:
        """Get stacked hidden states from all samples.

        Args:
            layer: Layer index.

        Returns:
            Tensor (num_samples, 3K, embed_dim)
        """
        if layer == -1:
            return self.get_stacked("final_hidden")
        return self.get_stacked(f"block_{layer}_hidden")

    def get_all_attention(self, layer: int) -> Optional[torch.Tensor]:
        """Get stacked attention weights from all samples.

        Args:
            layer: Layer index.

        Returns:
            Tensor (num_samples, num_heads, 3K, 3K)
        """
        num_layers = self.model.num_layers
        if layer < 0:
            layer = num_layers + layer
        return self.get_stacked(f"block_{layer}_attn_weights")

    def num_samples(self) -> int:
        """Number of samples in the cache."""
        if not self._batched_cache:
            return 0
        first_key = next(iter(self._batched_cache))
        return len(self._batched_cache[first_key])

    def clear(self) -> None:
        """Clear all cached activations."""
        self._batched_cache.clear()
        self.cache.clear()
