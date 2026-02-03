"""GPT-style causal transformer components."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Returns attention weights for interpretability analysis.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize the attention layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional mask of shape (batch, seq_len) where
                1 indicates valid positions and 0 indicates padding.
            return_attn_weights: Whether to return attention weights.

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, embed_dim)
                - Attention weights of shape (batch, num_heads, seq_len, seq_len)
                  if return_attn_weights=True, else None
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*D)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply causal mask (prevent attending to future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply attention mask (for padding)
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, H, T, D_h)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attn_weights:
            return out, attn_weights
        return out, None


class MLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize the MLP.

        Args:
            embed_dim: Input/output dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
            activation: Activation function ("gelu" or "relu").
        """
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., embed_dim)

        Returns:
            Output tensor of shape (..., embed_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalTransformerBlock(nn.Module):
    """Single transformer block with causal self-attention.

    Designed with interpretability hooks for activation caching.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize the transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension ratio.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim,
            int(embed_dim * mlp_ratio),
            dropout,
            activation,
        )

        # Interpretability: cached activations
        self._attn_weights: Optional[torch.Tensor] = None
        self._attn_output: Optional[torch.Tensor] = None
        self._mlp_output: Optional[torch.Tensor] = None
        self._residual_pre_attn: Optional[torch.Tensor] = None
        self._residual_pre_mlp: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask.
            cache_activations: Whether to cache intermediate activations.

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Cache pre-attention residual
        if cache_activations:
            self._residual_pre_attn = x.detach().clone()

        # Attention with residual
        attn_out, attn_weights = self.attn(
            self.ln1(x),
            attention_mask,
            return_attn_weights=cache_activations,
        )

        if cache_activations:
            self._attn_weights = attn_weights
            self._attn_output = attn_out.detach().clone()

        x = x + attn_out

        # Cache pre-MLP residual
        if cache_activations:
            self._residual_pre_mlp = x.detach().clone()

        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))

        if cache_activations:
            self._mlp_output = mlp_out.detach().clone()

        x = x + mlp_out

        return x

    def get_cached_activations(self) -> dict[str, Optional[torch.Tensor]]:
        """Get cached activations from last forward pass.

        Returns:
            Dictionary with cached tensors.
        """
        return {
            "attn_weights": self._attn_weights,
            "attn_output": self._attn_output,
            "mlp_output": self._mlp_output,
            "residual_pre_attn": self._residual_pre_attn,
            "residual_pre_mlp": self._residual_pre_mlp,
        }

    def clear_cache(self) -> None:
        """Clear cached activations."""
        self._attn_weights = None
        self._attn_output = None
        self._mlp_output = None
        self._residual_pre_attn = None
        self._residual_pre_mlp = None


class GPT(nn.Module):
    """GPT-style causal transformer.

    This is a generic GPT that can be used as the backbone for
    Decision Transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """Initialize the GPT model.

        Args:
            embed_dim: Embedding dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length.
            mlp_ratio: MLP hidden dimension ratio.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, activation
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask of shape (batch, seq_len)
            cache_activations: Whether to cache intermediate activations.

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence too long: {seq_len} > {self.max_seq_len}"

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions)
        x = self.embed_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask, cache_activations)

        # Final layer norm
        x = self.ln_f(x)

        return x

    def get_attention_weights(self, layer: int = -1) -> Optional[torch.Tensor]:
        """Get attention weights from specified layer.

        Args:
            layer: Layer index (-1 for last layer).

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
            or None if not cached.
        """
        if layer < 0:
            layer = self.num_layers + layer
        return self.blocks[layer]._attn_weights

    def get_all_attention_weights(self) -> list[Optional[torch.Tensor]]:
        """Get attention weights from all layers.

        Returns:
            List of attention weight tensors.
        """
        return [block._attn_weights for block in self.blocks]

    def clear_cache(self) -> None:
        """Clear activation caches in all blocks."""
        for block in self.blocks:
            block.clear_cache()
