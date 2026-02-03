"""Embedding layers for Decision Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn


class StateEmbedding(nn.Module):
    """Embeds environment states into transformer dimension.

    For symbolic observations, uses a linear projection.
    For pixel observations, uses a CNN encoder.
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        obs_type: str = "symbolic",
    ):
        """Initialize the embedding.

        Args:
            state_dim: Dimension of input state.
            embed_dim: Output embedding dimension.
            obs_type: Type of observations ("symbolic" or "pixels").
        """
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.obs_type = obs_type

        if obs_type == "symbolic":
            self.embedding = nn.Linear(state_dim, embed_dim)
        elif obs_type == "pixels":
            # Simple CNN for pixel observations
            # Assumes input shape (H, W, C) with small grids
            self.embedding = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(embed_dim),
            )
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Embed states.

        Args:
            states: For symbolic: (batch, seq_len, state_dim)
                   For pixels: (batch, seq_len, H, W, C)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        if self.obs_type == "symbolic":
            return self.embedding(states)
        else:
            # Handle pixel observations
            batch_size, seq_len = states.shape[:2]
            # Reshape to (batch * seq_len, C, H, W)
            states = states.view(-1, *states.shape[2:])
            states = states.permute(0, 3, 1, 2)  # NHWC -> NCHW
            embeddings = self.embedding(states)
            return embeddings.view(batch_size, seq_len, self.embed_dim)


class ActionEmbedding(nn.Module):
    """Embeds discrete actions into transformer dimension."""

    def __init__(self, num_actions: int, embed_dim: int):
        """Initialize the embedding.

        Args:
            num_actions: Number of discrete actions.
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_actions, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed actions.

        Args:
            actions: Action indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.embedding(actions)


class ReturnEmbedding(nn.Module):
    """Embeds return-to-go scalars into transformer dimension."""

    def __init__(self, embed_dim: int):
        """Initialize the embedding.

        Args:
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """Embed returns-to-go.

        Args:
            returns: Return values of shape (batch, seq_len, 1)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.linear(returns)


class TimestepEmbedding(nn.Module):
    """Embeds timesteps into transformer dimension.

    Uses a learned embedding table for discrete timesteps.
    """

    def __init__(self, max_timestep: int, embed_dim: int):
        """Initialize the embedding.

        Args:
            max_timestep: Maximum timestep value.
            embed_dim: Output embedding dimension.
        """
        super().__init__()
        self.max_timestep = max_timestep
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(max_timestep, embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            timesteps: Timestep indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        # Clamp to valid range
        timesteps = timesteps.clamp(0, self.max_timestep - 1)
        return self.embedding(timesteps)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for sequence positions."""

    def __init__(self, max_length: int, embed_dim: int):
        """Initialize the embedding.

        Args:
            max_length: Maximum sequence length.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get positional embeddings.

        Args:
            seq_len: Sequence length.
            device: Device to place tensor on.

        Returns:
            Embeddings of shape (1, seq_len, embed_dim)
        """
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions).unsqueeze(0)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (non-learnable)."""

    def __init__(self, embed_dim: int, max_length: int = 5000):
        """Initialize the embedding.

        Args:
            embed_dim: Embedding dimension.
            max_length: Maximum sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Create positional encodings
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get positional embeddings.

        Args:
            seq_len: Sequence length.
            device: Device to place tensor on.

        Returns:
            Embeddings of shape (1, seq_len, embed_dim)
        """
        return self.pe[:, :seq_len].to(device)
