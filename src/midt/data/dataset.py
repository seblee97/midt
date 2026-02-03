"""PyTorch Dataset for Decision Transformer training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from midt.data.storage import TransitionStorage


class DecisionTransformerDataset(Dataset):
    """PyTorch Dataset for Decision Transformer training.

    Converts trajectory data into sequences of (state, action, return-to-go)
    for training the Decision Transformer.
    """

    def __init__(
        self,
        storage: TransitionStorage | str | Path,
        context_length: int = 20,
        rtg_scale: float = 1.0,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        max_episodes: Optional[int] = None,
    ):
        """Initialize the dataset.

        Args:
            storage: TransitionStorage instance or path to HDF5 file.
            context_length: Number of timesteps in each sequence (K).
            rtg_scale: Scaling factor for returns-to-go.
            state_mean: Mean for state normalization. If None, computed from data.
            state_std: Std for state normalization. If None, computed from data.
            max_episodes: Maximum number of episodes to load. If None, load all.
        """
        if isinstance(storage, (str, Path)):
            storage = TransitionStorage.load(storage)

        self.storage = storage
        self.context_length = context_length
        self.rtg_scale = rtg_scale

        # Load all data into memory
        self._load_data(max_episodes)

        # Get or compute normalization statistics
        if state_mean is not None and state_std is not None:
            self.state_mean = state_mean
            self.state_std = state_std
        else:
            stats = storage.get_statistics()
            self.state_mean = stats["state_mean"]
            self.state_std = stats["state_std"]

        # Build index mapping: (episode_idx, start_timestep) for each sample
        self._build_index()

    def _load_data(self, max_episodes: Optional[int]) -> None:
        """Load trajectory data into memory."""
        data = self.storage.get_all_arrays()

        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.rtg = data["rtg"]
        self.timesteps = data["timesteps"]
        self.episode_starts = data["episode_starts"]
        self.episode_lengths = data["episode_lengths"]

        if max_episodes is not None:
            # Truncate to max_episodes
            max_idx = min(max_episodes, len(self.episode_starts))
            if max_idx < len(self.episode_starts):
                end_idx = self.episode_starts[max_idx]
                self.states = self.states[:end_idx]
                self.actions = self.actions[:end_idx]
                self.rewards = self.rewards[:end_idx]
                self.dones = self.dones[:end_idx]
                self.rtg = self.rtg[:end_idx]
                self.timesteps = self.timesteps[:end_idx]
                self.episode_starts = self.episode_starts[:max_idx]
                self.episode_lengths = self.episode_lengths[:max_idx]

    def _build_index(self) -> None:
        """Build index for sampling.

        Each sample is identified by (episode_idx, start_timestep).
        We allow sampling from any timestep in any episode.
        """
        self.sample_indices = []

        for ep_idx, (start, length) in enumerate(
            zip(self.episode_starts, self.episode_lengths)
        ):
            # Sample from each timestep in the episode
            for t in range(length):
                self.sample_indices.append((ep_idx, t))

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - states: (K, state_dim) normalized states
                - actions: (K,) discrete actions
                - returns_to_go: (K, 1) scaled returns-to-go
                - timesteps: (K,) timestep indices
                - attention_mask: (K,) mask for valid positions
        """
        ep_idx, start_t = self.sample_indices[idx]
        ep_start = self.episode_starts[ep_idx]
        ep_length = self.episode_lengths[ep_idx]

        K = self.context_length
        state_dim = self.states.shape[1]

        # Initialize arrays
        states = np.zeros((K, state_dim), dtype=np.float32)
        actions = np.zeros(K, dtype=np.int64)
        rtg = np.zeros((K, 1), dtype=np.float32)
        timesteps = np.zeros(K, dtype=np.int64)
        mask = np.zeros(K, dtype=np.float32)

        # Fill in data
        # We want the sequence ending at start_t, with history going back K steps
        for i in range(K):
            # Position in sequence (0 is oldest, K-1 is most recent)
            t_in_ep = start_t - (K - 1 - i)

            if 0 <= t_in_ep < ep_length:
                global_idx = ep_start + t_in_ep

                states[i] = self.states[global_idx]
                actions[i] = self.actions[global_idx]
                rtg[i, 0] = self.rtg[global_idx] / self.rtg_scale
                timesteps[i] = t_in_ep
                mask[i] = 1.0

        # Normalize states
        states = (states - self.state_mean) / self.state_std

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
        }

    @property
    def state_dim(self) -> int:
        """Dimension of state observations."""
        return self.states.shape[1]

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the dataset."""
        return len(self.episode_starts)

    def get_episode(self, ep_idx: int) -> dict[str, np.ndarray]:
        """Get a complete episode.

        Args:
            ep_idx: Episode index.

        Returns:
            Dictionary with states, actions, rewards, rtg, timesteps.
        """
        start = self.episode_starts[ep_idx]
        length = self.episode_lengths[ep_idx]
        end = start + length

        return {
            "states": self.states[start:end],
            "actions": self.actions[start:end],
            "rewards": self.rewards[start:end],
            "rtg": self.rtg[start:end],
            "timesteps": self.timesteps[start:end],
        }

    def split(
        self,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> tuple["DecisionTransformerDataset", "DecisionTransformerDataset"]:
        """Split dataset into train and eval sets by episode.

        Args:
            train_ratio: Fraction of episodes for training.
            seed: Random seed for splitting.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        rng = np.random.RandomState(seed)
        num_eps = self.num_episodes
        indices = rng.permutation(num_eps)

        split_idx = int(num_eps * train_ratio)
        train_eps = set(indices[:split_idx])
        eval_eps = set(indices[split_idx:])

        # Create new datasets with filtered indices
        train_dataset = _FilteredDataset(self, train_eps)
        eval_dataset = _FilteredDataset(self, eval_eps)

        return train_dataset, eval_dataset


class _FilteredDataset(Dataset):
    """Dataset filtered to specific episodes."""

    def __init__(
        self,
        parent: DecisionTransformerDataset,
        episode_indices: set[int],
    ):
        self.parent = parent
        self.filtered_indices = [
            i for i, (ep_idx, _) in enumerate(parent.sample_indices)
            if ep_idx in episode_indices
        ]
        self.state_mean = parent.state_mean
        self.state_std = parent.state_std
        self.state_dim = parent.state_dim

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.parent[self.filtered_indices[idx]]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        dataset: Dataset to load from.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for GPU transfer.

    Returns:
        DataLoader instance.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
