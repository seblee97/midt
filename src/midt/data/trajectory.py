"""Data structures for transitions and trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Transition:
    """A single environment transition.

    Attributes:
        state: Observation at time t.
        action: Action taken at time t.
        reward: Reward received after taking action.
        next_state: Observation at time t+1.
        done: Whether the episode terminated.
        truncated: Whether the episode was truncated.
        info: Additional information from the environment.
        timestep: Step within the episode (0-indexed).
        episode_id: Unique identifier for the episode.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    timestep: int = 0
    episode_id: int = 0

    @property
    def terminal(self) -> bool:
        """Whether this transition ends the episode."""
        return self.done or self.truncated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "truncated": self.truncated,
            "info": self.info,
            "timestep": self.timestep,
            "episode_id": self.episode_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Trajectory:
    """A complete episode trajectory.

    Attributes:
        transitions: List of transitions in the episode.
        episode_id: Unique identifier for the episode.
        metadata: Additional episode-level metadata.
    """

    transitions: list[Transition]
    episode_id: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of transitions in the trajectory."""
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Transition:
        """Get transition by index."""
        return self.transitions[idx]

    @property
    def total_reward(self) -> float:
        """Sum of rewards in the trajectory."""
        return sum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        """Length of the trajectory."""
        return len(self.transitions)

    @property
    def states(self) -> np.ndarray:
        """Array of states (T, state_dim)."""
        return np.stack([t.state for t in self.transitions])

    @property
    def actions(self) -> np.ndarray:
        """Array of actions (T,)."""
        return np.array([t.action for t in self.transitions])

    @property
    def rewards(self) -> np.ndarray:
        """Array of rewards (T,)."""
        return np.array([t.reward for t in self.transitions])

    @property
    def dones(self) -> np.ndarray:
        """Array of done flags (T,)."""
        return np.array([t.done for t in self.transitions])

    @property
    def truncated_flags(self) -> np.ndarray:
        """Array of truncated flags (T,)."""
        return np.array([t.truncated for t in self.transitions])

    @property
    def timesteps(self) -> np.ndarray:
        """Array of timesteps (T,)."""
        return np.array([t.timestep for t in self.transitions])

    def compute_returns_to_go(self, gamma: float = 1.0) -> np.ndarray:
        """Compute return-to-go for each timestep.

        Args:
            gamma: Discount factor. Default 1.0 (undiscounted).

        Returns:
            Array of shape (T,) with return-to-go at each timestep.
        """
        rewards = self.rewards
        T = len(rewards)
        rtg = np.zeros(T, dtype=np.float32)

        # Compute backwards: rtg[t] = r[t] + gamma * rtg[t+1]
        rtg[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            rtg[t] = rewards[t] + gamma * rtg[t + 1]

        return rtg

    def compute_returns(self, gamma: float = 1.0) -> np.ndarray:
        """Compute discounted returns for each timestep.

        This is the same as compute_returns_to_go but named differently
        for clarity in some contexts.

        Args:
            gamma: Discount factor.

        Returns:
            Array of shape (T,) with discounted return from each timestep.
        """
        return self.compute_returns_to_go(gamma)

    def get_segment(
        self,
        start_idx: int,
        length: int,
        pad_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a segment of the trajectory with padding if needed.

        Args:
            start_idx: Starting index in the trajectory.
            length: Length of the segment to extract.
            pad_value: Value to use for padding.

        Returns:
            Tuple of (states, actions, rewards, rtg, timesteps) arrays,
            each of shape (length, ...) with appropriate padding.
        """
        T = len(self.transitions)
        states = []
        actions = []
        rewards = []
        timesteps = []

        # Get state dimension from first transition
        state_dim = self.transitions[0].state.shape

        for i in range(length):
            idx = start_idx + i
            if idx < 0 or idx >= T:
                # Pad with zeros/pad_value
                states.append(np.zeros(state_dim, dtype=np.float32))
                actions.append(0)
                rewards.append(pad_value)
                timesteps.append(0)
            else:
                t = self.transitions[idx]
                states.append(t.state.astype(np.float32))
                actions.append(t.action)
                rewards.append(t.reward)
                timesteps.append(t.timestep)

        states = np.stack(states)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        timesteps = np.array(timesteps, dtype=np.int64)

        # Compute RTG for the full trajectory, then extract segment
        full_rtg = self.compute_returns_to_go()
        rtg = np.zeros(length, dtype=np.float32)
        for i in range(length):
            idx = start_idx + i
            if 0 <= idx < T:
                rtg[i] = full_rtg[idx]

        return states, actions, rewards, rtg, timesteps

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert trajectory to numpy arrays for storage.

        Returns:
            Dictionary with arrays for states, actions, rewards, dones,
            truncated, timesteps, and rtg.
        """
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "truncated": self.truncated_flags,
            "timesteps": self.timesteps,
            "rtg": self.compute_returns_to_go(),
        }

    @classmethod
    def from_arrays(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        truncated: np.ndarray,
        timesteps: np.ndarray,
        episode_id: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Trajectory:
        """Create trajectory from numpy arrays.

        Args:
            states: Array of states (T, state_dim).
            actions: Array of actions (T,).
            rewards: Array of rewards (T,).
            dones: Array of done flags (T,).
            truncated: Array of truncated flags (T,).
            timesteps: Array of timesteps (T,).
            episode_id: Episode identifier.
            metadata: Optional episode metadata.

        Returns:
            Trajectory object.
        """
        T = len(states)
        transitions = []

        for t in range(T):
            # For next_state, use next state or current state if terminal
            if t < T - 1:
                next_state = states[t + 1]
            else:
                next_state = states[t]

            transition = Transition(
                state=states[t],
                action=int(actions[t]),
                reward=float(rewards[t]),
                next_state=next_state,
                done=bool(dones[t]),
                truncated=bool(truncated[t]),
                timestep=int(timesteps[t]),
                episode_id=episode_id,
            )
            transitions.append(transition)

        return cls(
            transitions=transitions,
            episode_id=episode_id,
            metadata=metadata or {},
        )


@dataclass
class TrajectoryBatch:
    """A batch of trajectory segments for training.

    All arrays have shape (batch_size, seq_len, ...).
    """

    states: np.ndarray  # (B, K, state_dim)
    actions: np.ndarray  # (B, K)
    returns_to_go: np.ndarray  # (B, K)
    timesteps: np.ndarray  # (B, K)
    attention_mask: np.ndarray  # (B, K)

    def to_torch(self, device: str = "cpu") -> dict[str, "torch.Tensor"]:
        """Convert to PyTorch tensors.

        Args:
            device: Device to place tensors on.

        Returns:
            Dictionary of tensors.
        """
        import torch

        return {
            "states": torch.tensor(self.states, dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "returns_to_go": torch.tensor(
                self.returns_to_go, dtype=torch.float32, device=device
            ).unsqueeze(-1),
            "timesteps": torch.tensor(self.timesteps, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                self.attention_mask, dtype=torch.float32, device=device
            ),
        }
