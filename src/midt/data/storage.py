"""Storage utilities for trajectories using HDF5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import h5py
import numpy as np

from midt.data.trajectory import Trajectory


class TransitionStorage:
    """Efficient storage for trajectory data using HDF5.

    Supports streaming writes during data collection and efficient
    random access for training.
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "a",
        compression: str = "gzip",
        compression_opts: int = 4,
    ):
        """Initialize storage.

        Args:
            path: Path to HDF5 file.
            mode: File mode ('w' for write, 'r' for read, 'a' for append).
            compression: Compression algorithm.
            compression_opts: Compression level.
        """
        self.path = Path(path)
        self.mode = mode
        self.compression = compression
        self.compression_opts = compression_opts

        self._trajectories: list[Trajectory] = []
        self._file: Optional[h5py.File] = None
        self._num_episodes: int = 0

        if mode in ("r", "a") and self.path.exists():
            self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from existing file."""
        with h5py.File(self.path, "r") as f:
            if "metadata" in f:
                self._num_episodes = int(f["metadata/num_episodes"][()])

    def open(self) -> None:
        """Open the HDF5 file."""
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(self.path, self.mode)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "TransitionStorage":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to storage.

        The trajectory is buffered in memory until save() is called.

        Args:
            trajectory: Trajectory to add.
        """
        self._trajectories.append(trajectory)

    def save(self, env_config: Optional[dict[str, Any]] = None) -> None:
        """Flush buffered trajectories to disk.

        Args:
            env_config: Optional environment configuration to store.
        """
        if not self._trajectories:
            return

        self.open()
        f = self._file
        assert f is not None

        # Create or get groups
        if "trajectories" not in f:
            f.create_group("trajectories")
        if "metadata" not in f:
            f.create_group("metadata")

        traj_group = f["trajectories"]

        # Write each trajectory
        start_id = self._num_episodes
        for i, traj in enumerate(self._trajectories):
            episode_id = start_id + i
            ep_group = traj_group.create_group(f"episode_{episode_id}")

            arrays = traj.to_arrays()
            for key, arr in arrays.items():
                ep_group.create_dataset(
                    key,
                    data=arr,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            # Store metadata as JSON
            ep_group.attrs["metadata"] = json.dumps(traj.metadata)
            ep_group.attrs["total_reward"] = traj.total_reward
            ep_group.attrs["length"] = traj.length

        # Update metadata
        self._num_episodes += len(self._trajectories)
        if "num_episodes" in f["metadata"]:
            del f["metadata/num_episodes"]
        f["metadata"].create_dataset("num_episodes", data=self._num_episodes)

        if env_config is not None:
            if "env_config" in f["metadata"]:
                del f["metadata/env_config"]
            f["metadata"].attrs["env_config"] = json.dumps(env_config)

        # Clear buffer
        self._trajectories.clear()
        f.flush()

    def compute_and_save_statistics(self) -> dict[str, np.ndarray]:
        """Compute dataset statistics (mean, std) and save to file.

        Returns:
            Dictionary with state_mean, state_std, return_mean, return_std.
        """
        self.open()
        f = self._file
        assert f is not None

        # Collect all states and returns
        all_states = []
        all_returns = []

        for i in range(self._num_episodes):
            ep_key = f"trajectories/episode_{i}"
            if ep_key not in f:
                continue
            ep = f[ep_key]
            all_states.append(ep["states"][:])
            all_returns.append(ep["rtg"][0])  # Episode return (RTG at t=0)

        if not all_states:
            raise ValueError("No trajectories in storage")

        # Concatenate and compute statistics
        states = np.concatenate(all_states, axis=0)
        returns = np.array(all_returns)

        stats = {
            "state_mean": states.mean(axis=0).astype(np.float32),
            "state_std": states.std(axis=0).astype(np.float32) + 1e-6,
            "return_mean": np.array(returns.mean(), dtype=np.float32),
            "return_std": np.array(returns.std(), dtype=np.float32) + 1e-6,
        }

        # Save to file
        if "statistics" not in f:
            f.create_group("statistics")
        stat_group = f["statistics"]

        for key, arr in stats.items():
            if key in stat_group:
                del stat_group[key]
            stat_group.create_dataset(key, data=arr)

        f.flush()
        return stats

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Load dataset statistics from file.

        Returns:
            Dictionary with state_mean, state_std, return_mean, return_std.
        """
        with h5py.File(self.path, "r") as f:
            if "statistics" not in f:
                raise ValueError("Statistics not computed. Call compute_and_save_statistics first.")

            stat_group = f["statistics"]
            return {
                "state_mean": stat_group["state_mean"][:],
                "state_std": stat_group["state_std"][:],
                "return_mean": float(stat_group["return_mean"][()]),
                "return_std": float(stat_group["return_std"][()]),
            }

    @property
    def num_episodes(self) -> int:
        """Number of episodes in storage."""
        return self._num_episodes

    def get_trajectory(self, episode_id: int) -> Trajectory:
        """Load a single trajectory from storage.

        Args:
            episode_id: Episode index.

        Returns:
            Trajectory object.
        """
        with h5py.File(self.path, "r") as f:
            ep_key = f"trajectories/episode_{episode_id}"
            if ep_key not in f:
                raise KeyError(f"Episode {episode_id} not found")

            ep = f[ep_key]
            metadata = json.loads(ep.attrs.get("metadata", "{}"))

            return Trajectory.from_arrays(
                states=ep["states"][:],
                actions=ep["actions"][:],
                rewards=ep["rewards"][:],
                dones=ep["dones"][:],
                truncated=ep["truncated"][:],
                timesteps=ep["timesteps"][:],
                episode_id=episode_id,
                metadata=metadata,
            )

    def iter_trajectories(self) -> Iterator[Trajectory]:
        """Iterate over all trajectories in storage.

        Yields:
            Trajectory objects.
        """
        for i in range(self._num_episodes):
            yield self.get_trajectory(i)

    def get_all_arrays(self) -> dict[str, np.ndarray]:
        """Load all data as concatenated arrays.

        Returns:
            Dictionary with states, actions, rewards, dones, truncated,
            timesteps, rtg, and episode_starts arrays.
        """
        all_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "truncated": [],
            "timesteps": [],
            "rtg": [],
        }
        episode_starts = [0]
        total_len = 0

        with h5py.File(self.path, "r") as f:
            for i in range(self._num_episodes):
                ep_key = f"trajectories/episode_{i}"
                if ep_key not in f:
                    continue
                ep = f[ep_key]

                for key in all_data:
                    all_data[key].append(ep[key][:])

                total_len += len(ep["states"])
                episode_starts.append(total_len)

        # Concatenate
        result = {key: np.concatenate(vals, axis=0) for key, vals in all_data.items()}
        result["episode_starts"] = np.array(episode_starts[:-1], dtype=np.int64)
        result["episode_lengths"] = np.diff(episode_starts).astype(np.int64)

        return result

    def get_metadata(self) -> dict[str, Any]:
        """Get storage metadata.

        Returns:
            Dictionary with num_episodes, state_dim, action_dim, etc.
        """
        with h5py.File(self.path, "r") as f:
            metadata = {
                "num_episodes": int(f["metadata/num_episodes"][()]),
            }

            if "env_config" in f["metadata"].attrs:
                metadata["env_config"] = json.loads(f["metadata"].attrs["env_config"])

            # Infer dimensions from first episode
            if "trajectories/episode_0" in f:
                ep = f["trajectories/episode_0"]
                metadata["state_dim"] = ep["states"].shape[1]
                metadata["total_transitions"] = sum(
                    f[f"trajectories/episode_{i}/states"].shape[0]
                    for i in range(metadata["num_episodes"])
                )

            return metadata

    @classmethod
    def load(cls, path: str | Path) -> "TransitionStorage":
        """Load storage from file.

        Args:
            path: Path to HDF5 file.

        Returns:
            TransitionStorage instance.
        """
        return cls(path, mode="r")
