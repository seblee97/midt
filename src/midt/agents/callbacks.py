"""Stable-Baselines3 callbacks for data collection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from midt.data.storage import TransitionStorage
from midt.data.trajectory import Trajectory, Transition


class TransitionCollectorCallback(BaseCallback):
    """Callback that captures ALL transitions during training.

    This callback intercepts environment steps and stores complete
    trajectories for offline RL training. It captures every transition,
    not just the ones sampled from the replay buffer.
    """

    def __init__(
        self,
        storage: TransitionStorage,
        save_freq: int = 10000,
        capture_info_keys: Optional[list[str]] = None,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            storage: TransitionStorage instance for saving data.
            save_freq: How often to flush data to disk (in timesteps).
            capture_info_keys: List of info dict keys to capture.
                If None, captures all keys.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.storage = storage
        self.save_freq = save_freq
        self.capture_info_keys = capture_info_keys

        # Episode tracking
        self._current_transitions: list[Transition] = []
        self._episode_id: int = 0
        self._last_obs: Optional[np.ndarray] = None
        self._episode_timestep: int = 0

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        # Store initial observation
        self._last_obs = self.training_env.reset()[0]
        if isinstance(self._last_obs, dict):
            # Handle dict observations by flattening or selecting
            self._last_obs = self._last_obs.get("observation", self._last_obs)
        self._last_obs = np.array(self._last_obs).flatten()

    def _on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            True to continue training, False to stop.
        """
        # Get current step data from locals
        # SB3 stores these in self.locals during training
        new_obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if new_obs is None or rewards is None:
            return True

        # Handle vectorized env (take first env)
        if len(rewards.shape) > 0:
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0] if infos else {}
            obs = np.array(new_obs[0]).flatten()
        else:
            reward = float(rewards)
            done = bool(dones)
            info = infos[0] if infos else {}
            obs = np.array(new_obs).flatten()

        # Get action from locals
        actions = self.locals.get("actions")
        if actions is not None:
            action = int(actions[0]) if len(actions.shape) > 0 else int(actions)
        else:
            action = 0

        # Filter info keys if specified
        if self.capture_info_keys is not None:
            info = {k: v for k, v in info.items() if k in self.capture_info_keys}

        # Check for truncation (SB3 puts this in info)
        truncated = info.get("TimeLimit.truncated", False)
        if "terminal_observation" in info:
            # Episode ended due to done, not truncation
            truncated = not done

        # Create transition
        transition = Transition(
            state=self._last_obs.copy(),
            action=action,
            reward=reward,
            next_state=obs.copy(),
            done=done and not truncated,
            truncated=truncated,
            info=info,
            timestep=self._episode_timestep,
            episode_id=self._episode_id,
        )
        self._current_transitions.append(transition)
        self._episode_timestep += 1

        # Update last observation
        self._last_obs = obs.copy()

        # Handle episode end
        if done:
            self._finish_episode()

        # Periodic save
        if self.n_calls % self.save_freq == 0:
            self.storage.save()
            if self.verbose > 0:
                print(f"Saved {self.storage.num_episodes} episodes to storage")

        return True

    def _finish_episode(self) -> None:
        """Finish current episode and save trajectory."""
        if not self._current_transitions:
            return

        # Create trajectory
        trajectory = Trajectory(
            transitions=self._current_transitions,
            episode_id=self._episode_id,
            metadata={
                "total_reward": sum(t.reward for t in self._current_transitions),
                "length": len(self._current_transitions),
            },
        )

        # Add to storage
        self.storage.add_trajectory(trajectory)

        # Reset for next episode
        self._current_transitions = []
        self._episode_id += 1
        self._episode_timestep = 0

        # Get new initial observation after reset
        # Note: SB3 auto-resets, so new_obs is already the reset obs
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            if len(new_obs.shape) > 1:
                self._last_obs = np.array(new_obs[0]).flatten()
            else:
                self._last_obs = np.array(new_obs).flatten()

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save any remaining transitions
        if self._current_transitions:
            self._finish_episode()

        # Final save
        self.storage.save()

        if self.verbose > 0:
            print(f"Training complete. Saved {self.storage.num_episodes} episodes.")


class EpisodeLoggerCallback(BaseCallback):
    """Callback for logging episode statistics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_reward: float = 0.0
        self._current_length: int = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self._current_reward += float(rewards[0])
            self._current_length += 1

            if dones is not None and dones[0]:
                self._episode_rewards.append(self._current_reward)
                self._episode_lengths.append(self._current_length)
                self._current_reward = 0.0
                self._current_length = 0

        return True

    def get_statistics(self) -> dict[str, float]:
        """Get episode statistics."""
        if not self._episode_rewards:
            return {}

        return {
            "mean_reward": np.mean(self._episode_rewards),
            "std_reward": np.std(self._episode_rewards),
            "mean_length": np.mean(self._episode_lengths),
            "num_episodes": len(self._episode_rewards),
        }
