"""Stable-Baselines3 callbacks for data collection."""

from __future__ import annotations

from pathlib import Path
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


class VideoRecorderCallback(BaseCallback):
    """Callback to periodically record training and evaluation episodes as videos.

    Records both:
    - Training episodes (with exploration noise) to see actual learning behavior
    - Test episodes (deterministic) to see best-effort policy performance
    """

    def __init__(
        self,
        eval_env,
        video_dir: str | Path,
        record_freq: int = 10,
        log_to_wandb: bool = False,
        fps: int = 10,
        verbose: int = 0,
    ):
        """Initialize the video recorder callback.

        Args:
            eval_env: Environment for test rollouts (must have render_mode="rgb_array").
            video_dir: Directory to save videos.
            record_freq: Record videos every N episodes.
            log_to_wandb: Whether to log videos to Weights & Biases.
            fps: Frames per second for video playback.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = Path(video_dir)
        self.record_freq = record_freq
        self.log_to_wandb = log_to_wandb
        self.fps = fps

        # Episode tracking
        self._episode_count = 0
        self._video_count = 0

        # Training episode recording state
        self._recording_train = False
        self._train_frames: list = []

    def _on_training_start(self) -> None:
        """Create video directory and check if training env supports rendering."""
        self.video_dir.mkdir(parents=True, exist_ok=True)
        # Start recording first training episode
        self._recording_train = True
        self._train_frames = []
        self._capture_train_frame()

    def _on_step(self) -> bool:
        """Capture frames during training if recording."""
        dones = self.locals.get("dones")

        # Capture frame if recording training episode
        if self._recording_train:
            self._capture_train_frame()

        # Check for episode end
        if dones is not None and dones[0]:
            self._on_episode_end()

        return True

    def _capture_train_frame(self) -> None:
        """Capture a frame from the training environment."""
        try:
            frame = self.training_env.render()
            if frame is not None:
                self._train_frames.append(frame)
        except Exception:
            # Training env may not support rendering
            pass

    def _on_episode_end(self) -> None:
        """Handle end of training episode."""
        self._episode_count += 1

        # Save training video if we were recording
        if self._recording_train and self._train_frames:
            self._save_video(self._train_frames, "train")
            self._train_frames = []
            self._recording_train = False

            # Also record a test rollout
            self._record_test_video()

        # Check if we should start recording next episode
        if self._episode_count % self.record_freq == 0:
            self._recording_train = True
            self._train_frames = []
            # Capture initial frame on reset (will happen in next _on_step)

    def _record_test_video(self) -> None:
        """Record a deterministic test rollout."""
        try:
            import imageio
        except ImportError:
            if self.verbose > 0:
                print("imageio not installed, skipping video recording")
            return

        frames = []
        obs, _ = self.eval_env.reset()

        frame = self.eval_env.render()
        if frame is not None:
            frames.append(frame)

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = self.eval_env.step(action)

            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break

        if frames:
            self._save_video(frames, "test")

    def _save_video(self, frames: list, video_type: str) -> None:
        """Save frames as video file."""
        try:
            import imageio
        except ImportError:
            return

        if not frames:
            return

        video_path = self.video_dir / f"ep{self._episode_count:04d}_{video_type}.mp4"
        imageio.mimsave(str(video_path), frames, fps=self.fps)

        if self.verbose > 0:
            print(f"Saved {video_type} video to {video_path}")

        # Log to wandb if enabled
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        f"video/{video_type}": wandb.Video(str(video_path), fps=self.fps, format="mp4"),
                        "video_episode": self._episode_count,
                    })
            except ImportError:
                pass

        self._video_count += 1
