"""Evaluation and rollout utilities for Decision Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from midt.models.decision_transformer import DecisionTransformer
from midt.utils.seeding import get_device


class DTRollout:
    """Roll out a trained Decision Transformer in the environment."""

    def __init__(
        self,
        model: DecisionTransformer,
        env: gym.Env,
        context_length: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        rtg_scale: float = 1.0,
        device: Optional[str] = None,
    ):
        """Initialize the rollout handler.

        Args:
            model: Trained Decision Transformer.
            env: Gymnasium environment.
            context_length: Context length K.
            state_mean: Mean for state normalization.
            state_std: Std for state normalization.
            rtg_scale: Scaling factor for returns-to-go.
            device: Device to run on.
        """
        self.model = model
        self.env = env
        self.context_length = context_length
        self.state_mean = state_mean
        self.state_std = state_std
        self.rtg_scale = rtg_scale
        self.device = get_device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize a state observation."""
        return (state - self.state_mean) / self.state_std

    def _prepare_inputs(
        self,
        states: list[np.ndarray],
        actions: list[int],
        returns_to_go: list[float],
        timesteps: list[int],
    ) -> dict[str, torch.Tensor]:
        """Prepare model inputs from episode history.

        Pads or truncates to context_length.
        """
        K = self.context_length
        state_dim = states[0].shape[0]

        # Get the most recent K timesteps
        T = len(states)
        if T < K:
            # Pad at the beginning
            pad_len = K - T
            states_arr = np.zeros((K, state_dim), dtype=np.float32)
            actions_arr = np.zeros(K, dtype=np.int64)
            rtg_arr = np.zeros((K, 1), dtype=np.float32)
            timesteps_arr = np.zeros(K, dtype=np.int64)
            mask_arr = np.zeros(K, dtype=np.float32)

            for i in range(T):
                states_arr[pad_len + i] = self._normalize_state(states[i])
                actions_arr[pad_len + i] = actions[i]
                rtg_arr[pad_len + i, 0] = returns_to_go[i] / self.rtg_scale
                timesteps_arr[pad_len + i] = timesteps[i]
                mask_arr[pad_len + i] = 1.0
        else:
            # Use most recent K
            states_arr = np.array([self._normalize_state(s) for s in states[-K:]], dtype=np.float32)
            actions_arr = np.array(actions[-K:], dtype=np.int64)
            rtg_arr = np.array([[r / self.rtg_scale] for r in returns_to_go[-K:]], dtype=np.float32)
            timesteps_arr = np.array(timesteps[-K:], dtype=np.int64)
            mask_arr = np.ones(K, dtype=np.float32)

        # Convert to tensors and add batch dimension
        return {
            "states": torch.tensor(states_arr, dtype=torch.float32, device=self.device).unsqueeze(0),
            "actions": torch.tensor(actions_arr, dtype=torch.long, device=self.device).unsqueeze(0),
            "returns_to_go": torch.tensor(rtg_arr, dtype=torch.float32, device=self.device).unsqueeze(0),
            "timesteps": torch.tensor(timesteps_arr, dtype=torch.long, device=self.device).unsqueeze(0),
            "attention_mask": torch.tensor(mask_arr, dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    @torch.no_grad()
    def rollout(
        self,
        target_return: float,
        max_steps: int = 200,
        deterministic: bool = True,
        render: bool = False,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run a single episode with the Decision Transformer.

        Args:
            target_return: Target return to condition on.
            max_steps: Maximum steps per episode.
            deterministic: If True, take argmax actions. If False, sample.
            render: If True, render the environment.
            seed: Random seed for environment.

        Returns:
            Dictionary with episode data:
                - states: list of states
                - actions: list of actions
                - rewards: list of rewards
                - total_return: sum of rewards
                - length: episode length
                - info: final info dict
                - action_logits: list of action logit arrays
        """
        # Reset environment
        if seed is not None:
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()

        state = np.array(state).flatten().astype(np.float32)

        # Episode buffers
        states = [state]
        actions = [0]  # Placeholder for first action
        rewards = []
        returns_to_go = [target_return]
        timesteps = [0]
        action_logits_history = []

        done = False
        t = 0

        while not done and t < max_steps:
            # Prepare inputs
            inputs = self._prepare_inputs(states, actions, returns_to_go, timesteps)

            # Get action from model
            action_logits = self.model(
                states=inputs["states"],
                actions=inputs["actions"],
                returns_to_go=inputs["returns_to_go"],
                timesteps=inputs["timesteps"],
                attention_mask=inputs["attention_mask"],
            )

            # Get action for current timestep (last valid position)
            last_logits = action_logits[0, -1].cpu().numpy()
            action_logits_history.append(last_logits)

            if deterministic:
                action = int(np.argmax(last_logits))
            else:
                probs = np.exp(last_logits) / np.exp(last_logits).sum()
                action = np.random.choice(len(probs), p=probs)

            # Update current action (for next iteration's input)
            if len(actions) > len(states) - 1:
                actions[-1] = action
            else:
                actions.append(action)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if render:
                self.env.render()

            # Update buffers
            next_state = np.array(next_state).flatten().astype(np.float32)
            rewards.append(reward)

            if not done:
                states.append(next_state)
                actions.append(0)  # Placeholder
                returns_to_go.append(returns_to_go[-1] - reward)
                timesteps.append(t + 1)

            t += 1

        return {
            "states": states,
            "actions": actions[:-1] if len(actions) > len(rewards) else actions,  # Remove placeholder
            "rewards": rewards,
            "total_return": sum(rewards),
            "length": len(rewards),
            "info": info,
            "action_logits": action_logits_history,
            "target_return": target_return,
        }

    def evaluate(
        self,
        target_returns: list[float],
        num_episodes: int = 10,
        max_steps: int = 200,
        deterministic: bool = True,
        seeds: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Evaluate over multiple episodes and target returns.

        Args:
            target_returns: List of target returns to evaluate.
            num_episodes: Number of episodes per target return.
            max_steps: Maximum steps per episode.
            deterministic: If True, take argmax actions.
            seeds: Optional list of seeds (one per episode).

        Returns:
            DataFrame with evaluation results.
        """
        results = []

        for target_return in target_returns:
            for ep_idx in range(num_episodes):
                seed = seeds[ep_idx] if seeds is not None else None

                episode = self.rollout(
                    target_return=target_return,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    render=False,
                    seed=seed,
                )

                results.append({
                    "target_return": target_return,
                    "episode": ep_idx,
                    "actual_return": episode["total_return"],
                    "length": episode["length"],
                    "seed": seed,
                })

        return pd.DataFrame(results)

    def evaluate_summary(
        self,
        target_returns: list[float],
        num_episodes: int = 10,
        max_steps: int = 200,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Get summary statistics for evaluation.

        Args:
            target_returns: List of target returns.
            num_episodes: Episodes per target return.
            max_steps: Maximum steps per episode.
            deterministic: If True, take argmax actions.

        Returns:
            Dictionary with summary statistics.
        """
        df = self.evaluate(
            target_returns=target_returns,
            num_episodes=num_episodes,
            max_steps=max_steps,
            deterministic=deterministic,
        )

        summary = {}
        for tr in target_returns:
            subset = df[df["target_return"] == tr]
            key = f"target_{tr}"
            summary[f"{key}_mean_return"] = subset["actual_return"].mean()
            summary[f"{key}_std_return"] = subset["actual_return"].std()
            summary[f"{key}_mean_length"] = subset["length"].mean()

        summary["overall_mean_return"] = df["actual_return"].mean()
        summary["overall_std_return"] = df["actual_return"].std()
        summary["overall_mean_length"] = df["length"].mean()

        return summary


def create_rollout_from_checkpoint(
    checkpoint_path: str | Path,
    env: gym.Env,
    data_path: str | Path,
    rtg_scale: float = 1.0,
    device: Optional[str] = None,
) -> DTRollout:
    """Create a DTRollout from a checkpoint and data file.

    Args:
        checkpoint_path: Path to model checkpoint.
        env: Environment for rollout.
        data_path: Path to transition data (for normalization stats).
        rtg_scale: RTG scaling factor.
        device: Device to use.

    Returns:
        Configured DTRollout instance.
    """
    from midt.data.storage import TransitionStorage
    from midt.training.trainer import load_model_from_checkpoint

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load statistics
    storage = TransitionStorage.load(data_path)
    stats = storage.get_statistics()

    return DTRollout(
        model=model,
        env=env,
        context_length=model.context_length,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        rtg_scale=rtg_scale,
        device=device,
    )
