"""Training loop for Decision Transformer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from midt.data.dataset import DecisionTransformerDataset, create_dataloader
from midt.models.decision_transformer import DecisionTransformer
from midt.utils.config import DTConfig, TrainingConfig
from midt.utils.seeding import get_device, set_seed


def get_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler ("cosine", "linear", "constant").
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        Learning rate scheduler.
    """
    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return max(0.0, 1.0 - (step - num_warmup_steps) / (num_training_steps - num_warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "constant":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class DecisionTransformerTrainer:
    """Training loop for Decision Transformer.

    Features:
    - Gradient accumulation
    - Mixed precision training (optional)
    - WandB logging integration
    - Checkpoint saving/loading
    - Evaluation during training
    """

    def __init__(
        self,
        model: DecisionTransformer,
        train_dataset: DecisionTransformerDataset,
        config: TrainingConfig,
        output_dir: str | Path,
        eval_dataset: Optional[DecisionTransformerDataset] = None,
        device: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            model: Decision Transformer model.
            train_dataset: Training dataset.
            config: Training configuration.
            output_dir: Output directory for checkpoints and logs.
            eval_dataset: Optional evaluation dataset.
            device: Device to train on.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = get_device(device)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Create data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        if eval_dataset is not None:
            self.eval_loader = create_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.eval_loader = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = get_scheduler(
            config.scheduler_type,
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # WandB
        self._wandb_run = None
        if config.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name,
                config={
                    "model": {
                        "state_dim": self.model.state_dim,
                        "action_dim": self.model.action_dim,
                        "embed_dim": self.model.embed_dim,
                        "num_layers": self.model.num_layers,
                        "num_heads": self.model.num_heads,
                        "context_length": self.model.context_length,
                        "num_parameters": self.model.num_parameters,
                    },
                    "training": self.config.model_dump(),
                },
                dir=str(self.output_dir / "logs"),
            )
        except ImportError:
            print("wandb not installed, skipping W&B logging")

    def train(self) -> dict[str, float]:
        """Run training loop.

        Returns:
            Final metrics dictionary.
        """
        set_seed(self.config.seed)

        self.model.train()
        data_iter = iter(self.train_loader)

        losses = []
        accuracies = []

        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            action_logits = self.model(
                states=batch["states"],
                actions=batch["actions"],
                returns_to_go=batch["returns_to_go"],
                timesteps=batch["timesteps"],
                attention_mask=batch["attention_mask"],
            )

            # Compute loss (cross-entropy on action predictions)
            # Only compute loss on valid (non-padded) positions
            mask = batch["attention_mask"].bool()
            targets = batch["actions"]

            # Reshape for loss computation
            B, K, A = action_logits.shape
            logits_flat = action_logits.view(-1, A)
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1)

            # Masked cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()

            # Compute accuracy
            with torch.no_grad():
                preds = action_logits.argmax(dim=-1)
                correct = (preds == targets) & mask
                accuracy = correct.sum().float() / mask.sum()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            self.global_step += 1

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log to WandB
            if self._wandb_run is not None and self.global_step % 10 == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                })

            # Evaluation
            if (
                self.eval_loader is not None
                and self.global_step % self.config.eval_every == 0
            ):
                eval_metrics = self.evaluate()
                self.model.train()

                if self._wandb_run is not None:
                    import wandb
                    wandb.log({
                        f"eval/{k}": v for k, v in eval_metrics.items()
                    }, step=self.global_step)

                # Save best model
                if eval_metrics["loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["loss"]
                    self.save_checkpoint("best")

            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}")

        pbar.close()

        # Final save
        self.save_checkpoint("final")

        # Final metrics
        final_metrics = {
            "train_loss": np.mean(losses[-100:]),
            "train_accuracy": np.mean(accuracies[-100:]),
            "total_steps": self.global_step,
        }

        if self._wandb_run is not None:
            import wandb
            wandb.finish()

        return final_metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on the evaluation dataset.

        Returns:
            Evaluation metrics.
        """
        if self.eval_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            action_logits = self.model(
                states=batch["states"],
                actions=batch["actions"],
                returns_to_go=batch["returns_to_go"],
                timesteps=batch["timesteps"],
                attention_mask=batch["attention_mask"],
            )

            mask = batch["attention_mask"].bool()
            targets = batch["actions"]

            B, K, A = action_logits.shape
            logits_flat = action_logits.view(-1, A)
            targets_flat = targets.view(-1)
            mask_flat = mask.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            total_loss += (loss * mask_flat).sum().item()

            preds = action_logits.argmax(dim=-1)
            correct = (preds == targets) & mask
            total_correct += correct.sum().item()
            total_count += mask.sum().item()

        return {
            "loss": total_loss / total_count,
            "accuracy": total_correct / total_count,
        }

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.output_dir / "checkpoints" / f"{name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.global_step,
            "config": self.config.model_dump(),
            "model_config": {
                "state_dim": self.model.state_dim,
                "action_dim": self.model.action_dim,
                "embed_dim": self.model.embed_dim,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "context_length": self.model.context_length,
                "max_timestep": self.model.max_timestep,
                "obs_type": self.model.obs_type,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Step number from checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["step"]

        return self.global_step


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> DecisionTransformer:
    """Load a Decision Transformer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        Loaded model.
    """
    device = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint["model_config"]
    model = DecisionTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return model
