"""Decision Transformer model implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from midt.models.embeddings import (
    ActionEmbedding,
    ReturnEmbedding,
    StateEmbedding,
    TimestepEmbedding,
)
from midt.models.gpt import CausalTransformerBlock


class DecisionTransformer(nn.Module):
    """Decision Transformer for offline RL.

    Architecture (per Chen et al. 2021):
    - Input: K timesteps of (return-to-go, state, action) = 3K tokens
    - Each modality has its own embedding layer
    - Timestep embeddings added to all tokens
    - GPT-style causal transformer
    - Action prediction from state token positions

    Interpretability features:
    - Built-in activation caching
    - Methods for attention pattern extraction
    - Support for activation patching interventions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        context_length: int = 20,
        max_timestep: int = 1000,
        dropout: float = 0.1,
        obs_type: str = "symbolic",
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
    ):
        """Initialize the Decision Transformer.

        Args:
            state_dim: Dimension of state observations.
            action_dim: Number of discrete actions.
            embed_dim: Transformer embedding dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads per layer.
            context_length: Number of timesteps in context (K).
            max_timestep: Maximum timestep for embedding.
            dropout: Dropout rate.
            obs_type: Observation type ("symbolic" or "pixels").
            mlp_ratio: MLP hidden dimension ratio.
            activation: Activation function.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.max_timestep = max_timestep
        self.obs_type = obs_type

        # Embedding layers for each modality
        self.state_embed = StateEmbedding(state_dim, embed_dim, obs_type)
        self.action_embed = ActionEmbedding(action_dim, embed_dim)
        self.return_embed = ReturnEmbedding(embed_dim)

        # Timestep embedding (shared across modalities within same timestep)
        self.timestep_embed = TimestepEmbedding(max_timestep, embed_dim)

        # Embedding layer norm
        self.embed_ln = nn.LayerNorm(embed_dim)

        # Positional embedding for sequence positions (3K positions)
        max_seq_len = 3 * context_length
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

        # Output layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Action prediction head (predicts from state tokens)
        self.action_head = nn.Linear(embed_dim, action_dim)

        # Interpretability: activation cache
        self._cache: Dict[str, torch.Tensor] = {}
        self._hooks: Dict[str, Any] = {}

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
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_activations: bool = False,
    ) -> torch.Tensor:
        """Forward pass through Decision Transformer.

        Token ordering: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        Action at position t is predicted from state at position t.

        Args:
            states: States of shape (batch, K, state_dim)
            actions: Actions of shape (batch, K)
            returns_to_go: RTG of shape (batch, K, 1)
            timesteps: Timesteps of shape (batch, K)
            attention_mask: Optional mask of shape (batch, K) where
                1 indicates valid positions.
            cache_activations: Whether to cache intermediate activations.

        Returns:
            Action logits of shape (batch, K, action_dim)
        """
        batch_size, seq_len = states.shape[:2]

        # Embed each modality
        state_embeddings = self.state_embed(states)  # (B, K, D)
        action_embeddings = self.action_embed(actions)  # (B, K, D)
        return_embeddings = self.return_embed(returns_to_go)  # (B, K, D)

        # Add timestep embeddings
        time_embeddings = self.timestep_embed(timesteps)  # (B, K, D)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        # Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        # Stack along new dimension then reshape
        stacked = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        )  # (B, K, 3, D)
        tokens = stacked.reshape(batch_size, 3 * seq_len, self.embed_dim)

        # Add positional embeddings
        positions = torch.arange(3 * seq_len, device=tokens.device)
        tokens = tokens + self.pos_embed(positions)

        # Layer norm and dropout after embedding
        tokens = self.embed_ln(tokens)
        tokens = self.embed_dropout(tokens)

        # Expand attention mask for interleaved tokens
        if attention_mask is not None:
            # (B, K) -> (B, 3K)
            expanded_mask = attention_mask.repeat_interleave(3, dim=1)
        else:
            expanded_mask = None

        # Cache input embeddings if requested
        if cache_activations:
            self._cache["input_embeddings"] = tokens.detach().clone()
            self._cache["state_embeddings"] = state_embeddings.detach().clone()
            self._cache["action_embeddings"] = action_embeddings.detach().clone()
            self._cache["return_embeddings"] = return_embeddings.detach().clone()

        # Transformer forward
        hidden = tokens
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, expanded_mask, cache_activations)
            if cache_activations:
                self._cache[f"block_{i}_hidden"] = hidden.detach().clone()
                self._cache[f"block_{i}_attn_weights"] = block._attn_weights
                self._cache[f"block_{i}_attn_output"] = block._attn_output
                self._cache[f"block_{i}_mlp_output"] = block._mlp_output

        # Final layer norm
        hidden = self.ln_f(hidden)

        if cache_activations:
            self._cache["final_hidden"] = hidden.detach().clone()

        # Extract state token positions (indices 1, 4, 7, ... in 0-indexed)
        # These are at positions 3*t + 1 for t = 0, 1, ..., K-1
        state_indices = torch.arange(1, 3 * seq_len, 3, device=hidden.device)
        state_hidden = hidden[:, state_indices, :]  # (B, K, D)

        # Predict actions from state representations
        action_logits = self.action_head(state_hidden)  # (B, K, action_dim)

        return action_logits

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = True,
    ) -> int:
        """Get action for the most recent timestep.

        Args:
            states: States of shape (1, K, state_dim) or (K, state_dim)
            actions: Actions of shape (1, K) or (K,)
            returns_to_go: RTG of shape (1, K, 1) or (K, 1)
            timesteps: Timesteps of shape (1, K) or (K,)
            deterministic: If True, return argmax. If False, sample.

        Returns:
            Action for the current timestep.
        """
        # Add batch dimension if needed
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)

        # Forward pass
        action_logits = self.forward(states, actions, returns_to_go, timesteps)

        # Get logits for the last position
        last_logits = action_logits[0, -1]  # (action_dim,)

        if deterministic:
            action = last_logits.argmax().item()
        else:
            probs = F.softmax(last_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        return action

    # ==================== Interpretability Methods ====================

    def get_attention_patterns(
        self,
        layer: int = -1,
    ) -> Optional[torch.Tensor]:
        """Get attention weights from specified layer.

        Args:
            layer: Layer index (-1 for last layer).

        Returns:
            Attention weights of shape (batch, num_heads, 3K, 3K)
        """
        if layer < 0:
            layer = self.num_layers + layer
        key = f"block_{layer}_attn_weights"
        return self._cache.get(key)

    def get_hidden_states(
        self,
        layer: int = -1,
    ) -> Optional[torch.Tensor]:
        """Get hidden states from specified layer.

        Args:
            layer: Layer index (-1 for last layer, -2 for second to last, etc.)
                   Use layer=num_layers for final hidden (after ln_f).

        Returns:
            Hidden states of shape (batch, 3K, embed_dim)
        """
        if layer == self.num_layers or layer == -1:
            return self._cache.get("final_hidden")
        if layer < 0:
            layer = self.num_layers + layer
        key = f"block_{layer}_hidden"
        return self._cache.get(key)

    def get_modality_hidden_states(
        self,
        layer: int = -1,
        modality: str = "state",
    ) -> Optional[torch.Tensor]:
        """Get hidden states for a specific modality.

        Args:
            layer: Layer index.
            modality: One of "return", "state", "action".

        Returns:
            Hidden states of shape (batch, K, embed_dim)
        """
        hidden = self.get_hidden_states(layer)
        if hidden is None:
            return None

        # Modality indices in interleaved sequence
        modality_offsets = {"return": 0, "state": 1, "action": 2}
        if modality not in modality_offsets:
            raise ValueError(f"Unknown modality: {modality}")

        offset = modality_offsets[modality]
        seq_len = hidden.shape[1] // 3
        indices = torch.arange(offset, hidden.shape[1], 3, device=hidden.device)

        return hidden[:, indices, :]

    def get_attention_by_modality(
        self,
        layer: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Decompose attention patterns by source/target modality.

        Returns:
            Dictionary with keys like "state_to_state", "state_to_return", etc.
        """
        attn = self.get_attention_patterns(layer)
        if attn is None:
            return {}

        batch_size, num_heads, total_len, _ = attn.shape
        K = total_len // 3

        modalities = ["return", "state", "action"]
        result = {}

        for src_mod in modalities:
            for tgt_mod in modalities:
                src_offset = modalities.index(src_mod)
                tgt_offset = modalities.index(tgt_mod)

                src_indices = torch.arange(src_offset, total_len, 3, device=attn.device)
                tgt_indices = torch.arange(tgt_offset, total_len, 3, device=attn.device)

                # Extract attention from src to tgt
                # attn has shape (B, H, src, tgt) where attention is FROM src TO tgt
                extracted = attn[:, :, src_indices, :][:, :, :, tgt_indices]
                result[f"{src_mod}_to_{tgt_mod}"] = extracted

        return result

    def clear_cache(self) -> None:
        """Clear activation cache."""
        self._cache.clear()
        for block in self.blocks:
            block.clear_cache()

    def get_cache(self) -> Dict[str, torch.Tensor]:
        """Get the full activation cache.

        Returns:
            Dictionary of cached tensors.
        """
        return self._cache.copy()

    def register_activation_hook(
        self,
        layer: int,
        component: str,
        hook_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> str:
        """Register a forward hook for activation patching.

        Args:
            layer: Layer index.
            component: Component name ("attn", "mlp", or "hidden").
            hook_fn: Function that takes activation tensor and returns modified tensor.

        Returns:
            Hook ID for removal.
        """
        hook_id = f"layer_{layer}_{component}"

        if component == "attn":
            target = self.blocks[layer].attn
        elif component == "mlp":
            target = self.blocks[layer].mlp
        else:
            raise ValueError(f"Unknown component: {component}")

        def hook(module, input, output):
            if isinstance(output, tuple):
                return (hook_fn(output[0]),) + output[1:]
            return hook_fn(output)

        handle = target.register_forward_hook(hook)
        self._hooks[hook_id] = handle

        return hook_id

    def remove_hook(self, hook_id: str) -> None:
        """Remove a registered hook.

        Args:
            hook_id: Hook ID from register_activation_hook.
        """
        if hook_id in self._hooks:
            self._hooks[hook_id].remove()
            del self._hooks[hook_id]

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DecisionTransformer(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  context_length={self.context_length},\n"
            f"  num_parameters={self.num_parameters:,}\n"
            f")"
        )
