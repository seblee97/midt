# MIDT: Mechanistic Interpretability of Decision Transformers

Research codebase for studying mechanistic interpretability of Decision Transformers trained on gridworld environments.

## Overview

This project provides:

1. **DQN Training with Data Collection** - Train DQN agents using Stable-Baselines3 while capturing all transitions for offline RL
2. **Decision Transformer** - From-scratch PyTorch implementation with built-in interpretability hooks
3. **Mechanistic Interpretability Tools** - Attention analysis, linear probing, and activation patching

## Installation

Requires Python 3.10+.

```bash
cd /Users/sebastianlee/Dropbox/Documents/Research/Projects/midt
pip install -e .

# For development (includes testing, linting, jupyter)
pip install -e ".[dev]"
```

The gridworld environment should be installed separately:
```bash
pip install -e /Users/sebastianlee/Dropbox/Documents/Research/Projects/gridworld_env
```

## Quick Start

### 1. Train DQN and Collect Data

```bash
python -m midt.scripts.train_dqn --config configs/dqn/gridworld.yaml
```

This will:
- Train a DQN agent on the gridworld environment
- Save ALL transitions to `outputs/dqn/data/transitions.h5`
- Log training metrics to WandB (if enabled)

### 2. Train Decision Transformer

```bash
python -m midt.scripts.train_dt \
    --config configs/dt/gridworld.yaml \
    --model-config configs/dt/model.yaml
```

This will:
- Load the collected trajectory data
- Train a Decision Transformer on offline data
- Save checkpoints to `outputs/dt/checkpoints/`

### 3. Evaluate the Decision Transformer

```bash
python -m midt.scripts.evaluate \
    --checkpoint outputs/dt/checkpoints/best.pt \
    --data-path outputs/dqn/data/transitions.h5 \
    --env-layout /path/to/layout.txt \
    --target-returns 0.5 0.8 1.0 \
    --num-episodes 20
```

### 4. Run Interpretability Experiments

```bash
# Attention analysis
python -m midt.scripts.run_interp \
    --checkpoint outputs/dt/checkpoints/best.pt \
    --data-path outputs/dqn/data/transitions.h5 \
    --run-attention

# Linear probing
python -m midt.scripts.run_interp \
    --checkpoint outputs/dt/checkpoints/best.pt \
    --data-path outputs/dqn/data/transitions.h5 \
    --run-probing \
    --probe-target action
```

## Project Structure

```
midt/
├── configs/
│   ├── dqn/gridworld.yaml       # DQN training configuration
│   ├── dt/gridworld.yaml        # DT training configuration
│   ├── dt/model.yaml            # DT model architecture
│   └── interp/default.yaml      # Interpretability experiments
├── src/midt/
│   ├── data/
│   │   ├── trajectory.py        # Transition/Trajectory dataclasses
│   │   ├── storage.py           # HDF5 storage for trajectories
│   │   └── dataset.py           # PyTorch Dataset for DT training
│   ├── agents/
│   │   ├── callbacks.py         # SB3 callback for data collection
│   │   └── dqn_trainer.py       # DQN training wrapper
│   ├── models/
│   │   ├── embeddings.py        # State/Action/Return embeddings
│   │   ├── gpt.py               # Causal transformer blocks
│   │   └── decision_transformer.py  # Main DT model
│   ├── training/
│   │   └── trainer.py           # DT training loop
│   ├── evaluation/
│   │   └── rollout.py           # DT environment rollouts
│   ├── interp/
│   │   ├── cache.py             # Activation caching
│   │   ├── attention.py         # Attention pattern analysis
│   │   ├── probing.py           # Linear probing experiments
│   │   └── patching.py          # Activation patching
│   ├── utils/
│   │   ├── config.py            # Pydantic configuration classes
│   │   └── seeding.py           # Reproducibility utilities
│   └── scripts/
│       ├── train_dqn.py         # CLI: Train DQN
│       ├── train_dt.py          # CLI: Train Decision Transformer
│       ├── evaluate.py          # CLI: Evaluate DT
│       └── run_interp.py        # CLI: Run interpretability experiments
├── notebooks/                    # Jupyter notebooks for analysis
├── outputs/                      # Generated outputs (gitignored)
└── tests/                        # Unit tests
```

## Configuration

### DQN Configuration (`configs/dqn/gridworld.yaml`)

Key parameters:
- `env_layout_path`: Path to gridworld layout file
- `obs_mode`: Observation mode (`symbolic_minimal`, `symbolic`, `pixel`)
- `total_timesteps`: Total training steps
- `use_wandb`: Enable Weights & Biases logging

### DT Model Configuration (`configs/dt/model.yaml`)

Key parameters:
- `embed_dim`: Transformer embedding dimension (default: 128)
- `num_layers`: Number of transformer layers (default: 3)
- `num_heads`: Number of attention heads (default: 4)
- `action_dim`: Number of discrete actions (default: 4 for gridworld)

### DT Training Configuration (`configs/dt/gridworld.yaml`)

Key parameters:
- `context_length`: Number of timesteps in context window (default: 20)
- `batch_size`: Training batch size (default: 64)
- `max_steps`: Total training steps (default: 50000)
- `learning_rate`: Learning rate (default: 1e-4)

## Interpretability Methods

### Attention Analysis

Analyze attention patterns in the Decision Transformer:

```python
from midt.interp.attention import AttentionAnalyzer
from midt.training.trainer import load_model_from_checkpoint

model = load_model_from_checkpoint("outputs/dt/checkpoints/best.pt")
analyzer = AttentionAnalyzer(model)

# Get attention decomposed by modality (RTG/state/action)
attn_by_modality = analyzer.decompose_by_modality(attn_weights)

# Analyze head specialization
head_stats = analyzer.analyze_head_specialization(dataloader)
```

### Linear Probing

Probe for features in intermediate representations:

```python
from midt.interp.probing import ProbingExperiment

prober = ProbingExperiment(model)

# Probe for action prediction across all layers
results = prober.probe_all_layers(
    dataloader=dataloader,
    target="action",
    modality="state",
)
```

Supported probe targets:
- `action`: Predicted action
- `rtg`: Return-to-go (regression)
- `timestep`: Current timestep

### Activation Patching

Causal interventions to identify critical components:

```python
from midt.interp.patching import ActivationPatcher

patcher = ActivationPatcher(model)

# Patch hidden states from one input to another
patched_logits = patcher.patch_hidden_states(
    clean_inputs, patch_inputs, layer=2, positions=[1, 4, 7]
)

# Compute patching effect across all layers and positions
effects = patcher.compute_patching_effect(
    clean_inputs, patch_inputs, metric_fn
)
```

## Data Format

Trajectory data is stored in HDF5 format:

```
transitions.h5
├── trajectories/
│   ├── episode_0/
│   │   ├── states      # (T, state_dim) float32
│   │   ├── actions     # (T,) int32
│   │   ├── rewards     # (T,) float32
│   │   ├── rtg         # (T,) float32 - return-to-go
│   │   └── timesteps   # (T,) int32
│   └── ...
├── metadata/
│   ├── num_episodes
│   └── env_config
└── statistics/
    ├── state_mean
    ├── state_std
    ├── return_mean
    └── return_std
```

## References

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) (Chen et al., 2021)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021)
