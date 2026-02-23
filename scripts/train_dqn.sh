#!/bin/bash
#SBATCH --job-name=midt-dqn
#SBATCH --output=outputs/logs/dqn_%A_%a.out
#SBATCH --error=outputs/logs/dqn_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0          # 5 seeds by default; adjust or remove for a single run

# --- User configuration ---
PROJECT_DIR="$HOME/midt"      # Update to match cluster path
VENV="midt"                       # conda environment name
CONFIG="configs/dqn/gridworld.yaml"

# Derive seed and output dir from array task ID (ignored if not an array job)
SEED=$((42 + SLURM_ARRAY_TASK_ID))
OUTPUT_DIR="$HOME/ceph/midt_dqn/seed_${SEED}"

# --- Setup ---
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p outputs/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
source "$HOME/venvs/$VENV/bin/activate"

echo "Job:        $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:       $(hostname)"
echo "Seed:       $SEED"
echo "Output dir: $OUTPUT_DIR"
echo "Config:     $CONFIG"
echo ""

# --- Run ---
python -m midt.scripts.train_dqn \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
