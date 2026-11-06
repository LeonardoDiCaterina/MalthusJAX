#!/bin/bash
# =============================================================================
# Slurm Submission Script: DeepMind MuJoCo MJX Benchmark Showcase
#
# Usage:
#   sbatch scripts/submit_mujoco_showcase.sh [cartpole|ant|both]
#   ./scripts/submit_mujoco_showcase.sh cartpole
# =============================================================================

#SBATCH --job-name=mjx_showcase
#SBATCH --output=results/logs/mjx_showcase_%j.out
#SBATCH --error=results/logs/mjx_showcase_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

TASK="${1:-cartpole}"

echo "======================================================================"
echo "Starting DeepMind MuJoCo MJX Benchmark Showcase on Slurm"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Selected Task: ${TASK}"
echo "======================================================================"

# Ensure output directory exists
mkdir -p results/logs

# Enable headless MuJoCo rendering on cluster
# On GPU clusters, EGL delivers hardware-accelerated offscreen rendering
export MUJOCO_GL="egl"
export JAX_ENABLE_X64=False

# If nvidia/cuda devices are present, verify GPU visibility
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Detection:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Run the 3-way benchmark showcase with GIF rendering
python examples/showcase_mujoco_triad.py \
    --task "${TASK}" \
    --render-gifs \
    --generations 50 \
    --pop-size 32 \
    --seeds 42 43 44

echo "======================================================================"
echo "Showcase finished! Check results/showcase_${TASK}/ for GIFs & plots."
echo "======================================================================"
