#!/bin/bash
# =============================================================================
# Slurm Submission Script: Generate Empirical Paper Curves on GPU Cluster
#
# Usage:
#   sbatch scripts/submit_empirical_curves.sh --out results/NEW_RUN/curves.json
#   sbatch scripts/submit_empirical_curves.sh --smoke --out results/NEW_SMOKE/curves.json
#   bash scripts/submit_empirical_curves.sh --out results/NEW_RUN/curves.json
# =============================================================================

#SBATCH --job-name=mjx_paper_curves
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs results/logs

echo "=========================================================="
echo "Job Name:    ${SLURM_JOB_NAME:-interactive}"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Hostname:    $(hostname)"
echo "Started at:  $(date)"
echo "=========================================================="

# 1. Activate Conda Environment (Cluster setup)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate post_sub 2>/dev/null || conda activate malthusjax 2>/dev/null || true
elif [ -d "/home/20240485/.conda/envs/post_sub" ]; then
    export PATH="/home/20240485/.conda/envs/post_sub/bin:$PATH"
fi

# 2. Hardware and Environment Safety Guards
# Clamp CPU BLAS threads to avoid OpenBLAS thread pool exhaustion on high-core cluster nodes
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Prevent JAX greedy memory preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60

# Offscreen GPU accelerated rendering for MuJoCo
export MUJOCO_GL="egl"

# 3. Preserve the scheduler/user GPU assignment. Never choose a GPU by free memory.
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-scheduler-unset}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Checkout=$(pwd) Revision=$(git rev-parse HEAD) Python=$(command -v python)"

python -c "import jax; print('JAX Version:', jax.__version__, '| Default Backend:', jax.default_backend(), '| Devices:', jax.devices())"

# 4. Forward every argument unchanged, including quoted output paths.
# The generator requires a fresh explicit --out destination.
python -u examples/generate_empirical_paper_curves.py "$@"

echo "Empirical curve generation complete at $(date)."
