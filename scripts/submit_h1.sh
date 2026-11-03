#!/bin/bash
#SBATCH --job-name=mjx_h1_parity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

mkdir -p logs

echo "=========================================================="
echo "Job Name:    $SLURM_JOB_NAME"
echo "Job ID:      $SLURM_JOB_ID"
echo "Hostname:    $(hostname)"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Started at:  $(date)"
echo "=========================================================="

# Activate Conda environment
source ~/.bashrc
conda activate post_sub

# Prevent JAX greedy preallocation / OOM on shared GPU cluster
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

# Verify environment and GPU connectivity
python -c "import jax; print('JAX version:', jax.__version__, '| Default backend:', jax.default_backend(), '| Devices:', jax.devices())"
nvidia-smi

echo ">>> Launching H1 Parity Benchmark Suite..."
python scripts/benchmark_runner.py --toml configs/h1_parity_lhs.toml

echo ">>> Benchmark completed at $(date). Launching Statistical Analyzer..."
python scripts/benchmark_analyzer.py --toml configs/h1_parity_lhs.toml --dir results/h1_parity_lhs

echo "=========================================================="
echo "Suite H1 Execution & Analysis complete at $(date)"
echo "=========================================================="
