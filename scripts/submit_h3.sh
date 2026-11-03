#!/bin/bash
#SBATCH --job-name=mjx_h3_precision
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

# Verify environment and GPU connectivity
python -c "import jax; print('JAX version:', jax.__version__, '| Default backend:', jax.default_backend(), '| Devices:', jax.devices())"
nvidia-smi

echo ">>> Launching H3 Precision Benchmark Suite..."
python scripts/benchmark_runner.py --toml configs/h3_precision_lhs.toml

echo ">>> Benchmark completed at $(date). Launching Statistical Analyzer..."
python scripts/benchmark_analyzer.py --toml configs/h3_precision_lhs.toml --dir results/h3_precision_lhs

echo "=========================================================="
echo "Suite H3 Execution & Analysis complete at $(date)"
echo "=========================================================="
