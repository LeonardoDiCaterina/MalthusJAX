#!/bin/bash
#SBATCH --job-name=mjx_publication_study
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from checkout root inside Slurm}"
test -f scripts/run_publication_validation.py
# Physical GPU 1 is expected; never overwrite Slurm's CUDA remapping.
if [[ "${SLURM_JOB_GPUS:-}" != "1" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Unexpected GPU allocation; expected physical GPU 1 and scheduler CUDA mapping.' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Use submitting environment; no fallback activation or package installation.
echo "Job=$SLURM_JOB_ID Python=$(command -v python) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec python -u scripts/run_publication_study.py "$@"
