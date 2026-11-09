#!/bin/bash
#SBATCH --job-name=mjx_pt5_timing
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
# Submit from the checkout root, with logs/ created before sbatch opens its files.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f scripts/validate_timing.py ]]; then
    echo 'Submit from the MalthusJAX checkout root.' >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo 'Required: --output NEW_DIRECTORY --expect-revision FULL_COMMIT' >&2
    exit 2
fi
# Use the submitting shell's activated environment. No silent environment fallback.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
# Preserve Slurm's CUDA_VISIBLE_DEVICES exactly; never search for an idle GPU.
echo "Job=${SLURM_JOB_ID:-local} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-scheduler-unset}"
echo "Checkout=$(pwd) Revision=$(git rev-parse HEAD) Python=$(command -v python)"
exec python -u scripts/validate_timing.py "$@"
