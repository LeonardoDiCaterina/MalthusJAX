#!/bin/bash
#SBATCH --job-name=mjx_paper_collect
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the deployed checkout root}"
if [[ $# -lt 4 || "$1" != "--expect-revision" || "$3" != "--output" ]]; then
    echo 'Required: --expect-revision FULL_COMMIT --output CAMPAIGN_ROOT [retry options]' >&2
    exit 2
fi
expected_revision=$2
campaign_root=$4
shift 4
test "$(git rev-parse HEAD)" = "$expected_revision"
if [[ "${SLURM_JOB_GPUS:-}" != "1" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Expected Slurm physical GPU 1; no manual CUDA reassignment permitted.' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "${SLURM_ARRAY_TASK_ID:?array index required}" in
    0) phase=2 ;;
    1) phase=1 ;;
    2) phase=3 ;;
    *) echo 'Unknown phase array index' >&2; exit 2 ;;
esac
reuse=()
if [[ "$phase" == 2 ]]; then
    reuse=(--reuse-results results/study-v2-constant-unit0-e83506dd results/study-v2-cosine-unit1-e83506dd)
fi
echo "Job=$SLURM_JOB_ID Phase=$phase Python=$(command -v python) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec python -u scripts/run_publication_campaign.py --phase "$phase" --execute \
    --expect-revision "$expected_revision" --output "$campaign_root/phase$phase" "$@" "${reuse[@]}"
