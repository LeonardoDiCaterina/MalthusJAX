#!/bin/bash
#SBATCH --job-name=mjx_confirmatory
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/paper-confirmatory-v1/%x-%j.out
#SBATCH --error=logs/paper-confirmatory-v1/%x-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the deployed checkout root}"

part=1
stage="pilot"
config="configs/publication/part1_confirmatory.json"
expected_revision=""
output_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --part)
            part="$2"
            shift 2
            ;;
        --stage)
            stage="$2"
            shift 2
            ;;
        --config)
            config="$2"
            shift 2
            ;;
        --expect-revision)
            expected_revision="$2"
            shift 2
            ;;
        --output)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    echo "Error: --output directory is required" >&2
    exit 2
fi

if [[ -n "$expected_revision" ]]; then
    current_rev="$(git rev-parse HEAD)"
    if [[ "$current_rev" != "$expected_revision" ]]; then
        echo "Error: Revision mismatch! Current: $current_rev, Expected: $expected_revision" >&2
        exit 2
    fi
fi

mkdir -p "$(dirname "$output_dir")" "logs/paper-confirmatory-v1"

# Environment configuration
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=========================================================="
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Hostname:    $(hostname)"
echo "Started at:  $(date)"
echo "Part:        $part"
echo "Stage:       $stage"
echo "Config:      $config"
echo "Output:      $output_dir"
echo "Python:      $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=========================================================="

case "$part" in
    1)
        exec python -u scripts/run_publication_part1_confirmatory.py \
            --config "$config" \
            --stage "$stage" \
            --output "$output_dir"
        ;;
    2)
        exec python -u scripts/audit_publication_part2.py \
            --config "$config" \
            --output "$output_dir"
        ;;
    3)
        exec python -u scripts/run_publication_part3_tensor_qd.py \
            --config "$config" \
            --stage "$stage" \
            --output "$output_dir"
        ;;
    *)
        echo "Unknown part: $part" >&2
        exit 2
        ;;
esac
