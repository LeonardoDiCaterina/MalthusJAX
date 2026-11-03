#!/bin/bash
# Interactive / tmux execution script for MalthusJAX publication benchmarks.
# Runs H2 and H3 sequentially with full memory guards and automated analysis.

set -e

mkdir -p logs

# Memory safety guards
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

echo "=========================================================="
echo "MalthusJAX Benchmark Driver (tmux / interactive mode)"
echo "Started at: $(date)"
echo "Host:       $(hostname)"
echo "=========================================================="

python -c "import jax; print('JAX version:', jax.__version__, '| Default backend:', jax.default_backend(), '| Devices:', jax.devices())"

echo ""
echo ">>> [1/2] Executing H2 Ablation Suite..."
python scripts/benchmark_runner.py --toml configs/h2_ablation_final_lhs.toml
echo ">>> Analyzing H2 Ablation Suite..."
python scripts/benchmark_analyzer.py --toml configs/h2_ablation_final_lhs.toml --dir results/h2_ablation_final

echo ""
echo ">>> [2/2] Executing H3 Precision Suite..."
python scripts/benchmark_runner.py --toml configs/h3_precision_lhs.toml
echo ">>> Analyzing H3 Precision Suite..."
python scripts/benchmark_analyzer.py --toml configs/h3_precision_lhs.toml --dir results/h3_precision_lhs

echo "=========================================================="
echo "All benchmarks completed successfully at: $(date)"
echo "Tables generated in:"
echo "  - results/h2_ablation_final/analysis/"
echo "  - results/h3_precision_lhs/analysis/"
echo "=========================================================="
