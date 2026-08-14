#!/bin/bash
# Core baseline script: runs pytest coverage, ruff, and mypy on Level 1 (core)
# Usage: ./scripts/run_core_baseline.sh

set -e

echo "=========================================="
echo "Level 1 (Core) Baseline Report"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Create output directory
if [ ! -d tmp ]; then
  mkdir tmp
fi

OUTPUT_FILE="tmp/core_baseline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $OUTPUT_FILE"
echo ""

{
  echo "=========================================="
  echo "Level 1 (Core) Baseline Report"
  echo "=========================================="
  echo "Date: $(date)"
  echo ""

  # 1. Pytest + Coverage
  echo "1. Running pytest with coverage..."
  echo "Command: pytest --maxfail=1 --disable-warnings -q --cov=src/malthusjax/composer/composer.py --cov=src/malthusjax/operators/mutation/real.py --cov=src/malthusjax/stats/comparator.py --cov-report=term-missing tests/composer/test_composer_edge_cases.py tests/operators/mutation/test_real_edge_cases.py tests/stats/test_comparator_edge_cases.py"
  echo ""
  pytest --maxfail=1 --disable-warnings -q --cov=src/malthusjax/composer/composer.py --cov=src/malthusjax/operators/mutation/real.py --cov=src/malthusjax/stats/comparator.py --cov-report=term-missing tests/composer/test_composer_edge_cases.py tests/operators/mutation/test_real_edge_cases.py tests/stats/test_comparator_edge_cases.py || true
  echo ""
  echo "✓ Pytest coverage report complete"
  echo ""

  # 2. Ruff check
  echo "2. Running ruff check..."
  echo "Command: ruff check src/malthusjax/core/"
  echo ""
  ruff check src/malthusjax/core/ || true
  echo ""
  echo "✓ Ruff check complete"
  echo ""

  # 3. Mypy type check
  echo "3. Running mypy type check..."
  echo "Command: mypy --strict src/malthusjax/core/ --ignore-missing-imports"
  echo ""
  mypy --strict src/malthusjax/core/ --ignore-missing-imports || true
  echo ""
  echo "✓ Mypy check complete"
  echo ""

  echo "=========================================="
  echo "Baseline report completed: $(date)"
  echo "=========================================="
} | tee "$OUTPUT_FILE"

echo ""
echo "✅ Baseline report saved to: $OUTPUT_FILE"
