#!/usr/bin/env bash
set -euo pipefail

# Run BBOB pop1024 TOMLs with optional smoke mode and nohup backgrounding.
#
# Default behavior:
#   - Launches itself in nohup/background
#   - Uses JAX_PLATFORMS=cpu
#   - Runs all examples/bbob_*_pop1024.toml (excluding *_smoke*.toml)
#   - Uses make run-toml-with-artifacts for each file
#
# Smoke mode:
#   --smoke_run
#   - Runs only examples/bbob_weierstrass_pop1024_smoke100.toml (if present)
#   - Falls back to examples/bbob_weierstrass_pop1024.toml
#
# Examples:
#   scripts/run_bbob_pop1024_nohup.sh
#   scripts/run_bbob_pop1024_nohup.sh --smoke_run
#   scripts/run_bbob_pop1024_nohup.sh --foreground --smoke_run

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_RUN=0
FOREGROUND=0
JAX_PLATFORM="cpu"
JAX_EFFECTIVE_PLATFORM=""
LOG_DIR="logs"
SMOKE_TOML_DEFAULT="examples/bbob_weierstrass_pop1024_smoke100.toml"
SMOKE_TOML_FALLBACK="examples/bbob_weierstrass_pop1024.toml"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --smoke_run           Run only smoke TOML (prefers bbob_weierstrass_pop1024_smoke100.toml)
  --foreground          Run in current shell (no nohup/background)
  --jax-platform <val>  JAX_PLATFORMS value (default: cpu)
  --log-dir <dir>       Log directory for nohup mode (default: logs)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke_run)
      SMOKE_RUN=1
      shift
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --jax-platform)
      JAX_PLATFORM="${2:-}"
      if [[ -z "$JAX_PLATFORM" ]]; then
        echo "Error: --jax-platform requires a value"
        exit 1
      fi
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:-}"
      if [[ -z "$LOG_DIR" ]]; then
        echo "Error: --log-dir requires a value"
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$FOREGROUND" -eq 0 ]]; then
  mkdir -p "$LOG_DIR"
  ts="$(date +%Y%m%d_%H%M%S)"
  mode="full"
  if [[ "$SMOKE_RUN" -eq 1 ]]; then
    mode="smoke"
  fi
  log_file="${LOG_DIR}/bbob_pop1024_${mode}_${ts}.log"

  cmd=(bash "$SCRIPT_PATH" --foreground --jax-platform "$JAX_PLATFORM" --log-dir "$LOG_DIR")
  if [[ "$SMOKE_RUN" -eq 1 ]]; then
    cmd+=(--smoke_run)
  fi

  nohup "${cmd[@]}" > "$log_file" 2>&1 &
  pid=$!

  echo "Started background run"
  echo "PID: $pid"
  echo "Log: $log_file"
  echo "Monitor: tail -f $log_file"
  exit 0
fi

# Resolve runtime platform string passed to JAX.
# "gpu" can trigger ROCm probing on some builds; prefer explicit CUDA.
if [[ "$JAX_PLATFORM" == "gpu" ]]; then
  JAX_EFFECTIVE_PLATFORM="cuda"
else
  JAX_EFFECTIVE_PLATFORM="$JAX_PLATFORM"
fi

# Safer defaults for GPU runs to avoid aggressive preallocation and instability.
# Users can still override these externally before launching.
if [[ "$JAX_PLATFORM" == "gpu" ]]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
fi

files=()
if [[ "$SMOKE_RUN" -eq 1 ]]; then
  if [[ -f "$SMOKE_TOML_DEFAULT" ]]; then
    files+=("$SMOKE_TOML_DEFAULT")
  elif [[ -f "$SMOKE_TOML_FALLBACK" ]]; then
    files+=("$SMOKE_TOML_FALLBACK")
  else
    echo "No smoke TOML found. Checked: $SMOKE_TOML_DEFAULT and $SMOKE_TOML_FALLBACK"
    exit 1
  fi
else
  while IFS= read -r f; do
    files+=("$f")
  done < <(find examples -maxdepth 1 -type f -name "bbob_*_pop1024.toml" ! -name "*smoke*" | sort)
fi

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No matching TOML files found."
  exit 1
fi

echo "Repo: $REPO_ROOT"
echo "Mode: $( [[ "$SMOKE_RUN" -eq 1 ]] && echo smoke || echo full )"
echo "Python executable: $(command -v python || true)"
echo "Python version: $(python -V 2>&1 || true)"
echo "JAX_PLATFORMS (requested): $JAX_PLATFORM"
echo "JAX_PLATFORMS (effective): $JAX_EFFECTIVE_PLATFORM"
if [[ "$JAX_PLATFORM" == "gpu" ]]; then
  echo "XLA_PYTHON_CLIENT_PREALLOCATE: ${XLA_PYTHON_CLIENT_PREALLOCATE}"
  echo "XLA_PYTHON_CLIENT_MEM_FRACTION: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
  echo "Running JAX GPU preflight..."
  if ! JAX_PLATFORMS="$JAX_EFFECTIVE_PLATFORM" python - <<'PY'
import os
import jax
import jaxlib
print('jax:', jax.__version__)
print('jaxlib:', jaxlib.__version__)
try:
    devices = jax.devices()
except Exception as e:
    raise SystemExit(
    f"JAX backend initialization failed for JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r}: {e!r}"
    )
platforms = sorted({getattr(d, 'platform', 'unknown') for d in devices})
print('platforms:', platforms)
print('devices:', devices)
if not any(getattr(d, 'platform', '') == 'gpu' for d in devices):
    raise SystemExit('No GPU devices visible to JAX (gpu mode requested).')
PY
  then
    echo "[fatal] JAX GPU preflight failed. Aborting before benchmark loop."
    exit 1
  fi
fi
echo "Total TOMLs: ${#files[@]}"

fail_count=0
failed_files=()

for idx in "${!files[@]}"; do
  file="${files[$idx]}"
  echo
  echo "[$((idx + 1))/${#files[@]}] Running $file"

  if JAX_PLATFORMS="$JAX_EFFECTIVE_PLATFORM" make run-toml-with-artifacts TOML="$file" PYTHON="$(command -v python)"; then
    echo "[ok] $file"
  else
    echo "[error] $file"
    fail_count=$((fail_count + 1))
    failed_files+=("$file")
  fi
done

echo
if [[ "$fail_count" -eq 0 ]]; then
  echo "Completed successfully: all ${#files[@]} TOMLs ran with artifacts."
  exit 0
fi

echo "Completed with failures: $fail_count / ${#files[@]}"
echo "Failed files:"
for f in "${failed_files[@]}"; do
  echo "  - $f"
done
exit 2
