"""Capture selected nonsecret environment metadata; optional JAX probe in a GPU allocation."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = (
    "jax",
    "jaxlib",
    "numpy",
    "flax",
    "chex",
    "scipy",
    "pandas",
    "statsmodels",
    "evosax",
    "qdax",
    "bbobax",
    "brax",
    "mujoco",
    "tensorneat",
    "malthusjax",
    "malthus-gp",
    "malthus-neat",
    "malthus-optimize",
)
ENV_KEYS = (
    "CONDA_DEFAULT_ENV",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_GPUS",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
)


def command(args, root):
    try:
        return subprocess.check_output(
            args, cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-jax", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    spec = importlib.util.find_spec("malthusjax")
    diff = command(["git", "diff", "HEAD", "--"], root)
    status = command(["git", "status", "--porcelain"], root)
    result = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Metadata observation, not an approved publication lock",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "checkout": str(root),
        "revision": command(["git", "rev-parse", "HEAD"], root),
        "dirty": bool(status) if status is not None else None,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff is not None
        else None,
        "diff_scope": "Tracked HEAD diff only; dirty also detects untracked files",
        "package_import_origin": spec.origin if spec else None,
        "packages": packages,
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            root,
        ),
        "jax_probe": "not requested",
    }
    if args.probe_jax:
        import jax

        result["jax_probe"] = {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "enable_x64": bool(jax.config.jax_enable_x64),
            "module_path": jax.__file__,
        }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
