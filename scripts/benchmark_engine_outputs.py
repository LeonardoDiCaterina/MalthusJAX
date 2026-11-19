"""Measure engine-output costs from an installed wheel in a fresh process."""

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import jax

import malthusjax
from malthusjax.benchmarking.io import write_experiment_artifacts
from malthusjax.composer import Composer

OPTIONS = {
    "default": {},
    "no-history": {"collect_history": False},
    "no-solution": {"capture_solution": False},
    "no-csv": {"write_history_csv": False},
    "fingerprint": {"capture_population_fingerprint": True},
    "archive": {"export_archive": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=OPTIONS, default="default")
    parser.add_argument("--backend", choices=("malthusjax", "qdax"), default="malthusjax")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    if min(args.population, args.dimensions, args.generations, args.seeds) <= 0:
        parser.error("population, dimensions, generations, and seeds must be positive")
    if args.mode == "archive" and args.backend != "qdax":
        parser.error("archive mode requires the qdax backend")
    if args.mode == "fingerprint" and args.backend != "malthusjax":
        parser.error("fingerprint mode requires the malthusjax backend")

    with tempfile.TemporaryDirectory(prefix="malthusjax-output-benchmark-") as directory:
        root = Path(directory)
        started = time.perf_counter()
        result = (
            Composer.create_default()
            .compare(
                pipelines={"benchmark": {}},
                backend=args.backend,
                fitness="native_sphere",
                pop_size=args.population,
                genome_length=args.dimensions,
                generations=args.generations,
                seeds=tuple(range(1, args.seeds + 1)),
                output_dir=root,
                write_artifacts=False,
                **OPTIONS[args.mode],
            )
            .pipelines["benchmark"]
        )
        runner_wall = time.perf_counter() - started
        for run in result.runs:
            if run.status != "success":
                raise RuntimeError(f"Seed {run.seed}: {run.error}")
            if args.mode == "archive" and "archive" not in run.artifacts:
                raise RuntimeError(f"Seed {run.seed}: archive was not captured")
        started = time.perf_counter()
        history_csv = result.metadata.get("output_options", {}).get("write_history_csv", True)
        export_options = {} if history_csv else {"write_history_csv": False}
        write_experiment_artifacts(result, root / "exports", **export_options)
        export_wall = time.perf_counter() - started
        output_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    device = jax.devices()[0]
    device_stats = device.memory_stats() or {}
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "configuration": vars(args),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "malthusjax": malthusjax.__version__,
                "import_origin": str(Path(malthusjax.__file__).resolve()),
                "jax": jax.__version__,
                "device": device.device_kind,
                "runner_wall_seconds": runner_wall,
                "standard_export_seconds": export_wall,
                "output_bytes": output_bytes,
                "peak_host_bytes": host_peak if sys.platform == "darwin" else host_peak * 1024,
                "peak_device_bytes": device_stats.get("peak_bytes_in_use"),
                "runs": [
                    {
                        "seed": run.seed,
                        "timings": run.timings,
                        "duration_seconds": run.duration_seconds,
                        "best_fitness": run.metrics["best_fitness"],
                        "total_evaluations": run.metrics["total_evaluations"],
                    }
                    for run in result.runs
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
