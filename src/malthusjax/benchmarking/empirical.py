"""Validate and preserve observations used by the empirical paper generator.

These helpers do not infer equivalence or normalize backend-native QD scores.
They keep observed values separate from explicitly derived best-so-far curves.
"""

from pathlib import Path
from typing import Any

import numpy as np


def _runs(comparison: Any, pipeline: str, generations: int) -> list[Any]:
    if generations <= 0:
        raise ValueError("generations must be positive when extracting curves")
    runs = list(comparison.pipelines[pipeline].runs)
    if not runs:
        raise ValueError(f"Pipeline '{pipeline}' has no runs")
    seeds = set()
    for run in runs:
        if run.seed in seeds:
            raise ValueError(f"Pipeline '{pipeline}' has duplicate seed {run.seed}")
        seeds.add(run.seed)
        if run.status != "success":
            raise ValueError(f"Pipeline '{pipeline}' failed on seed {run.seed}: {run.error}")
        if len(run.history) != generations:
            raise ValueError(
                f"Pipeline '{pipeline}' seed {run.seed}: expected {generations} history entries, "
                f"got {len(run.history)}"
            )
    return runs


def _curve(run: Any, metric: str) -> np.ndarray:
    values = []
    for generation, entry in enumerate(run.history, start=1):
        try:
            value = float(entry[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Seed {run.seed} generation {generation}: missing/invalid {metric}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Seed {run.seed} generation {generation}: nonfinite {metric}")
        values.append(value)
    return np.asarray(values)


def _execution_seconds(run: Any) -> float:
    from ..core.timing_records import select_timing

    return select_timing(run)[0]


def extract_control_runs(
    comparison: Any, pipeline: str, pop_size: int, generations: int, max_steps: int
) -> dict[str, Any]:
    """Extract signed, already-reported returns and explicitly derived running maxima."""
    if pop_size <= 0 or max_steps <= 0:
        raise ValueError("pop_size and max_steps must be positive")
    from ..core.timing_records import timing_values

    runs = _runs(comparison, pipeline, generations)
    timing_values(runs)
    records = []
    for run in runs:
        curve = _curve(run, "best_fitness")
        records.append(
            {
                "seed": int(run.seed),
                "best_fitness": curve.tolist(),
                "best_so_far": np.maximum.accumulate(curve).tolist(),
                "execution_seconds": _execution_seconds(run),
                "timing_metadata": dict(getattr(run, "timing_metadata", {}) or {}),
                "timing_record": dict(getattr(run, "timing_record", {}) or {}),
            }
        )
    observed = np.asarray([r["best_fitness"] for r in records])
    best_so_far = np.asarray([r["best_so_far"] for r in records])
    execution = np.asarray([r["execution_seconds"] for r in records])
    mean = np.mean(observed, axis=0)
    std = np.std(observed, axis=0)
    mean_exec = float(np.mean(execution))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "best_so_far_mean": np.mean(best_so_far, axis=0).tolist(),
        "best_so_far_std": np.std(best_so_far, axis=0).tolist(),
        "final_return": float(mean[-1]),
        "final_std": float(std[-1]),
        "exec_time_mean": mean_exec,
        "exec_time_std": float(np.std(execution)),
        "throughput": int(round(pop_size * generations * max_steps / mean_exec)),
        "per_seed": records,
    }


def extract_qd_runs(comparison: Any, pipeline: str, generations: int) -> dict[str, Any]:
    """Preserve backend-native signed QD observations without asserting comparability."""
    from ..core.timing_records import timing_values

    runs = _runs(comparison, pipeline, generations)
    timing_values(runs)
    records = []
    for run in runs:
        records.append(
            {
                "seed": int(run.seed),
                "best_fitness": _curve(run, "best_fitness").tolist(),
                "coverage": _curve(run, "coverage").tolist(),
                "qd_score": _curve(run, "qd_score").tolist(),
                "execution_seconds": _execution_seconds(run),
                "timing_metadata": dict(getattr(run, "timing_metadata", {}) or {}),
                "timing_record": dict(getattr(run, "timing_record", {}) or {}),
            }
        )
    return {
        "fit_arr": np.asarray([r["best_fitness"] for r in records]),
        "cov_arr": np.asarray([r["coverage"] for r in records]),
        "qd_arr": np.asarray([r["qd_score"] for r in records]),
        "ms_arr": np.asarray([r["execution_seconds"] * 1000 / generations for r in records]),
        "per_seed": records,
    }


def timing_comparison(native_ms: Any, reference_ms: Any) -> dict[str, Any]:
    """Return the ratio of measured medians, including ratios below one."""
    medians = []
    for values in (native_ms, reference_ms):
        samples = np.asarray(values, dtype=float)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("Timing samples must be nonempty one-dimensional arrays")
        if not np.isfinite(samples).all() or np.any(samples <= 0):
            raise ValueError("Timing samples must be finite and positive")
        medians.append(float(np.median(samples)))
    return {
        "ms_per_gen_malthusjax": medians[0],
        "ms_per_gen_qdax": medians[1],
        "speedup_ratio": medians[1] / medians[0],
        "timing_phase": "prepared_first_execution",
        "timing_estimator": "median_across_seeds",
        "timing_sample_counts": [len(native_ms), len(reference_ms)],
    }


def prepare_output_paths(output: Path) -> tuple[Path, Path]:
    """Require a fresh run directory so previous measurements cannot be overwritten."""
    output = Path(output)
    if output.exists() or (output.parent.exists() and any(output.parent.iterdir())):
        raise FileExistsError(f"Use an empty/new output directory: {output.parent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.parent / "tables", output.parent / "runs"
