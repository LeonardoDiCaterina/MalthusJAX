"""Benchmarking runner and engine protocol definitions.

This lightweight module defines the :class:`BenchmarkRunner` class which
executes an evolutionary engine across multiple random seeds, collects the
results, and optionally writes artifact files.  A small ``Engine`` protocol
is specified here for engines to implement, and a simple deterministic
:class:`StubEngine` is provided for testing purposes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chex
import jax

from ..core.logger import get_logger
from ..core.random import create_key, resolve_prng_impl
from ..core.timing_records import TimingPolicy, TimingRecord, environment_identity, json_safe
from .io import write_experiment_artifacts
from .results import ExperimentResult, RunResult, standard_json

logger = get_logger("benchmarking.runner")

# tqdm is optional; import lazily for progress bars in loops
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class BenchmarkRunner:
    """Runs benchmarking experiments across multiple seeds."""

    engine: Engine
    experiment_name: str = "benchmark_experiment"
    output_dir: Optional[Path] = None
    write_artifacts: bool = True
    prng_impl: Optional[str] = None
    trace_dir: Optional[Path] = None  # Legacy option; profiler capture is currently disabled.
    configuration: Optional[Dict[str, Any]] = None
    serialize_history: bool = True  # If False, drops generation history to save disk space

    timing_policy: TimingPolicy = field(default_factory=TimingPolicy)

    def run(
        self,
        seeds: Sequence[int],
        timeout_seconds: Optional[float] = None,
    ) -> ExperimentResult:
        """Drive the supplied engine over a sequence of seeds.

        Each seed is converted into a PRNG key (honouring any configured
        *prng_impl*), and ``_run_single_seed`` is invoked in turn.  Progress
        bars are shown if ``tqdm`` is installed.  Results from all seeds are
        collated into an :class:`ExperimentResult`; metadata such as the list of
        seeds and run counts are automatically populated.  When an output
        directory is provided the corresponding JSON/CSV artifacts are also
        written.
        """
        if timeout_seconds is not None:
            raise NotImplementedError(
                "Hard timeouts require a subprocess/job boundary; this runner cannot enforce them"
            )
        runs: List[RunResult] = []
        identity = environment_identity()
        configuration_snapshot = standard_json(copy.deepcopy(self.configuration))
        configuration_json = json.dumps(configuration_snapshot, sort_keys=True, allow_nan=False)
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds are ambiguous timing observations")

        impl = resolve_prng_impl(self.prng_impl) if self.prng_impl else None

        # iterate through seeds with optional progress feedback
        iterable = enumerate(seeds)
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(seeds), desc="seeds")

        for i, seed in iterable:
            logger.debug("Seed %d/%d start (seed=%s)", i + 1, len(seeds), seed)
            key = create_key(seed, impl=impl)
            # Only trace the first seed
            trace_this = self.trace_dir if i == 0 else None
            run_result = self._run_single_seed(seed, key, timeout_seconds, trace_dir=trace_this)
            run_result.timing_record = TimingRecord.from_producer(
                run_result.timings,
                run_result.timing_metadata,
                seed=seed,
                order=i,
                environment=identity,
                duration=run_result.duration_seconds,
            ).to_dict()
            record = run_result.timing_record
            record["target_phase"] = self.timing_policy.target_phase
            record["policy"] = self.timing_policy.to_dict()
            record["trial_results"] = []
            for trial_id in range(
                1, 1 + self.timing_policy.preconditioning + self.timing_policy.repetitions
            ):
                repeat = self._run_single_seed(seed, create_key(seed, impl=impl), timeout_seconds)
                repeated_record = TimingRecord.from_producer(
                    repeat.timings,
                    repeat.timing_metadata,
                    seed=seed,
                    order=i,
                    environment=identity,
                    duration=repeat.duration_seconds,
                ).to_dict()
                record["trial_results"].append(
                    {
                        "trial_id": trial_id,
                        "result": json_safe(repeat.to_dict()),
                        "timing_record": repeated_record,
                    }
                )
                trial = (
                    repeated_record.get("trials")
                    or [
                        {
                            "seconds": None,
                            "status": "unavailable",
                            "reason": repeated_record.get("reason"),
                        }
                    ]
                )[0].copy()
                trial.update(
                    trial_id=trial_id,
                    phase="preconditioning"
                    if trial_id <= self.timing_policy.preconditioning
                    else "prepared_execution_repeat",
                )
                record["trials"].append(trial)
                def objective(row: RunResult) -> dict[str, Any]:
                    return {
                        k: v
                        for k, v in row.metrics.items()
                        if k != "duration_seconds" and not k.startswith("time_")
                    }
                agrees = (
                    objective(repeat) == objective(run_result)
                    and repeat.history == run_result.history
                )
                record["trial_results"][-1]["objective_agreement"] = agrees
                if (
                    not agrees
                    or repeated_record["status"] != "valid"
                    or repeated_record["signature"] != record["signature"]
                ):
                    record.update(
                        status="unavailable",
                        reason="Repeated trial failed, seeded output differed, or executable signature changed",
                    )
            runs.append(run_result)
            logger.info(
                "Seed %d/%d done status=%s duration=%.2fs",
                i + 1,
                len(seeds),
                run_result.status,
                run_result.duration_seconds,
            )

        # Create experiment result
        experiment = ExperimentResult(
            name=self.experiment_name,
            runs=runs,
            metadata={
                "seeds": list(seeds),
                "total_runs": len(runs),
                "successful_runs": len([r for r in runs if r.status == "success"]),
                "serialize_history": self.serialize_history,
                "configuration": configuration_snapshot,
                "configuration_status": (
                    "partial_runtime_objects"
                    if "unserializable_type" in configuration_json
                    else "declared"
                )
                if self.configuration is not None
                else "unknown",
                "configuration_sha256": hashlib.sha256(configuration_json.encode()).hexdigest(),
                "timing_policy": self.timing_policy.to_dict(),
                "timing_environment": identity,
            },
        )

        # Write artifacts if requested
        if self.write_artifacts and self.output_dir:
            written_paths = write_experiment_artifacts(experiment, self.output_dir)
            # Update metadata with paths
            experiment.metadata["artifact_paths"] = {k: str(v) for k, v in written_paths.items()}

        return experiment

    def _run_single_seed(
        self,
        seed: int,
        key: chex.Array,
        timeout_seconds: Optional[float],
        trace_dir: Optional[Path] = None,
    ) -> RunResult:
        """Execute the engine for one seed and package the outcome.

        The method measures wall time and normalizes returned summary metrics.
        The legacy trace_dir option currently produces no profiler output.
        Any exceptions raised by the engine are caught and recorded in
        the resulting :class:`RunResult` with ``status="error"``.
        """

        start_time = time.perf_counter()

        try:
            if trace_dir is not None:
                # Tracing during compilation corrupts the XLA executable for subsequent
                # seeds that run without tracing, causing CUDA_ERROR_LAUNCH_FAILED.
                # Bypassing the profiler to preserve executable portability across seeds.
                #
                # trace_path = Path(trace_dir)
                # trace_path.mkdir(parents=True, exist_ok=True)
                # try:
                #     with jax.profiler.trace(str(trace_path)):
                #         engine_result = self.engine.run_once(key)
                # except RuntimeError:
                #     print("running without JAX trace due to profiler conflict", flush=True)
                #     engine_result = self.engine.run_once(key)
                engine_result = self.engine.run_once(key)
            else:
                engine_result = self.engine.run_once(key)

            # Extract results
            history = engine_result.get("history", [])
            summary = engine_result.get("summary", {})
            timings = engine_result.get("timings")

            # Convert summary to metrics (ensure float values)
            metrics = {}
            for k, v in summary.items():
                try:
                    metrics[k] = float(v)
                except (ValueError, TypeError):
                    # Skip non-numeric metrics or log warning
                    pass

            import math

            # Fallback: if engine disabled summary tracking for speed, pull from the final generation
            best_val = metrics.get("best_fitness")
            missing_best = best_val is None or (
                isinstance(best_val, (int, float))
                and (math.isnan(best_val) or math.isinf(best_val))
            )
            if missing_best and history:
                last = history[-1]
                if "best_fitness" in last:
                    metrics["best_fitness"] = float(last["best_fitness"])
                if "generation" in last:
                    metrics["final_generation"] = float(last["generation"])

            if "best_fitness" not in metrics or not math.isfinite(metrics["best_fitness"]):
                raise ValueError("Run returned no finite required best_fitness")
            expected = getattr(
                self.engine, "num_generations", getattr(self.engine, "generations", None)
            )
            if expected is None:
                candidate = getattr(
                    self.engine, "genetic_engine", getattr(self.engine, "mo_engine", self.engine)
                )
                expected = getattr(
                    getattr(candidate, "engine_params", None), "num_generations", None
                )
            if (
                expected is not None
                and "final_generation" in metrics
                and metrics["final_generation"] != expected
            ):
                raise ValueError("Reported completed horizon differs from requested horizon")
            duration = time.perf_counter() - start_time
            metrics["duration_seconds"] = duration

            if timings:
                for tk, tv in timings.items():
                    try:
                        metrics[f"time_{tk}"] = float(tv)
                    except (ValueError, TypeError):
                        pass

            # Discard massive history arrays if Lightweight Mode is enabled
            if not self.serialize_history:
                history = []

            return RunResult(
                seed=seed,
                status="success",
                metrics=metrics,
                history=history,
                duration_seconds=duration,
                timings=timings,
                timing_metadata=dict(engine_result.get("timing_metadata") or {}),
                metric_metadata=dict(engine_result.get("metric_metadata") or {}),
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            duration = time.perf_counter() - start_time
            return RunResult(
                seed=seed,
                status="error",
                metrics={},
                history=[],
                duration_seconds=duration,
                error=str(e),
            )


# Stub engine for testing
@dataclass
class StubEngine:
    """Deterministic stub engine for testing."""

    generations: int = 3
    base_fitness: float = 1.0
    improvement_rate: float = 0.1

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Produce a predictable synthetic result for testing.

        The returned history and summary values are driven deterministically by
        the supplied key so that test cases can assert against known outputs.
        This stub avoids any actual computation while still exercising the
        benchmarking machinery.
        """
        # Use seed to make results deterministic but varied across both legacy and typed PRNG keys
        try:
            if hasattr(jax.random, "key_data"):
                try:
                    data = jax.random.key_data(key)
                    seed_int = int(data[0]) if hasattr(data, "__getitem__") else int(data)
                except Exception:
                    seed_int = int(key[0]) if hasattr(key, "ndim") and key.ndim > 0 else 0
            elif hasattr(key, "ndim") and key.ndim > 0:
                seed_int = int(key[0])
            else:
                seed_int = 0
        except Exception:
            seed_int = 0

        history = []
        current_fitness = self.base_fitness

        for gen in range(self.generations):
            # Simulate improvement (deterministic based on seed and gen)
            improvement = self.improvement_rate * (1 + (seed_int % 10) / 10.0)
            current_fitness -= improvement * (gen + 1) / self.generations

            history.append(
                {
                    "generation": gen + 1,
                    "best_fitness": current_fitness,
                    "mean_fitness": current_fitness + 0.1,
                    "std_fitness": 0.05,
                }
            )

        summary = {
            "best_fitness": current_fitness,
            "final_generation": self.generations,
            "total_evaluations": self.generations * 50,  # fake pop size
        }

        timings = {
            "warmup": 0.01,
            "execution": 0.05 * self.generations,
            "total": 0.01 + 0.05 * self.generations,
        }

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
        }


# Deferred import to break circular dependency with composer
from malthusjax.composer.engine_protocol import Engine as Engine
