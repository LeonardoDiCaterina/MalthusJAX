"""Benchmarking runner and engine protocol definitions.

This lightweight module defines the :class:`BenchmarkRunner` class which
executes an evolutionary engine across multiple random seeds, collects the
results, and optionally writes artifact files.  A small ``Engine`` protocol
is specified here for engines to implement, and a simple deterministic
:class:`StubEngine` is provided for testing purposes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

# import chex
import chex
import jax.random as jr

from ..core.random import create_key, resolve_prng_impl
from .io import write_experiment_artifacts
from .results import ExperimentResult, RunResult

# tqdm is optional; import lazily for progress bars in loops
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class Engine(Protocol):
    """Simple protocol that engines must satisfy.

    Any engine plugged into :class:`BenchmarkRunner` needs to expose a
    ``run_once`` method accepting a JAX random key and returning a dictionary
    containing three standard entries: ``'history'`` for per-generation
    statistics, ``'summary'`` for final metrics, and an optional ``'timings'``
    dictionary capturing performance data.  This loose contract allows the
    benchmarking infrastructure to remain agnostic to concrete engine
    implementations.
    """

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Execute a single run of the engine and return its result dictionary.

        The caller is responsible for interpreting the resulting keys as
        described in the class docstring above.
        """
        ...


@dataclass
class BenchmarkRunner:
    """Runs benchmarking experiments across multiple seeds."""

    engine: Engine
    experiment_name: str = "benchmark_experiment"
    output_dir: Optional[Path] = None
    write_artifacts: bool = True
    prng_impl: Optional[str] = None
    trace_dir: Optional[Path] = None  # If set, capture JAX trace for seed[0]

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
        runs: List[RunResult] = []

        impl = resolve_prng_impl(self.prng_impl) if self.prng_impl else None

        # iterate through seeds with optional progress feedback
        iterable = enumerate(seeds)
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(seeds), desc="seeds")

        for i, seed in iterable:
            key = create_key(seed, impl=impl) if impl else jr.PRNGKey(seed)
            # Only trace the first seed
            trace_this = self.trace_dir if i == 0 else None
            run_result = self._run_single_seed(seed, key, timeout_seconds, trace_dir=trace_this)
            runs.append(run_result)

        # Create experiment result
        experiment = ExperimentResult(
            name=self.experiment_name,
            runs=runs,
            metadata={
                "seeds": list(seeds),
                "total_runs": len(runs),
                "successful_runs": len([r for r in runs if r.status == "success"]),
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

        The method measures wall time, optionally wraps the engine invocation
        in a JAX profiler trace, and normalizes the returned summary metrics to
        floats.  Any exceptions raised by the engine are caught and recorded in
        the resulting :class:`RunResult` with ``status="error"``.
        """
        import jax

        start_time = time.time()

        try:
            # Optionally wrap in JAX profiler trace
            if trace_dir is not None:
                trace_path = Path(trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                with jax.profiler.trace(str(trace_path)):
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

            duration = time.time() - start_time

            return RunResult(
                seed=seed,
                status="success",
                metrics=metrics,
                history=history,
                duration_seconds=duration,
                timings=timings,
            )

        except Exception as e:
            duration = time.time() - start_time
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
        # Use seed to make results deterministic but varied
        seed_int = int(key[0]) if hasattr(key, "__getitem__") else 42

        history = []
        current_fitness = self.base_fitness

        for gen in range(self.generations):
            # Simulate improvement (deterministic based on seed and gen)
            improvement = self.improvement_rate * (1 + (seed_int % 10) / 10.0)
            current_fitness -= improvement * (gen + 1) / self.generations

            history.append(
                {
                    "generation": gen,
                    "best_fitness": current_fitness,
                    "mean_fitness": current_fitness + 0.1,
                    "std_fitness": 0.05,
                }
            )

        summary = {
            "best_fitness": current_fitness,
            "final_generation": self.generations - 1,
            "total_evaluations": self.generations * 50,  # fake pop size
        }

        timings = {
            "initialization": 0.01,
            "evolution": 0.05 * self.generations,
        }

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
        }
