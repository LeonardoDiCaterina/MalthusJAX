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
    """Protocol for evolutionary engines used by BenchmarkRunner."""

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Run one evolutionary experiment.
        Returns:
            dict with keys:
            - 'history': List[Dict[str, Any]] - per-generation stats
            - 'summary': Dict[str, Any] - final summary metrics
            - 'timings': Dict[str, float] - optional timing info
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
        """Run benchmark across multiple seeds.
        Args:
            seeds: List of random seeds to run
            timeout_seconds: Optional timeout per seed (not implemented yet)
        Returns:
            ExperimentResult with all runs and aggregated metrics
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
        """Run a single seed and collect results."""
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
        """Generate deterministic fake evolution data."""
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
