from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..benchmarking import BenchmarkRunner, ExperimentResult, StubEngine


@dataclass
class Composer:
    """Compose and run evolutionary experiments with sensible defaults."""

    registry: Optional[Any] = None  # Will be Registry when implemented
    config: Dict[str, Any] = field(default_factory=dict)

    def quick_run(
        self,
        seeds: Sequence[int] = (1, 2, 3),
        experiment_name: str = "quick_experiment",
        output_dir: Optional[Path | str] = None,
        engine: Optional[Any] = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Quick-run an experiment with sensible defaults.
        This is the main product-first entry point for running experiments.
        Args:
            seeds: Random seeds to run (default: 3 seeds)
            experiment_name: Name for the experiment
            output_dir: Where to write results (default: ./results/{experiment_name})
            engine: Engine to use (default: StubEngine for now)
            **kwargs: Additional config passed to engine/runner
        Returns:
            ExperimentResult with all runs and aggregated metrics
        """
        # Set default output directory
        if output_dir is None:
            output_dir = Path("results") / experiment_name
        else:
            output_dir = Path(output_dir)

        # Use stub engine for now (will be replaced with real engine factory)
        if engine is None:
            # Extract engine config from kwargs
            generations = kwargs.get("generations", 10)
            base_fitness = kwargs.get("base_fitness", 1.0)
            improvement_rate = kwargs.get("improvement_rate", 0.1)

            engine = StubEngine(
                generations=generations,
                base_fitness=base_fitness,
                improvement_rate=improvement_rate,
            )

        # Create and run benchmark
        runner = BenchmarkRunner(
            engine=engine,
            experiment_name=experiment_name,
            output_dir=output_dir,
            write_artifacts=True,
        )

        return runner.run(seeds)

    @classmethod
    def create_default(cls) -> "Composer":
        """Create composer with default configuration."""
        return cls(config={"version": "0.1"})
