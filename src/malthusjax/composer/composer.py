from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..benchmarking import BenchmarkRunner, ExperimentResult, StubEngine
from .catalog import OperatorCatalog
from .engine_factory import build_engine_from_catalog


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
        fitness: Optional[str] = None,
        selection: Optional[str] = None,
        crossover: Optional[str] = None,
        mutation: Optional[str] = None,
        genome_type: str = "real",
        pop_size: int = 50,
        generations: int = 100,
        genome_length: int = 10,
        bounds: tuple = (-5.0, 5.0),
        elitism: int = 2,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Quick-run an experiment with sensible defaults.
        This is the main product-first entry point for running experiments.
        Can use either real evolutionary operators via string specs OR
        fall back to StubEngine for testing/demos.
        Args:
            seeds: Random seeds to run (default: 3 seeds)
            experiment_name: Name for the experiment
            output_dir: Where to write results (default: ./results/{experiment_name})
            engine: Engine to use (overrides operator specs if provided)
            # Real operator specifications
            fitness: Fitness evaluator spec, e.g. "sphere:dim=10"
            selection: Selection operator spec,
                e.g. "tournament:num_selections=25,tournament_size=3"
            crossover: Crossover operator spec, e.g. "blend:alpha=0.5"
            mutation: Mutation operator spec, e.g. "gaussian:mutation_rate=0.1"
            # Engine configuration
            genome_type: "real" or "binary" (default: "real")
            pop_size: Population size (default: 50)
            generations: Number of generations (default: 100)
            genome_length: Length of genome vectors (default: 10)
            bounds: Bounds for real genomes as (min, max) (default: (-5.0, 5.0))
            elitism: Number of elite individuals preserved (default: 2)
            **kwargs: Additional config passed to engine/runner
        Returns:
            ExperimentResult with all runs and aggregated metrics
        Examples:
            # Quick demo with StubEngine (existing behavior)
            result = composer.quick_run()
            # Real evolutionary computation
            result = composer.quick_run(
                fitness="sphere:dim=5",
                selection="tournament:num_selections=25,tournament_size=3",
                crossover="blend:alpha=0.5",
                mutation="gaussian:mutation_rate=0.1",
                generations=50,
                pop_size=30
            )
        """
        if output_dir is None:
            output_dir = Path("results") / experiment_name
        else:
            output_dir = Path(output_dir)

        if engine is None:
            if self._has_real_operators(fitness, selection, crossover, mutation):
                engine = self._build_real_engine(
                    fitness=fitness,
                    selection=selection,
                    crossover=crossover,
                    mutation=mutation,
                    genome_type=genome_type,
                    pop_size=pop_size,
                    generations=generations,
                    genome_length=genome_length,
                    bounds=bounds,
                    elitism=elitism,
                    **kwargs,
                )
            else:
                engine = self._build_stub_engine(generations, **kwargs)

        runner = BenchmarkRunner(
            engine=engine,
            experiment_name=experiment_name,
            output_dir=output_dir,
            write_artifacts=True,
        )

        return runner.run(seeds)

    def _has_real_operators(
        self,
        fitness: Optional[str],
        selection: Optional[str],
        crossover: Optional[str],
        mutation: Optional[str],
    ) -> bool:
        """Check if any real operator specs are provided."""
        return any([fitness, selection, crossover, mutation])

    def _build_real_engine(
        self,
        fitness: Optional[str],
        selection: Optional[str],
        crossover: Optional[str],
        mutation: Optional[str],
        **config: Any,
    ) -> Any:
        """Build real GeneticEngine from operator specs and config."""
        catalog = OperatorCatalog()

        catalog_operators = {
            "fitness": catalog.get(fitness or "sphere:dim=10"),
            "selection": catalog.get(
                selection
                or f"tournament:num_selections={config['pop_size'] // 2},tournament_size=3"
            ),
            "crossover": catalog.get(crossover or "blend:alpha=0.5"),
            "mutation": catalog.get(mutation or "gaussian:mutation_rate=0.1"),
        }

        return build_engine_from_catalog(catalog_operators, config)

    def _build_stub_engine(self, generations: int, **kwargs: Any) -> StubEngine:
        """Build StubEngine with legacy behavior for backward compatibility."""
        base_fitness = kwargs.get("base_fitness", 1.0)
        improvement_rate = kwargs.get("improvement_rate", 0.1)

        return StubEngine(
            generations=generations,
            base_fitness=base_fitness,
            improvement_rate=improvement_rate,
        )

    @classmethod
    def create_default(cls) -> "Composer":
        """Create composer with default configuration."""
        return cls(config={"version": "0.1"})
