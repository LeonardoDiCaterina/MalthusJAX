from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..benchmarking import BenchmarkRunner, ExperimentResult, StubEngine
from .catalog import OperatorCatalog
from .engine_factory import build_engine_from_catalog
from .evosax_adapter import build_evosax_engine


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
        # Backend selection
        backend: str = "malthusjax",
        evosax_strategy: str = "SimpleGA",
        # Real operator specifications (malthusjax backend)
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
        maximize: bool = False,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Quick-run an experiment with sensible defaults.

        This is the main product-first entry point for running experiments.
        Supports two backends:

        - ``"malthusjax"`` (default): Uses MalthusJAX operators via string specs
        - ``"evosax"``: Uses evosax population-based strategies (ask/tell)

        When no operator specs are provided, falls back to StubEngine.

        Args:
            seeds: Random seeds to run (default: 3 seeds)
            experiment_name: Name for the experiment
            output_dir: Where to write results
            engine: Pre-built engine (overrides everything if provided)
            backend: ``"malthusjax"`` or ``"evosax"`` (default: ``"malthusjax"``)
            evosax_strategy: Evosax strategy name when backend="evosax"
                (``"SimpleGA"``, ``"MR15_GA"``, ``"DifferentialEvolution"``)
            fitness: Fitness evaluator spec, e.g. ``"sphere:dim=10"``
            selection: Selection operator spec (malthusjax only)
            crossover: Crossover operator spec (malthusjax only)
            mutation: Mutation operator spec (malthusjax only)
            genome_type: ``"real"`` or ``"binary"`` (default: ``"real"``)
            pop_size: Population size (default: 50)
            generations: Number of generations (default: 100)
            genome_length: Length of genome vectors (default: 10)
            bounds: Bounds for real genomes (default: (-5.0, 5.0))
            elitism: Elite count (malthusjax only, default: 2)
            maximize: Report fitness in maximisation convention (default: False)
            **kwargs: Additional config passed to engine/runner

        Returns:
            ExperimentResult with all runs and aggregated metrics

        Examples::

            # MalthusJAX backend (default)
            result = composer.quick_run(
                fitness="sphere:dim=5",
                selection="tournament:num_selections=25,tournament_size=3",
                crossover="blend:alpha=0.5",
                mutation="gaussian:mutation_rate=0.1",
            )

            # Evosax backend
            result = composer.quick_run(
                backend="evosax",
                evosax_strategy="SimpleGA",
                fitness="sphere:dim=10",
                pop_size=100,
                generations=200,
            )

            # StubEngine fallback (no operators specified)
            result = composer.quick_run()
        """
        if output_dir is None:
            output_dir = Path("results") / experiment_name
        else:
            output_dir = Path(output_dir)

        if engine is None:
            if backend == "evosax":
                engine = self._build_evosax_engine(
                    strategy_name=evosax_strategy,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    num_dims=genome_length,
                    bounds=bounds,
                    maximize=maximize,
                    **kwargs,
                )
            elif self._has_real_operators(fitness, selection, crossover, mutation):
                engine = self._build_real_engine(
                    fitness=fitness,
                    selection=selection,
                    crossover=crossover,
                    mutation=mutation,
                    genome_type=genome_type,
                    pop_size=pop_size,
                    generations=generations,
                    genome_shape=genome_length,
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

    def _build_evosax_engine(
        self,
        strategy_name: str,
        fitness_spec: Optional[str],
        pop_size: int,
        generations: int,
        num_dims: int,
        bounds: tuple,
        maximize: bool,
        **kwargs: Any,
    ) -> Any:
        """Build EvosaxEngineAdapter from strategy name and config."""
        return build_evosax_engine(
            strategy_name=strategy_name,
            fitness_spec=fitness_spec or "sphere:dim=" + str(num_dims),
            num_dims=num_dims,
            pop_size=pop_size,
            generations=generations,
            bounds=bounds,
            maximize=maximize,
            seed=kwargs.get("seed", 42),
            strategy_params=kwargs.get("strategy_params"),
        )

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
