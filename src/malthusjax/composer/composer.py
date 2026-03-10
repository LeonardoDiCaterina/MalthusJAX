from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.random as jr

from ..benchmarking import BenchmarkRunner, ExperimentResult, StubEngine
from ..benchmarking.results import ComparisonResult
from .catalog import OperatorCatalog
from .config import load_experiment_config
from .engine_catalog import EngineRegistry
from .evosax_adapter import build_evosax_engine

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class Composer:
    """Compose and run evolutionary experiments with sensible defaults."""

    registry: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def quick_run(
        self,
        seeds: Sequence[int] = (1, 2, 3),
        experiment_name: str = "quick_experiment",
        output_dir: Optional[Path | str] = None,
        engine: Optional[Any] = None,
        # Backend selection
        backend: str = "malthusjax",
        engine_type: str = "ga",
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
        bounds: Tuple[float, float] = (-5.0, 5.0),
        elitism: int = 2,
        maximize: bool = False,
        prng_impl: Optional[str] = None,
        trace_dir: Optional[Path | str] = Path("results/traces"),
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
            engine_type: Engine type spec for malthusjax backend, e.g.
                ``"ga"`` (default), ``"ga:elitism=4"``, or any registered
                engine.  Use :class:`EngineRegistry` to list/register engines.
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
            prng_impl: JAX PRNG backend — ``"threefry"`` (default),
                ``"philox"`` (GPU-friendly), ``"rbg"``, or ``"unsafe_rbg"``.
                Controls how random keys are created and split.
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
                    prng_impl=prng_impl,
                    **kwargs,
                )
            elif self._has_real_operators(fitness, selection, crossover, mutation):
                engine = self._build_real_engine(
                    fitness=fitness,
                    selection=selection,
                    crossover=crossover,
                    mutation=mutation,
                    engine_type=engine_type,
                    genome_type=genome_type,
                    pop_size=pop_size,
                    generations=generations,
                    genome_shape=genome_length,
                    bounds=bounds,
                    elitism=elitism,
                    prng_impl=prng_impl,
                    **kwargs,
                )
            else:
                engine = self._build_stub_engine(generations, **kwargs)

        runner = BenchmarkRunner(
            engine=engine,
            experiment_name=experiment_name,
            output_dir=output_dir,
            write_artifacts=True,
            prng_impl=prng_impl,
            trace_dir=Path(trace_dir) if trace_dir is not None else None,
        )

        return runner.run(seeds)

    def compare(
        self,
        pipelines: Dict[str, Dict[str, Any]],
        seeds: Sequence[int] = (42, 43, 44),
        shared_initial_population: bool = True,
        pop_seed: int = 123,
        **shared_kwargs: Any,
    ) -> ComparisonResult:
        """Run multiple pipelines and return aligned results.

        Parameters
        ----------
        pipelines
            ``{name: kwargs_dict}`` — each dict contains **overrides**
            for :meth:`quick_run`.  Shared parameters should go into
            *shared_kwargs*.
        seeds
            Random seeds passed to every pipeline.
        shared_initial_population
            If ``True`` (default), generate a shared initial population
            so every pipeline starts from the same point.
        pop_seed
            Seed for generating the shared initial population.
        **shared_kwargs
            Parameters common to every pipeline (``fitness``,
            ``pop_size``, ``generations``, ``genome_length``, ``bounds``,
            ``prng_impl``, etc.).  Pipeline-level keys take precedence.

        Returns
        -------
        ComparisonResult
            Contains all :class:`ExperimentResult` objects keyed by
            pipeline name, plus helper methods for summarisation and
            plotting.

        Examples
        --------
        ::

            cmp = composer.compare(
                pipelines={
                    "Blend+Gaussian": dict(
                        crossover="blend:alpha=0.5",
                        mutation="gaussian:mutation_rate=0.1",
                    ),
                    "SBX+Polynomial": dict(
                        crossover="simulated_binary:eta=2",
                        mutation="polynomial:mutation_rate=0.1",
                    ),
                    "Evosax SimpleGA": dict(
                        backend="evosax",
                        evosax_strategy="SimpleGA",
                    ),
                },
                fitness="sphere:dim=10",
                pop_size=50,
                generations=100,
                seeds=(42, 43),
            )
            cmp.plot_convergence()
        """
        init_pop = None
        if shared_initial_population:
            pop_size = int(shared_kwargs.get("pop_size", 50))
            genome_length = int(shared_kwargs.get("genome_length", 10))
            bounds = shared_kwargs.get("bounds", (-5.0, 5.0))
            init_pop = jr.uniform(
                jr.PRNGKey(pop_seed),
                (pop_size, genome_length),
                minval=float(bounds[0]),
                maxval=float(bounds[1]),
            )

        trace_base = shared_kwargs.pop("trace_dir", None)

        results: Dict[str, ExperimentResult] = {}
        negate_map: Dict[str, bool] = {}

        iterable = pipelines.items()
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(pipelines), desc="pipelines")

        for name, pipeline_kwargs in iterable:
            merged = {**shared_kwargs, **pipeline_kwargs}
            merged["seeds"] = seeds
            merged.setdefault("experiment_name", name)

            if trace_base is not None:
                merged["trace_dir"] = Path(trace_base) / name

            if init_pop is not None and "initial_population" not in merged:
                merged["initial_population"] = init_pop

            results[name] = self.quick_run(**merged)
            backend = merged.get("backend", "malthusjax")
            negate_map[name] = backend != "evosax"

        return ComparisonResult(
            pipelines=results,
            shared_config=dict(shared_kwargs),
            initial_population=init_pop,
            negate_map=negate_map,
        )

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
        pipelines: Optional[List[str]] = None,
        shared_initial_population: bool = True,
        pop_seed: int = 123,
        trace_dir: Optional[str | Path] = None,
    ) -> ComparisonResult:
        """Load a TOML experiment file and run all pipelines.

        Parameters
        ----------
        path
            Path to a Composer-style TOML file.
        pipelines
            Optional list of pipeline names to run.  If ``None``, all
            pipelines defined in the file are executed.
        shared_initial_population
            Generate a shared initial population for fair comparison.
        pop_seed
            Seed for the shared initial population.
        trace_dir
            If set, capture JAX profiler traces for seed[0] of each
            pipeline.  Traces are written to ``trace_dir/<pipeline_name>/``
            in Perfetto-compatible format.

        Returns
        -------
        ComparisonResult

        Examples
        --------
        ::

            # experiment.toml
            # [experiment.shared]
            # fitness = "sphere:dim=10"
            # pop_size = 50
            # ...
            # [pipelines.blend_ga]
            # crossover = "blend:alpha=0.5"
            # [pipelines.sbx_ga]
            # crossover = "simulated_binary:eta=2.0"

            result = Composer.from_toml("experiment.toml")
            result.plot_convergence()
        """
        experiment_meta, resolved = load_experiment_config(str(path), pipelines=pipelines)
        shared = experiment_meta.get("shared", {})

        seeds = tuple(shared.pop("seeds", (42, 43, 44)))

        pipeline_overrides: Dict[str, Dict[str, Any]] = {}
        for name, merged_cfg in resolved.items():
            overrides = {
                k: v
                for k, v in merged_cfg.items()
                if (k not in shared) or (merged_cfg[k] != shared.get(k))
            }
            pipeline_overrides[name] = overrides

        composer = cls.create_default()
        return composer.compare(
            pipelines=pipeline_overrides,
            seeds=seeds,
            shared_initial_population=shared_initial_population,
            pop_seed=pop_seed,
            trace_dir=trace_dir,
            **shared,
        )

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
        engine_type: str = "ga",
        **config: Any,
    ) -> Any:
        """Build engine from operator specs and config via EngineRegistry."""
        catalog = OperatorCatalog()

        resolved_evaluator = catalog.get(fitness or "sphere:dim=10")
        resolved_selection = catalog.get(
            selection
            or f"tournament:num_selections={config['pop_size'] // 2},tournament_size=3"
        )
        resolved_crossover = catalog.get(crossover or "blend:alpha=0.5")
        resolved_mutation = catalog.get(mutation or "gaussian:mutation_rate=0.1")

        engine_registry = EngineRegistry()
        return engine_registry.get(
            engine_type,
            evaluator=resolved_evaluator,
            selection=resolved_selection,
            crossover=resolved_crossover,
            mutation=resolved_mutation,
            **config,
        )

    def _build_evosax_engine(
        self,
        strategy_name: str,
        fitness_spec: Optional[str],
        pop_size: int,
        generations: int,
        num_dims: int,
        bounds: Tuple[float, float],
        maximize: bool,
        prng_impl: Optional[str] = None,
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
            initial_population=kwargs.get("initial_population"),
            prng_impl=prng_impl,
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
