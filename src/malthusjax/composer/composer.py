"""High-level API for declarative and programmatic experiments.

The :class:`Composer` class wraps the benchmarking system with convenient
helper methods like :meth:`quick_run`, :meth:`compare`, and :meth:`from_toml`.
It orchestrates engine construction, seeding, result aggregation and provides
sensible defaults for rapid prototyping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.random as jr

from malthusjax.composer.config import infer_genome_length, normalize_seeds
from malthusjax.composer.factory import (
    build_evosax_engine,
    build_map_elites_engine,
    build_qdax_engine,
    build_real_engine,
    build_stub_engine,
    build_tensorneat_engine,
    has_real_operators,
)
from malthusjax.composer.strategies.base import BaseStrategy
from malthusjax.composer.strategies.core import (
    EvoSAXStrategy,
    GeneticStrategy,
    MapElitesStrategy,
    QDAXStrategy,
    TensorNEATStrategy,
)
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

from ..benchmarking import BenchmarkRunner, ExperimentResult
from ..benchmarking.results import ComparisonResult
from .catalog import OperatorCatalog
from .config import load_experiment_config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None




@dataclass
class Composer:
    """High-level interface for running evolutionary algorithms.

    The :class:`Composer` class provides three primary entry points for
    running genetic algorithms with different levels of configuration:

    1. **:meth:`quick_run` — Interactive exploration**
       Use when exploring hyperparameters, testing fitness functions, or
       learning MalthusJAX. Accepts programmatic operator specifications
       (e.g., ``fitness="sphere:dim=10"``) and runs a full experiment with
       multiple seeds, returning aggregated results immediately.

    2. **:meth:`from_toml` — Reproducible declarative experiments**
       Use when you want versioned, reproducible experiments for papers or
       production benchmarking. Define pipelines in a TOML file; the method
       loads, parses, and executes all pipelines automatically while
       preserving shared configuration like seeds and population bounds.

    3. **:meth:`compare` — Programmatic multi-pipeline benchmarking**
       Use when comparing algorithm variants. Accepts a dict of pipeline
       configurations (each a :meth:`quick_run` kwargs dict), shares initial
       populations and seeds across pipelines for fair comparison, and returns
       utilities for statistical summary and visualization.

    All methods return result objects (:class:`ExperimentResult` from
    :meth:`quick_run`, :class:`ComparisonResult` from :meth:`from_toml` and
    :meth:`compare`) with aggregated metrics and convergence histories.

    Examples
    --------
    Quick exploration::

        composer = Composer.create_default()
        result = composer.quick_run(
            fitness="sphere:dim=5",
            selection="tournament:num_selections=25,tournament_size=3",
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1,mutation_strength=0.2",
            pop_size=50,
            generations=50,
            seeds=(42, 43, 44),
        )
        print(result.aggregated_summary())

    Reproducible TOML-based experiment::

        result = Composer.from_toml(
            "experiment.toml",
            trace_dir="results/traces"
        )
        result.plot_convergence(seed_index=0)

    Benchmarking multiple pipelines::

        comparison = composer.compare(
            pipelines={
                "Blend+Gaussian": dict(
                    crossover="blend:alpha=0.5",
                    mutation="gaussian:mutation_rate=0.1",
                ),
                "SBX+Polynomial": dict(
                    crossover="simulated_binary:eta=2",
                    mutation="polynomial:mutation_rate=0.1",
                ),
            },
            fitness="sphere:dim=10",
            pop_size=50,
            generations=100,
        )
        print(comparison.summary_table())
        comparison.plot_convergence()
    """

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
        qdax_strategy: str = "MAPElites",
        tensorneat_algorithm: str = "NEAT",
        tensorneat_genome: str = "default",
        tensorneat_problem: Optional[str] = None,
        tensorneat_num_inputs: int = 2,
        tensorneat_num_outputs: int = 1,
        qdax_num_descriptors: int = 2,
        qdax_num_centroids: int = 100,
        qdax_mutation_sigma: float = 0.1,
        # Real operator specifications (malthusjax backend)
        strategy: Optional[BaseStrategy] = None,
        genome: Optional[str] = None,
        fitness: Optional[str] = None,
        selection: Optional[str] = None,
        crossover: Optional[str] = None,
        mutation: Optional[str] = None,
        genome_type: Optional[str] = None,
        pop_size: int = 50,
        generations: int = 100,
        genome_length: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        elitism: int = 2,
        maximize: bool = False,
        prng_impl: Optional[str] = None,
        use_history_for_final: bool = False,
        trace_dir: Optional[Path | str] = None,
        data_config: Optional[Dict[str, Any]] = None,
        history_metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run a full evolutionary experiment with programmatic operator specs.

        This is the main entry point for interactive exploration and benchmarking.
        It accepts operator specifications as string specs (for the ``"malthusjax"``
        backend) or strategy names (for the ``"evosax"`` backend), orchestrates
        multiple independent runs (one per seed), aggregates results, and returns
        an :class:`ExperimentResult`.

        Parameters
        ----------
        fitness : str, optional
            Fitness function specification (MalthusJAX backend only).
            Format: ``"name:key1=val,key2=val"``

            **Built-in fitness functions**:

            - ``"sphere:dim=INT"`` — Sphere function: $f(x) = \\sum_i x_i^2$
            - ``"rastrigin:dim=INT"`` — Rastrigin: highly multimodal
            - ``"griewank:dim=INT"`` — Griewank: multimodal with low local frequencies
            - ``"bbob:fn=INT,dims=INT"`` — Black-Box Optimization Benchmarking
              (BBOB) suite function (fn=1–24, dims=2–40)
            - ``"knapsack:capacity=INT,num_items=INT"`` — 0/1 knapsack
            - ``"binary_sum:length=INT"`` — Binary sum of bits (for binary genomes)

            Examples: ``"sphere:dim=10"``, ``"bbob:fn=3,dims=10"``,
            ``"knapsack:capacity=100,num_items=20"``.

        selection : str, optional
            Selection operator specification.
            Format: ``"name:key1=val,key2=val"``

            **Valid operators**:

            - ``"tournament:num_selections=INT,tournament_size=INT"``
              Select via tournament competition (default: size=3).
            - ``"roulette:num_selections=INT"``
              Fitness-proportional selection (SUS-like).
            - ``"elite_pool:num_selections=INT,elite_k=INT"``
              Keep top-k individuals, select from them.

            Examples: ``"tournament:num_selections=25,tournament_size=3"``,
            ``"roulette:num_selections=25"``.

        crossover : str, optional
            Crossover operator specification.
            Format: ``"name:key1=val,key2=val"``

            **For real genomes**:

            - ``"uniform_real"`` — Uniform crossover (each gene has 50% chance).
            - ``"blend:alpha=FLOAT"`` — Blend crossover with expansion factor.
            - ``"simulated_binary:eta=FLOAT"`` — SBX (eta controls spread).
            - ``"binomial:cr=FLOAT"`` — DE-style binomial crossover.

            **For binary genomes**:

            - ``"uniform_binary"`` — Uniform bit-wise crossover.
            - ``"single_point"`` — Single-point crossover.

            Examples: ``"blend:alpha=0.5"``, ``"simulated_binary:eta=20"``.

        mutation : str, optional
            Mutation operator specification.
            Format: ``"name:key1=val,key2=val"``

            **For real genomes**:

            - ``"gaussian:mutation_rate=FLOAT,mutation_strength=FLOAT"``
              Add Gaussian noise (mutation_strength = std dev).
            - ``"ball:mutation_rate=FLOAT"``
              Uniform ball (hypersphere) mutation.
            - ``"polynomial:mutation_rate=FLOAT,eta=FLOAT"``
              Polynomial mutation (eta controls distribution).

            **For binary genomes**:

            - ``"bitflip:mutation_rate=FLOAT"``
              Flip each bit independently.
            - ``"scramble"`` — Random reordering (order-based).
            - ``"swap"`` — Random bit swap (order-based).

            Examples:
            ``"gaussian:mutation_rate=0.5,mutation_strength=0.1"``,
            ``"polynomial:mutation_rate=0.1,eta=20"``,
            ``"bitflip:mutation_rate=0.05"``.

        seeds : Sequence[int], optional
            Random seeds for independent runs. Each seed generates a fully
            independent evolutionary run; results are then aggregated.
            Default: ``(1, 2, 3)``.

        generations : int, optional
            Number of full generational cycles to evolve. Default: 100.
            (Note: some backends may interpret this differently; see backend-specific
            documentation.)

        pop_size : int, optional
            Population size (number of individuals per generation).
            For GPU efficiency, powers of 2 (32, 64, 128, 256) are preferred.
            Default: 50.

        genome_length : int, optional
            For continuous genomes: number of decision variables (dimension).
            For discrete: number of bits or items. Default: 10.

        bounds : Tuple[float, float], optional
            Search space bounds ``(lower, upper)`` for real genomes.
            Default: ``(-5.0, 5.0)``.

        genome_type : str, optional
            Genome representation: ``"real"`` or ``"binary"``.
            Controls which operator specs are valid (real vs binary crossover/mutation).
            Default: ``"real"``.

        genome : str, optional
            Declarative specification of the genome type, shape, and bounds.
            Format: ``"type:key1=val,key2=val"``
            Examples: ``"real:dim=10,bounds=(-5.0, 5.0)"``, ``"binary:length=20"``.
            Explicit arguments (e.g., `genome_type`, `genome_length`) override the
            values provided in this specification.

        maximize : bool, optional
            Optimization direction: ``True`` for maximization, ``False`` for
            minimization. Default: ``False``.

        backend : str, optional
            Execution backend: ``"malthusjax"`` (default) or ``"evosax"``.

            - ``"malthusjax"``: Uses MalthusJAX operators (requires
              fitness/selection/crossover/mutation specs)
            - ``"evosax"``: Uses evosax strategy (requires evosax_strategy; ignores
              operator specs)

        evosax_strategy : str, optional
            Evosax strategy name (only if ``backend="evosax"``).
            Examples: ``"SimpleGA"``, ``"OpenES"``, ``"CMA_ES"``, ``"DE"``.
            Default: ``"SimpleGA"``.

        engine_type : str, optional
            MalthusJAX engine type (only if ``backend="malthusjax"``).
            Currently supported: ``"ga"`` (generational GA). Default: ``"ga"``.

        elitism : int, optional
            Number of best individuals to carry forward without modification
            (MalthusJAX backend only). Default: 2.

        experiment_name : str, optional
            Experiment identifier (used in output directory and logging).
            Default: ``"quick_experiment"``.

        output_dir : Path or str, optional
            Output directory for results, artifacts, and logs.
            If not provided, defaults to ``results/{experiment_name}/``.

        engine : optional
            A pre-configured engine object. If provided, all operator specs
            (fitness, selection, crossover, mutation) are ignored; use a custom
            engine to bypass automatic configuration.

        prng_impl : str, optional
            PRNG implementation for JAX key splitting (advanced).
            Default: ``None`` (auto-select).

        trace_dir : Path or str, optional
            Directory to write Perfetto-compatible JAX profiler traces.
            Default: ``None`` (disabled).

        Returns
        -------
        ExperimentResult
            Result object containing per-seed run records and aggregation methods:

            - ``.runs`` : List[:class:`RunResult`] — Individual runs, one per seed
            - ``.aggregated_summary()`` — Dict mapping metric names → (mean, median, stdev)
            - ``.combined_history(seed_field="seed")`` — All generation records with seed labels
            - ``.canonical_summary`` — Best-of metrics from first seed

        Raises
        ------
        ValueError
            If operator specs have invalid format or unknown operator names.
        RuntimeError
            If the underlying engine encounters a critical error during execution.

        Examples
        --------
        MalthusJAX backend (default)::

            composer = Composer.create_default()
            result = composer.quick_run(
                fitness="sphere:dim=25",
                selection="tournament:num_selections=25,tournament_size=3",
                crossover="blend:alpha=0.5",
                mutation="gaussian:mutation_rate=0.5,mutation_strength=0.1",
                pop_size=100,
                generations=200,
                seeds=(42, 43, 44),
            )
            print(result.aggregated_summary())

        Evosax backend::

            result = composer.quick_run(
                backend="evosax",
                evosax_strategy="OpenES",
                fitness="sphere:dim=10",
                pop_size=64,
                generations=500,
                seeds=(1, 2, 3),
            )

        No operator specs (uses StubEngine for testing the pipeline)::

            result = composer.quick_run(
                generations=50,
                seeds=(1, 2),
            )
        """
        if output_dir is None:
            output_dir = Path("results") / experiment_name
        else:
            output_dir = Path(output_dir)

        if engine is None:
            # Resolve genome defaults and specification
            if genome is not None:
                from .genome_catalog import GenomeCatalog

                cat = GenomeCatalog()
                g_type, g_params = cat.parse_spec(genome)
                if genome_type is None:
                    genome_type = g_type
                if genome_length is None:
                    genome_length = g_params.get("dim", g_params.get("length", 10))
                if bounds is None and "bounds" in g_params:
                    b = g_params["bounds"]
                    if isinstance(b, str):
                        b_str = b.strip("()[]")
                        parts = b_str.split(",")
                        bounds = (float(parts[0]), float(parts[1]))
                    else:
                        bounds = tuple(b)
                if genome_length is None and "shape" in g_params:
                    shape_val = g_params["shape"]
                    genome_length = shape_val[0] if hasattr(shape_val, "__len__") else shape_val

            # If genome length is still unset, infer from fitness spec (e.g. "sphere:dim=5").
            if genome_length is None and isinstance(fitness, str):
                parsed_name, parsed_params = OperatorCatalog().parse_spec(fitness)
                _ = parsed_name  # parsed_name intentionally unused; kept for clarity.
                dim_val = parsed_params.get("dim", parsed_params.get("num_dims"))
                if dim_val is not None:
                    genome_length = int(dim_val)

            if genome_type is None:
                genome_type = "real"
            if genome_length is None:
                genome_length = 10
            if bounds is None:
                bounds = (-5.0, 5.0)

            if strategy is None:
                if backend == "evosax":
                    strategy = EvoSAXStrategy(algorithm_name=evosax_strategy)
                elif backend == "qdax":
                    strategy = QDAXStrategy(
                        strategy_cls=qdax_strategy,
                        num_descriptors=qdax_num_descriptors,
                        num_centroids=qdax_num_centroids,
                        mutation_sigma=qdax_mutation_sigma,
                        algorithm_kwargs=kwargs,
                    )
                elif backend == "tensorneat":
                    strategy = TensorNEATStrategy(
                        algorithm_name=tensorneat_algorithm,
                        genome_name=tensorneat_genome,
                        problem_name=tensorneat_problem,
                        num_inputs=tensorneat_num_inputs,
                        num_outputs=tensorneat_num_outputs,
                        algorithm_kwargs=kwargs,
                    )
                elif backend == "malthusjax" and engine_type == "map_elites":
                    strategy = MapElitesStrategy(
                        emitter=kwargs.get("map_elites_emitter", kwargs.get("emitter", "mixing")),
                        num_descriptors=qdax_num_descriptors,
                        num_centroids=qdax_num_centroids,
                    )
                elif has_real_operators(genome, fitness, selection, crossover, mutation):
                    strategy = GeneticStrategy(
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                    )

            if isinstance(strategy, EvoSAXStrategy):
                engine = build_evosax_engine(
                    strategy_name=strategy.algorithm_name,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    num_dims=genome_length,
                    bounds=bounds,
                    maximize=maximize,
                    prng_impl=prng_impl,
                    history_metrics=history_metrics,
                    **strategy.algorithm_kwargs,
                    **kwargs,
                )
            elif isinstance(strategy, QDAXStrategy):
                engine = build_qdax_engine(
                    strategy=strategy,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    genome_length=genome_length,
                    bounds=bounds,
                    maximize=maximize,
                    history_metrics=history_metrics,
                    **kwargs,
                )
            elif isinstance(strategy, TensorNEATStrategy):
                engine = build_tensorneat_engine(
                    strategy=strategy,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    maximize=maximize,
                    history_metrics=history_metrics,
                    **kwargs,
                )
            elif isinstance(strategy, MapElitesStrategy):
                engine = build_map_elites_engine(
                    strategy=strategy,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    maximize=maximize,
                    history_metrics=history_metrics,
                    genome_length=genome_length,
                    bounds=bounds,
                    **kwargs,
                )
            elif isinstance(strategy, GeneticStrategy):
                engine = build_real_engine(
                    strategy=strategy,
                    genome=genome,
                    fitness=fitness,
                    engine_type=engine_type,
                    genome_type=genome_type,
                    pop_size=pop_size,
                    generations=generations,
                    genome_shape=genome_length,
                    bounds=bounds,
                    elitism=elitism,
                    maximize=maximize,
                    prng_impl=prng_impl,
                    data_config=data_config,
                    history_metrics=history_metrics,
                    **kwargs,
                )
            else:
                engine = build_stub_engine(generations, **kwargs)

        runner = BenchmarkRunner(
            engine=engine,
            experiment_name=experiment_name,
            output_dir=output_dir,
            write_artifacts=True,
            prng_impl=prng_impl,
            trace_dir=Path(trace_dir) if trace_dir is not None else Path("results/traces"),
            serialize_history=kwargs.get("serialize_history", True),
        )

        normalized_seeds = normalize_seeds(seeds)
        experiment = runner.run(normalized_seeds)

        # Composer-level postprocessing: when engines run with
        # TrackBest.NONE for speed they may omit or produce an
        # invalid `summary['best_fitness']`. Allow callers to request
        # that the final best is derived from the last history entry
        # (history[-1]) instead. We also auto-fix non-finite summaries.
        self._postprocess_experiment_final_from_history(experiment, use_history_for_final)

        return experiment

    def _postprocess_experiment_final_from_history(
        self, experiment: Any, force: bool = False
    ) -> None:
        """Ensure per-run summary best_fitness is consistent with history.

        When engines disable internal best-tracking (for speed), the
        returned ``summary`` may be missing or invalid. This helper will
        copy the final generation's best metrics from ``run.history[-1]``
        into ``run.metrics['best_fitness']`` (and ``final_generation``)
        when either *force* is True or when the existing metric is
        absent/non-finite.
        """
        import math

        for run in experiment.runs:
            # skip errored runs
            if run.status != "success":
                continue

            # determine whether we should replace the summary
            best_val = run.metrics.get("best_fitness")
            need_replace = force or (
                best_val is None
                or (
                    isinstance(best_val, (int, float))
                    and (math.isnan(best_val) or math.isinf(best_val))
                )
            )

            if not need_replace:
                continue

            if not run.history:
                # nothing to do
                continue

            last = run.history[-1]
            if "best_fitness" in last:
                try:
                    run.metrics["best_fitness"] = float(last["best_fitness"])
                except Exception:
                    pass

            if "generation" in last:
                try:
                    run.metrics.setdefault("final_generation", int(last["generation"]))
                except Exception:
                    pass


    def _generate_initial_population(self, config: Dict[str, Any], pop_seed: int) -> Any:
        """Deterministically generate a shared initial population matrix for a given pipeline config.

        Ensures that pipelines with identical bounds, population sizes, and dimensionality
        receive the exact same starting points, while dynamically scaling to the requested pop_size.
        """

        pop_size = int(config.get("pop_size", 50))
        genome_length = infer_genome_length(config)
        bounds = config.get("bounds", (-5.0, 5.0))
        fitness_spec = config.get("fitness")

        # Check if this pipeline uses TensorNEAT topology evolution
        if config.get("backend") == "tensorneat" or config.get("genome_type") == "tensorneat":
            import tensorneat.algorithm
            import tensorneat.genome
            from tensorneat.common import State

            # For parity, we just need a shared structure.
            # We'll assume default genome and minimal sizes if not provided.
            num_inputs = config.get("num_inputs", 2)
            num_outputs = config.get("num_outputs", 1)

            genome = tensorneat.genome.DefaultGenome(num_inputs=num_inputs, num_outputs=num_outputs)
            algorithm = tensorneat.algorithm.NEAT(pop_size=pop_size, genome=genome)

            state = State(randkey=jr.PRNGKey(pop_seed))
            state = algorithm.setup(state)

            # Extract the raw matrices to be shared as initial_population
            pop_nodes = getattr(state, "pop_nodes", state.state_dict.get("pop_nodes"))
            pop_conns = getattr(state, "pop_conns", state.state_dict.get("pop_conns"))
            return (pop_nodes, pop_conns)


        if fitness_spec and isinstance(fitness_spec, str) and "bbob" in fitness_spec.lower():
            cat = OperatorCatalog()
            parsed_name, parsed_params = cat.parse_spec(fitness_spec)
            if parsed_name == "bbob":
                fn = parsed_params.get("fn_name", parsed_params.get("fn", "rosenbrock"))
                dims = parsed_params.get("dim", parsed_params.get("num_dims", genome_length))
                bbob_seed = parsed_params.get("seed", 0)
                bbob_eval = BBOBEvaluator.create(
                    BBOBConfig(
                        fn_name=fn,
                        num_dims=dims,
                        seed=bbob_seed,
                        maximize=config.get("maximize", False),
                    )
                )
                pop_key = jr.PRNGKey(pop_seed)
                sample_keys = jr.split(pop_key, pop_size)
                return jax.vmap(bbob_eval.evosax_problem.sample)(sample_keys)

        return jr.uniform(
            jr.PRNGKey(pop_seed),
            (pop_size, genome_length),
            minval=float(bounds[0]),
            maxval=float(bounds[1]),
        )

    def compare(
        self,
        pipelines: Dict[str, Dict[str, Any]],
        seeds: Sequence[int] = (42, 43, 44),
        shared_initial_population: bool = True,
        pop_seed: int = 123,
        **shared_kwargs: Any,
    ) -> ComparisonResult:
        """Run multiple algorithm variants and return comparison results.

        This method executes several experiment pipelines with identical seeding
        and optional population initialization, enabling fair statistical
        comparison of different evolutionary strategies.

        Parameters
        ----------
        pipelines : Dict[str, Dict[str, Any]]
            Mapping of display names (pipeline identifiers) to kwarg dicts.
            Each dict contains :meth:`quick_run` parameters (fitness, selection,
            crossover, mutation, backend, evosax_strategy, etc.) that override
            the shared configuration. Keys determine plot legend labels and table
            headers in results.

            Example::

                pipelines={
                    "Blend+Gaussian": dict(
                        crossover="blend:alpha=0.5",
                        mutation="gaussian:mutation_rate=0.1",
                    ),
                    "SBX+Polynomial": dict(
                        crossover="simulated_binary:eta=2.0",
                        mutation="polynomial:mutation_rate=0.1,eta=20",
                    ),
                    "Evosax CMA-ES": dict(
                        backend="evosax",
                        evosax_strategy="CMA_ES",
                    ),
                }

            Backend selection: Include ``backend`` and ``evosax_strategy`` in
            each pipeline dict to mix MalthusJAX and Evosax strategies in one
            comparison.

        seeds : Sequence[int], optional
            Random seeds shared across all pipelines. Using the same seeds
            ensures statistically comparable runs. Default: ``(42, 43, 44)``.

        shared_initial_population : bool, optional
            If ``True``, generate a single random population and inject it into
            every pipeline's first run. This isolates algorithmic differences
            from initialization variance, enhancing fair comparison.
            Default: ``True``.

        pop_seed : int, optional
            Random seed for generating the shared initial population.
            Default: 123.
            (Only used if ``shared_initial_population=True``.)

        shared_kwargs : dict, optional
            Default configuration applied to all pipelines (merged before
            pipeline-specific overrides). Use for common settings like
            ``fitness``, ``pop_size``, ``generations``, ``bounds``.

            Example::

                compare(
                    pipelines={...},
                    fitness="sphere:dim=10",  # shared
                    pop_size=50,               # shared
                    generations=100,           # shared
                )

        Returns
        -------
        ComparisonResult
            Contains:

            - ``.pipelines`` : Dict[str, :class:`ExperimentResult`]
              Per-pipeline results with aggregation methods.
            - ``.summary_table()`` : Dict[str, Dict[str, float]]
              Per-pipeline aggregated metrics (mean across seeds).
              Fitness is always normalized to "lower is better".
            - ``.plot_convergence(seed_index=0, ax=None)`` -> matplotlib axis
              Overlay convergence curves for all pipelines.
            - ``.convergence_data(seed_index=0)`` : Dict[str, List[Dict]]
              Raw per-pipeline generation histories for custom plotting.

        Raises
        ------
        ValueError
            If pipelines dict is empty or contains invalid operator specs.
        KeyError
            If required shared config keys (e.g., fitness, pop_size) are missing
            from both shared and pipeline-specific settings.

        Notes
        -----
        The method automatically handles backend normalization: Evosax pipelines
        have their fitness values negated before display so that all pipelines
        use a consistent "lower is better" convention. This behavior is
        transparent to the user (controlled by ComparisonResult.negate_map).

        Examples
        --------
        Three-way comparison with shared fitness::

            composer = Composer.create_default()
            comparison = composer.compare(
                pipelines={
                    "Blend+Gaussian": dict(
                        crossover="blend:alpha=0.5",
                        mutation="gaussian:mutation_rate=0.5,mutation_strength=0.1",
                    ),
                    "SBX+Polynomial": dict(
                        crossover="simulated_binary:eta=20",
                        mutation="polynomial:mutation_rate=0.1,eta=20",
                    ),
                    "Evosax OpenES": dict(
                        backend="evosax",
                        evosax_strategy="OpenES",
                    ),
                },
                fitness="sphere:dim=10",           # shared across all
                pop_size=100,                     # shared
                generations=200,                  # shared
                seeds=(42, 43, 44, 45),          # shared
            )
            print(comparison.summary_table())
            comparison.plot_convergence(seed_index=0)

        Fair initialization (shared first population)::

            comparison = composer.compare(
                pipelines={...},
                shared_initial_population=True,   # use same init pop
                pop_seed=999,                     # populate seed
                fitness="rastrigin:dim=10",
                pop_size=50,
            )
            # All pipelines start from identical population
        """

        trace_base = shared_kwargs.pop("trace_dir", None)

        results: Dict[str, ExperimentResult] = {}
        negate_map: Dict[str, bool] = {}
        last_init_pop = None

        iterable = pipelines.items()
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(pipelines), desc="pipelines")

        for name, pipeline_kwargs in iterable:
            merged = {**shared_kwargs, **pipeline_kwargs}
            merged["seeds"] = seeds
            merged.setdefault("experiment_name", name)

            if trace_base is not None:
                merged["trace_dir"] = Path(trace_base) / name

            if shared_initial_population and "initial_population" not in merged:
                pipeline_init_pop = self._generate_initial_population(merged, pop_seed)
                p_genome_length = infer_genome_length(merged)

                if hasattr(pipeline_init_pop, "shape"):
                    if int(pipeline_init_pop.shape[1]) != p_genome_length:
                        raise ValueError(
                            "Shared initial population dimension mismatch: "
                            f"expected {p_genome_length}, got {pipeline_init_pop.shape[1]}"
                        )
                merged["initial_population"] = pipeline_init_pop
                last_init_pop = pipeline_init_pop

            results[name] = self.quick_run(**merged)
            maximize_flag = bool(merged.get("maximize", False))

            # Normalize displayed metrics/history to the canonical
            # "lower-is-better" convention used by ComparisonResult.
            #
            # Current adapters for both MalthusJAX and Evosax return metrics
            # in the objective's natural direction:
            #   - maximize=False -> lower is better (no sign flip needed)
            #   - maximize=True  -> higher is better (flip sign for display)
            #
            # avoid regressions when backend internals change.
            negate_map[name] = maximize_flag

            # Clear JAX compilation caches to prevent RESOURCE_EXHAUSTED OOMs
            # on large sweeps (e.g., 100+ pipelines)

            jax.clear_caches()

        return ComparisonResult(
            pipelines=results,
            shared_config=dict(shared_kwargs),
            initial_population=last_init_pop,
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
        """Load and execute a declarative TOML experiment specification.

        This classmethod provides a reproducible, version-controllable way to
        define and run experiments. TOML files specify shared configuration,
        pipeline-specific overrides, and seeds in a human-readable format.
        The method parses the file, runs all pipelines with fair comparison
        settings, and returns a :class:`ComparisonResult` with visualization
        and summary utilities.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML experiment configuration file. The file must contain
            at least ``[experiment.shared]`` section with default settings and
            one or more ``[pipelines.NAME]`` sections defining algorithm variants.
            See Notes for full TOML schema.

        pipelines : List[str], optional
            If provided, only execute pipelines with these names. If ``None``,
            all pipelines defined in the TOML file are executed.
            Default: ``None``.

        shared_initial_population : bool, optional
            If ``True``, generate a single random population and inject it into
            each pipeline's first run for fair initialization. Default: ``True``.

        pop_seed : int, optional
            Random seed for generating the shared initial population.
            Default: 123.

        trace_dir : str or pathlib.Path, optional
            Directory for Perfetto JAX profiler traces. If provided, traces are
            written to ``trace_dir/<pipeline_name>/`` for the first seed of each
            run. Default: ``None`` (no traces).

        Returns
        -------
        ComparisonResult
            Multi-pipeline result object. Contains:
            - ``.summary_table()`` : aggregated metrics per pipeline
            - ``.plot_convergence()`` : matplotlib visualization
            - ``.convergence_data()`` : raw history data
            - ``.pipelines`` : dict of per-pipeline :class:`ExperimentResult`

        Raises
        ------
        FileNotFoundError
            If the specified TOML file does not exist.
        ValueError
            If TOML structure is invalid or required keys are missing.
        KeyError
            If a pipeline references undefined operators or configs.

        Notes
        -----
        **TOML File Structure**:

        The TOML file must follow this schema::

            [experiment]
            name = "my_experiment"           # optional

            [experiment.shared]
            fitness = "sphere:dim=10"
            selection = "tournament:num_selections=25,tournament_size=3"
            pop_size = 50
            generations = 100
            seeds = [42, 43, 44]             # list of seeds
            bounds = [-5.0, 5.0]             # optional
            maximize = false                 # optional

            [pipelines.blend_gaussian]
            crossover = "blend:alpha=0.5"
            mutation = "gaussian:mutation_rate=0.5,mutation_strength=0.1"

            [pipelines.sbx_polynomial]
            crossover = "simulated_binary:eta=20"
            mutation = "polynomial:mutation_rate=0.1,eta=20"

            [pipelines.evosax_cmaes]
            backend = "evosax"
            evosax_strategy = "CMA_ES"

        **Operator Spec Format**:

        All operator specs (fitness, selection, crossover, mutation) use the
        format: ``"operator_name:param1=val1,param2=val2"``.

        Common parameters:
        - Fitness dim/num_dims/fn: Problem dimension or function ID
        - Selection num_selections: Number of individuals to select
        - Mutation/crossover rates: Probability values (0.0-1.0)

        **Inheritance and Override**:

        Each pipeline inherits all keys from ``[experiment.shared]`` and can
        override them. Pipeline-specific values take precedence.

        Examples
        --------
        Simple TOML-based experiment::

            # experiment.toml content:
            # [experiment.shared]
            # fitness = "sphere:dim=10"
            # pop_size = 100
            # generations = 200
            # seeds = [42, 43, 44]
            # [pipelines.ga_blend]
            # crossover = "blend:alpha=0.5"
            # [pipelines.ga_sbx]
            # crossover = "simulated_binary:eta=20"

            result = Composer.from_toml("experiment.toml")
            print(result.summary_table())
            result.plot_convergence()

        Execute selected pipelines::

            result = Composer.from_toml(
                "experiment.toml",
                pipelines=["ga_blend", "ga_sbx"],  # skip others
                trace_dir="results/traces",
            )

        Reproducible benchmark (same TOML used in paper)::

            # experiment.toml is versioned in git
            result = Composer.from_toml("experiment.toml")
            result.summary_table()  # Reproducible: same config -> same results
        """
        config_res = load_experiment_config(str(path), pipelines=pipelines)
        experiment_meta = config_res.meta
        resolved = config_res.pipelines
        data_registry = config_res.data_registry
        shared = experiment_meta.get("shared", {})

        output_dir = experiment_meta.get("output_dir")
        if output_dir:
            shared.setdefault("output_dir", output_dir)

        seeds = normalize_seeds(shared.pop("seeds", (42, 43, 44)))

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
            data_config=data_registry,
            **shared,
        )


    @classmethod
    def create_default(cls) -> "Composer":
        """Create a :class:`Composer` instance with default configuration.

        This is a convenience factory method that initializes a :class:`Composer`
        with empty registry and minimal default config. Equivalent to calling
        ``Composer(config={"version": "0.1"})``. Most users should call this
        before invoking :meth:`quick_run`, :meth:`compare`, or :meth:`from_toml`.

        Returns
        -------
        Composer
            A ready-to-use :class:`Composer` instance with defaults applied.

        Examples
        --------
        Standard initialization::

            composer = Composer.create_default()
            result = composer.quick_run(
                fitness="sphere:dim=10",
                pop_size=50,
                generations=100,
            )
        """
        return cls(config={"version": "0.1"})
