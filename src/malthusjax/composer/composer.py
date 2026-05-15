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

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

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

    @staticmethod
    def _normalize_seeds(seeds: Sequence[int] | int) -> Tuple[int, ...]:
        """Normalize seed input into an explicit tuple of integers.

        Accepts either:
        - an iterable of seeds (e.g., ``[42, 43, 44]``), or
        - an integer count (e.g., ``100`` -> ``(1, 2, ..., 100)``).
        """
        if isinstance(seeds, int):
            if seeds <= 0:
                raise ValueError("seeds must be > 0 when provided as an integer count")
            return tuple(range(1, seeds + 1))

        seeds_tuple = tuple(int(s) for s in seeds)
        if not seeds_tuple:
            raise ValueError("seeds must not be empty")
        return seeds_tuple

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
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run a full evolutionary experiment with programmatic operator specs.

        This is the main entry point for interactive exploration and benchmarking.
        It accepts operator specifications as string specs (for the ``"malthusjax""
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
            elif self._has_real_operators(genome, fitness, selection, crossover, mutation):
                engine = self._build_real_engine(
                    genome=genome,
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
                    maximize=maximize,
                    prng_impl=prng_impl,
                    data_config=data_config,
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
            trace_dir=Path(trace_dir) if trace_dir is not None else Path("results/traces"),
        )

        normalized_seeds = self._normalize_seeds(seeds)
        experiment = runner.run(normalized_seeds)

        # Composer-level postprocessing: when engines run with
        # TrackBest.NONE for speed they may omit or produce an
        # invalid `summary['best_fitness']`. Allow callers to request
        # that the final best is derived from the last history entry
        # (history[-1]) instead. We also auto-fix non-finite summaries.
        self._postprocess_experiment_final_from_history(experiment, use_history_for_final)

        return experiment

    def _postprocess_experiment_final_from_history(self, experiment, force: bool = False) -> None:
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
                or (isinstance(best_val, (int, float)) and (math.isnan(best_val) or math.isinf(best_val)))
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

        **shared_kwargs : Any
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
        def _infer_genome_length(cfg: Dict[str, Any]) -> int:
            """Infer genome length from config, preferring explicit values.

            Priority:
            1) ``genome_length`` kwarg
            2) ``fitness`` spec params ``dim`` / ``num_dims``
            3) default 10
            """
            if "genome_length" in cfg and cfg["genome_length"] is not None:
                return int(cfg["genome_length"])

            fitness_spec = cfg.get("fitness")
            if isinstance(fitness_spec, str):
                parsed_name, parsed_params = OperatorCatalog().parse_spec(fitness_spec)
                _ = parsed_name  # parsed_name is intentionally unused here
                dim_val = parsed_params.get("dim", parsed_params.get("num_dims"))
                if dim_val is not None:
                    return int(dim_val)

            return 10

        init_pop = None
        if shared_initial_population:
            pop_size = int(shared_kwargs.get("pop_size", 50))
            genome_length = _infer_genome_length(shared_kwargs)
            bounds = shared_kwargs.get("bounds", (-5.0, 5.0))

            # Prefer a shared fitness spec, but fall back to the first pipeline's
            # fitness when parity TOMLs keep fitness under each pipeline section.
            fitness_spec = shared_kwargs.get("fitness")
            if fitness_spec is None and pipelines:
                first_pipeline = next(iter(pipelines.values()))
                fitness_spec = first_pipeline.get("fitness")

            # Check if using BBOB - if so, use its sample() method for consistency
            if fitness_spec and isinstance(fitness_spec, str) and "bbob" in fitness_spec.lower():
                # Parse BBOB spec to create evaluator for sampling
                cat = OperatorCatalog()
                parsed_name, parsed_params = cat.parse_spec(fitness_spec)
                if parsed_name == "bbob":
                    fn = parsed_params.get("fn_name", parsed_params.get("fn", "rosenbrock"))
                    dims = parsed_params.get("dim", parsed_params.get("num_dims", genome_length))
                    # Use BBOB seed from fitness spec, not from population seed
                    bbob_seed = parsed_params.get("seed", 0)
                    bbob_eval = BBOBEvaluator.create(
                        BBOBConfig(fn_name=fn, num_dims=dims, seed=bbob_seed, maximize=shared_kwargs.get("maximize", False))
                    )
                    # Generate initial population using BBOB's sample method
                    pop_key = jr.PRNGKey(pop_seed)
                    sample_keys = jr.split(pop_key, pop_size)
                    init_pop = jax.vmap(bbob_eval.evosax_problem.sample)(sample_keys)

            # Fall back to uniform sampling if not BBOB
            if init_pop is None:
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
                expected_dim = _infer_genome_length(merged)
                if int(init_pop.shape[1]) != expected_dim:
                    raise ValueError(
                        "Shared initial population dimension mismatch: "
                        f"init_pop has dim={int(init_pop.shape[1])} but pipeline '{name}' "
                        f"expects dim={expected_dim}. "
                        "Ensure all pipelines share the same dimensionality when "
                        "shared_initial_population=True, or pass per-pipeline "
                        "initial_population explicitly."
                    )
                merged["initial_population"] = init_pop

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
            # Keep normalization objective-driven (not backend-driven) to
            # avoid regressions when backend internals change.
            negate_map[name] = maximize_flag

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
        experiment_meta, resolved = load_experiment_config(str(path), pipelines=pipelines)
        shared = experiment_meta.get("shared", {})

        seeds = cls._normalize_seeds(shared.pop("seeds", (42, 43, 44)))

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
        genome: Optional[str],
        fitness: Optional[str],
        selection: Optional[str],
        crossover: Optional[str],
        mutation: Optional[str],
    ) -> bool:
        """Check if any real operator specs are provided."""
        return any([genome, fitness, selection, crossover, mutation])

    def _build_data_registry(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a configuration of data sources into a resolved data registry."""
        from malthusjax.benchmarking.registry import DataRegistry

        reg = DataRegistry()
        for data_id, data_spec in data_config.items():
            reg.register(data_id, data_spec)

        resolved: Dict[str, Any] = {}
        for data_id in data_config.keys():
            resolved[data_id] = reg.resolve(data_id)
        return resolved

    def _build_real_engine(
        self,
        fitness: Optional[str],
        selection: Optional[str],
        crossover: Optional[str],
        mutation: Optional[str],
        engine_type: str = "ga",
        data_config: Optional[Dict[str, Any]] = None,
        **config: Any,
    ) -> Any:
        """Build engine from operator specs and config via EngineRegistry."""
        from malthusjax.engine.schedules import TrackBest

        catalog = OperatorCatalog()

        data_registry = self._build_data_registry(data_config) if data_config else None

        maximize_flag = config.get('maximize', False)
        if fitness and "maximize=" not in fitness:
            # Append maximize param correctly: if fitness has params (contains :),
            # use comma; if it's just a name, use colon to start params
            if ":" in fitness:
                fitness = f"{fitness},maximize={maximize_flag}"
            else:
                fitness = f"{fitness}:maximize={maximize_flag}"

        resolved_evaluator = catalog.get(
            fitness or f"sphere:dim=10,maximize={maximize_flag}",
            data_registry=data_registry,
        )
        resolved_selection = catalog.get(
            selection
            or f"tournament:num_selections={config.get('pop_size', 50) // 2},tournament_size=3",
            data_registry=data_registry,
        )
        resolved_crossover = catalog.get(
            crossover or "blend:alpha=0.5", data_registry=data_registry
        )
        resolved_mutation = catalog.get(
            mutation or "gaussian:mutation_rate=0.1", data_registry=data_registry
        )

        # Ensure we use LIGHT tracking for monotonic convergence curves
        if 'track_best' not in config:
            config['track_best'] = TrackBest.LIGHT

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
        if fitness_spec is not None:
            from .catalog import OperatorCatalog

            cat = OperatorCatalog()
            parsed_name, parsed_params = cat.parse_spec(fitness_spec)

            if parsed_name == "bbob":
                fn_param = parsed_params.get("fn_name", parsed_params.get("fn", None))
                if isinstance(fn_param, int):
                    import evosax.problems.bbob.meta_bbob as mb

                    bbob_keys = list(mb.bbob_fns.keys())
                    if fn_param < 1 or fn_param > len(bbob_keys):
                        raise ValueError(
                            f"BBOB function index {fn_param} is out of range (1-{len(bbob_keys)})"
                        )
                    fn = bbob_keys[fn_param - 1]
                elif fn_param is None:
                    raise ValueError(
                        "BBOB fitness specification requires either fn_name or fn index"
                    )
                else:
                    fn = fn_param
            else:
                fn = parsed_params.get("fn_name", parsed_name)

            dims = parsed_params.get("dim", parsed_params.get("num_dims", num_dims))
            seed = parsed_params.get("seed", kwargs.get("seed", 42))
            maxim = parsed_params.get("maximize", maximize)
        else:
            fn = "sphere"
            dims = num_dims
            seed = kwargs.get("seed", 42)
            maxim = maximize

        evalr = BBOBEvaluator.create(
            BBOBConfig(fn_name=fn, num_dims=dims, seed=seed, maximize=maxim)
        )

        # If no initial_population provided and evaluator has sample() method,
        # use it for consistent initialization across backends
        init_pop = kwargs.get("initial_population")
        if init_pop is None and hasattr(evalr, "evosax_problem"):
            # Use same seed as BBOB problem for consistent initialization
            pop_key = jr.PRNGKey(seed)
            sample_keys = jr.split(pop_key, pop_size)
            init_pop = jax.vmap(evalr.evosax_problem.sample)(sample_keys)

        return build_evosax_engine(
            strategy_name=strategy_name,
            evaluator=evalr,
            pop_size=pop_size,
            generations=generations,
            bounds=bounds,
            maximize=maxim,
            strategy_params=kwargs.get("strategy_params"),
            initial_population=init_pop,
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
