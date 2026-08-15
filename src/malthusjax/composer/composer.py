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

from malthusjax.composer.strategies.base import BaseStrategy
from malthusjax.composer.strategies.core import (
    EvoSAXStrategy,
    GeneticStrategy,
    MapElitesStrategy,
    QDAXStrategy,
    TensorNEATStrategy,
)
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


class MapElitesEngineAdapter:
    """Adapter to make Native MapElitesEngine compatible with BenchmarkRunner."""

    def __init__(
        self, engine, pop_size, maximize, history_metrics, initial_population=None, centroids=None
    ):
        self.engine = engine
        self.pop_size = pop_size
        self.maximize = maximize
        self.history_metrics = history_metrics
        self.initial_population = initial_population
        self.centroids = centroids

    def run_once(self, key):
        import time

        import jax

        # Check if we are running the exact QDAX replica for bit-parity
        is_qdax_replica = (
            getattr(self.engine, "engine_params", None)
            and getattr(self.engine.engine_params, "key_derivation", None) == "qdax_replica"
        )

        if is_qdax_replica:
            # Match UniversalAdapterEngine.run_once exact key flow:
            # key, key_init, key_eval = split(key, 3)
            _, k_init_qdax, _ = jax.random.split(key, 3)
            # The QDAX adapter stores key_init directly as its randkey in state.
            # But MapElitesEngine.init_state does k1, k2 = split(rng_key).
            # We want rng_key inside the state to be EXACTLY k_init_qdax.
            # We'll pass a dummy key and overwrite it after.
            k_init, k_run = jax.random.split(key)
        else:
            k_init, k_run = jax.random.split(key)

        init_pop = self.initial_population
        if init_pop is None:
            if hasattr(self.engine.emitter, "genome_config"):
                init_pop = self.engine.emitter.genome_config.init_population(k_init, self.pop_size)
            elif (
                hasattr(self.engine.emitter, "genome")
                and "TensorNeat" in self.engine.emitter.__class__.__name__
            ):
                import jax.numpy as jnp

                from malthusjax.core.genome.tensorneat_genome import (
                    TensorNeatGenome,
                    TensorNeatPopulation,
                )

                try:
                    from tensorneat.common import State
                except ImportError:
                    State = Any
                tn_state = State(randkey=k_init, generation=jnp.float32(0))
                # TensorNEAT initialize creates a single genome. We vmap it to create a population.
                import jax

                pop_keys = jax.random.split(k_init, self.pop_size)
                nodes, conns = jax.vmap(self.engine.emitter.genome.initialize, in_axes=(None, 0))(
                    tn_state, pop_keys
                )
                init_pop = TensorNeatPopulation(
                    genes=TensorNeatGenome(values=(nodes, conns)),
                    fitness=jnp.full((self.pop_size,), -jnp.inf),
                    config=None,
                    info={},
                )
            else:
                raise AttributeError(
                    "Emitter lacks genome_config or genome to generate initial population."
                )
        elif isinstance(init_pop, jax.numpy.ndarray):
            # Wrap the shared array into the correct Population PyTree
            init_pop_copy = jax.numpy.array(init_pop, copy=True)
            if hasattr(self.engine.emitter, "genome_config"):
                dummy_pop = self.engine.emitter.genome_config.init_population(k_init, self.pop_size)
                if hasattr(dummy_pop.genes, "replace"):
                    new_genes = dummy_pop.genes.replace(values=init_pop_copy)
                    init_pop = dummy_pop.replace(genes=new_genes)
                else:
                    init_pop = dummy_pop.replace(genes=init_pop_copy)
            else:
                # If we passed an ndarray for TensorNEAT, it is not well supported, so fail gracefully
                raise NotImplementedError(
                    "Passing ndarray init_pop to TensorNeatEmitter is not supported yet."
                )

        # Copy centroids to prevent "Buffer has been deleted or donated" error across seeds
        centroids_copy = (
            jax.numpy.array(self.centroids, copy=True) if self.centroids is not None else None
        )
        state = self.engine.init_state(k_run, init_pop, centroids_copy)

        if is_qdax_replica:
            # Force the exact QDAX key into the state for generation 1
            state = state.replace(rng_key=k_init_qdax)

        t_exec_start = time.perf_counter()
        final_state, scan_history, _ = self.engine.run(state, time_it=True, compile=True)
        t_exec_end = time.perf_counter()

        num_gens = int(self.engine.engine_params.num_generations)
        history = []
        track_keys = self.history_metrics or ["best_fitness", "qd_score", "coverage"]

        for g in range(num_gens):
            gen_stats = {"generation": g + 1}
            for k in track_keys:
                if hasattr(scan_history, k):
                    val = getattr(scan_history, k)[g]
                    gen_stats[k] = float(val)
            history.append(gen_stats)

        # Safely extract qd_score and coverage. If they exist on final_state use them,
        # otherwise try to get the last element from scan_history.
        qd_score = getattr(final_state, "qd_score", None)
        if qd_score is None and hasattr(scan_history, "qd_score"):
            qd_score = scan_history.qd_score[-1]

        coverage = getattr(final_state, "coverage", None)
        if coverage is None and hasattr(scan_history, "coverage"):
            coverage = scan_history.coverage[-1]

        summary = {
            "best_fitness": float(final_state.best_fitness),
            "qd_score": float(qd_score) if qd_score is not None else 0.0,
            "coverage": float(coverage) if coverage is not None else 0.0,
            "final_generation": int(final_state.generation),
            "total_evaluations": int(final_state.generation * self.pop_size),
        }

        return {
            "history": history,
            "summary": summary,
            "timings": {"total": t_exec_end - t_exec_start},
        }


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
                        emitter=kwargs.get("map_elites_emitter", "mixing"),
                        num_descriptors=qdax_num_descriptors,
                        num_centroids=qdax_num_centroids,
                    )
                elif self._has_real_operators(genome, fitness, selection, crossover, mutation):
                    strategy = GeneticStrategy(
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                    )

            if isinstance(strategy, EvoSAXStrategy):
                engine = self._build_evosax_engine(
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
                engine = self._build_qdax_engine(
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
                engine = self._build_tensorneat_engine(
                    strategy=strategy,
                    fitness_spec=fitness,
                    pop_size=pop_size,
                    generations=generations,
                    maximize=maximize,
                    history_metrics=history_metrics,
                    **kwargs,
                )
            elif isinstance(strategy, MapElitesStrategy):
                engine = self._build_map_elites_engine(
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
                engine = self._build_real_engine(
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
                engine = self._build_stub_engine(generations, **kwargs)

        runner = BenchmarkRunner(
            engine=engine,
            experiment_name=experiment_name,
            output_dir=output_dir,
            write_artifacts=True,
            prng_impl=prng_impl,
            trace_dir=Path(trace_dir) if trace_dir is not None else Path("results/traces"),
            serialize_history=kwargs.get("serialize_history", True),
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

    def _postprocess_experiment_final_from_history(self, experiment: Any, force: bool = False) -> None:
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

    def _infer_genome_length(self, cfg: Dict[str, Any]) -> int:
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
            dim_val = parsed_params.get("dim", parsed_params.get("num_dims"))
            if dim_val is not None:
                return int(dim_val)

        return 10

    def _generate_initial_population(self, config: Dict[str, Any], pop_seed: int) -> Any:
        """Deterministically generate a shared initial population matrix for a given pipeline config.

        Ensures that pipelines with identical bounds, population sizes, and dimensionality
        receive the exact same starting points, while dynamically scaling to the requested pop_size.
        """
        import jax
        import jax.random as jr

        pop_size = int(config.get("pop_size", 50))
        genome_length = self._infer_genome_length(config)
        bounds = config.get("bounds", (-5.0, 5.0))
        fitness_spec = config.get("fitness")

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
                p_genome_length = self._infer_genome_length(merged)

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
            import jax

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
            data_config=data_registry,
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
        strategy: BaseStrategy,
        fitness: Optional[str],
        engine_type: str = "ga",
        data_config: Optional[Dict[str, Any]] = None,
        **config: Any,
    ) -> Any:
        """Build engine from operator specs and config via EngineRegistry."""
        from malthusjax.composer.strategies.core import GeneticStrategy
        from malthusjax.engine.schedules import TrackBest

        catalog = OperatorCatalog()

        data_registry = None
        if data_config:
            data_registry = self._build_data_registry(data_config)

        seed_val = config.get("seed", 42)
        maximize_flag = config.get("maximize", False)

        # We append seed to fitness strings if missing so BBOB etc uses the right seed
        if isinstance(fitness, str):
            if "seed=" not in fitness:
                if ":" in fitness:
                    fitness = f"{fitness},seed={seed_val}"
                else:
                    fitness = f"{fitness}:seed={seed_val}"
            resolved_evaluator = catalog.get(
                fitness,
                data_registry=data_registry,
            )
        elif fitness is not None:
            resolved_evaluator = fitness
        else:
            resolved_evaluator = catalog.get(
                f"sphere:dim=10,maximize={maximize_flag},seed={seed_val}",
                data_registry=data_registry,
            )

        if isinstance(strategy, GeneticStrategy):
            resolved_selection = (
                catalog.get(
                    strategy.selection
                    or f"tournament:num_selections={config.get('pop_size', 50) // 2},tournament_size=3",
                    data_registry=data_registry,
                )
                if isinstance(strategy.selection, str) or strategy.selection is None
                else strategy.selection
            )

            resolved_crossover = (
                catalog.get(strategy.crossover or "blend:alpha=0.5", data_registry=data_registry)
                if isinstance(strategy.crossover, str) or strategy.crossover is None
                else strategy.crossover
            )

            resolved_mutation = (
                catalog.get(
                    strategy.mutation or "gaussian:mutation_rate=0.1", data_registry=data_registry
                )
                if isinstance(strategy.mutation, str) or strategy.mutation is None
                else strategy.mutation
            )
        else:
            resolved_selection = None
            resolved_crossover = None
            resolved_mutation = None

        # Ensure we use LIGHT tracking for monotonic convergence curves
        if "track_best" not in config:
            config["track_best"] = TrackBest.LIGHT

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
            if isinstance(fitness_spec, str):
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

                evalr = BBOBEvaluator.create(
                    BBOBConfig(fn_name=fn, num_dims=dims, seed=seed, maximize=maxim)
                )
            else:
                evalr = fitness_spec
                maxim = (
                    getattr(evalr.config, "maximize", maximize)
                    if hasattr(evalr, "config")
                    else maximize
                )
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
            num_dims=num_dims,
            bounds=bounds,
            maximize=maxim,
            strategy_params=kwargs.get("strategy_params"),
            initial_population=init_pop,
            prng_impl=prng_impl,
        )

    def _build_qdax_engine(
        self,
        strategy: QDAXStrategy,
        fitness_spec: Optional[str],
        pop_size: int,
        generations: int,
        genome_length: int,
        bounds: Tuple[float, float],
        maximize: bool,
        history_metrics: Optional[Sequence[str]],
        **kwargs: Any,
    ) -> Any:
        import functools

        from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
        from qdax.utils.metrics import default_qd_metrics

        from .qdax_adapter import build_qdax_engine

        # 1. Resolve strategy class
        strategy_cls = strategy.strategy_cls
        if isinstance(strategy_cls, str):
            import importlib
            import pkgutil

            import qdax.core

            resolved_cls = None
            for _, name, is_pkg in pkgutil.iter_modules(qdax.core.__path__):
                if not is_pkg:
                    mod = importlib.import_module(f"qdax.core.{name}")
                    if hasattr(mod, strategy_cls):
                        resolved_cls = getattr(mod, strategy_cls)
                        break

            if resolved_cls is None:
                raise ValueError(f"Unknown QDAX strategy: {strategy_cls}")
            strategy_cls = resolved_cls

        # 2. Auto-create emitter if not provided
        emitter = strategy.emitter
        if isinstance(emitter, str):
            if emitter.lower() == "mixing":
                sigma = strategy.mutation_sigma
                var_pct = kwargs.pop("qdax_variation_percentage", 0.5)
                emitter_spec = f"qdax_native:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},variation_percentage={var_pct}"
            else:
                if ":" in emitter:
                    emitter_spec = f"{emitter},batch_size={pop_size}"
                else:
                    emitter_spec = f"{emitter}:batch_size={pop_size}"

            from malthusjax.composer.catalog import OperatorCatalog

            catalog = OperatorCatalog()
            emitter = catalog.get(emitter_spec)
        elif emitter is None:
            sigma = strategy.mutation_sigma
            var_pct = kwargs.pop("qdax_variation_percentage", 0.5)
            emitter_spec = f"qdax_native:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},variation_percentage={var_pct}"
            from malthusjax.composer.catalog import OperatorCatalog

            catalog = OperatorCatalog()
            emitter = catalog.get(emitter_spec)

        # 3. Auto-create metrics function if not provided
        metrics_fn = strategy.metrics_function
        if metrics_fn is None:
            metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)

        # 4. Auto-compute centroids if not provided
        centroids = strategy.centroids
        if centroids is None:
            centroids = compute_cvt_centroids(
                num_descriptors=strategy.num_descriptors,
                num_init_cvt_samples=50000,
                num_centroids=strategy.num_centroids,
                minval=0.0,
                maxval=1.0,
                key=jr.PRNGKey(42),
            )

        # 5. Auto-generate init_variables if not provided
        # Note: We do NOT generate it here statically with PRNGKey(0) anymore.
        # It is handled dynamically by QDaxEngineAdapter._adapter_init using the runtime seed
        # or it is overridden by the shared_initial_population feature.
        init_variables = strategy.init_variables

        # 6. Resolve evaluator
        evaluator = self._resolve_qdax_evaluator(fitness_spec, bounds, maximize)

        return build_qdax_engine(
            strategy_cls=strategy_cls,
            emitter=emitter,
            metrics_function=metrics_fn,
            evaluator=evaluator,
            init_variables=init_variables,
            centroids=centroids,
            pop_size=pop_size,
            generations=generations,
            maximize=maximize,
            eval_mode="native",
            history_metrics=history_metrics or ["qd_score", "coverage"],
            bounds=bounds,
            genome_length=genome_length,
            **kwargs,
        )

    def _resolve_qdax_evaluator(
        self, fitness_spec: Optional[str], bounds: Tuple[float, float], maximize: bool
    ) -> Any:
        from malthusjax.core.genome.real_genome import RealGenome, RealPopulation

        class QDAXNativeEvaluator:
            def __init__(self, fitness_fn, evaluator=None, num_descriptors=2):
                self._fitness_fn = fitness_fn
                self._evaluator = evaluator
                self._num_descriptors = num_descriptors

            def scoring_function(self, genotypes, random_key):
                import jax
                import jax.numpy as jnp

                if self._evaluator is not None:
                    genes = RealGenome(values=genotypes)
                    pop = RealPopulation(
                        genes=genes, fitness=jnp.zeros(genotypes.shape[0]), config=None
                    )
                    updated_pop = self._evaluator.evaluate_population(pop)
                    fitnesses = updated_pop.fitness
                    if not maximize:
                        fitnesses = -fitnesses
                    if hasattr(updated_pop, "descriptors"):
                        descriptors = updated_pop.descriptors
                    else:
                        desc_dims = genotypes[:, : self._num_descriptors]
                        lo, hi = float(bounds[0]), float(bounds[1])
                        descriptors = (desc_dims - lo) / (hi - lo)
                else:
                    fitnesses = jax.vmap(self._fitness_fn)(genotypes)
                    desc_dims = genotypes[:, : self._num_descriptors]
                    lo, hi = float(bounds[0]), float(bounds[1])
                    descriptors = (desc_dims - lo) / (hi - lo)

                return fitnesses, descriptors, {}

        if isinstance(fitness_spec, str):
            cat = OperatorCatalog()
            resolved = cat.get(fitness_spec)
            return QDAXNativeEvaluator(None, evaluator=resolved)
        elif fitness_spec is not None:
            return QDAXNativeEvaluator(None, evaluator=fitness_spec)
        else:
            import jax.numpy as jnp

            def fn(x):
                return -jnp.sum(jnp.square(x))

            return QDAXNativeEvaluator(fn)

    def _build_tensorneat_engine(
        self,
        strategy: TensorNEATStrategy,
        fitness_spec: Optional[str],
        pop_size: int,
        generations: int,
        maximize: bool,
        history_metrics: Optional[Sequence[str]],
        **kwargs: Any,
    ) -> Any:
        import inspect

        # 1. Resolve algorithm
        import tensorneat.algorithm

        from .tensorneat_adapter import build_tensorneat_engine

        algorithm_cls = None
        for name, obj in inspect.getmembers(tensorneat.algorithm, inspect.isclass):
            if name.lower() == strategy.algorithm_name.lower():
                algorithm_cls = obj
                break

        if algorithm_cls is None:
            raise ValueError(f"Unknown TensorNEAT algorithm: {strategy.algorithm_name}")

        # 2. Resolve genome
        import tensorneat.genome

        genome_cls = None

        # In TensorNEAT, genome classes often end with "Genome" (e.g. DefaultGenome, RecurrentGenome)
        # So if user passes "default", we check for "default" or "defaultgenome"
        target_genome = strategy.genome_name.lower()
        for name, obj in inspect.getmembers(tensorneat.genome, inspect.isclass):
            name_lower = name.lower()
            if name_lower == target_genome or name_lower == f"{target_genome}genome":
                genome_cls = obj
                break

        if genome_cls is None:
            raise ValueError(f"Unknown TensorNEAT genome: {strategy.genome_name}")

        genome = genome_cls(num_inputs=strategy.num_inputs, num_outputs=strategy.num_outputs)
        algorithm = algorithm_cls(pop_size=pop_size, genome=genome, **strategy.algorithm_kwargs)

        problem, problem_state = self._resolve_tensorneat_problem(
            strategy.problem_name, fitness_spec
        )

        return build_tensorneat_engine(
            algorithm=algorithm,
            evaluator=(problem, problem_state),
            generations=generations,
            pop_size=pop_size,
            maximize=maximize,
            history_metrics=history_metrics,
        )

    def _resolve_tensorneat_problem(
        self, problem_name: Optional[str], fitness_spec: Optional[str]
    ) -> Tuple[Any, Any]:
        import inspect

        import tensorneat.problem

        name = problem_name or (fitness_spec if isinstance(fitness_spec, str) else "xor")
        base_name = name.split(":")[0].lower()

        problem_cls = None
        for cls_name, cls_obj in inspect.getmembers(tensorneat.problem, inspect.isclass):
            if cls_name.lower() == base_name:
                problem_cls = cls_obj
                break

        if problem_cls is None:
            raise ValueError(f"Unknown TensorNEAT problem: {base_name}.")

        problem = problem_cls()
        return problem, problem.setup()

    def _build_map_elites_engine(
        self,
        strategy: MapElitesStrategy,
        fitness_spec: Optional[str],
        pop_size: int,
        generations: int,
        maximize: bool,
        history_metrics: Optional[Sequence[str]],
        **kwargs: Any,
    ) -> Any:
        import jax.random as jr
        from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

        # Resolve Evaluator using EngineRegistry or direct resolution (for MAP-Elites parity)
        from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
        from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter

        evaluator: Any = None
        if isinstance(strategy.emitter, TensorNeatEmitter):
            if fitness_spec is not None:
                evaluator = fitness_spec
            else:
                from malthusjax.core.fitness.base import BaseEvaluatorConfig
                from malthusjax.core.fitness.tensorneat import TensorNeatQDEvaluator

                objective_fn = kwargs.get("objective_function")
                evaluator = TensorNeatQDEvaluator(
                    objective_function=objective_fn,  # type: ignore[arg-type]
                    config=BaseEvaluatorConfig(maximize=maximize),
                    data=None,
                )
        else:
            # We use the BaseQDEvaluator composition to match standard evaluation
            from malthusjax.composer.catalog import OperatorCatalog
            from malthusjax.core.fitness.base import BaseEvaluatorConfig
            from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
            from malthusjax.core.genome.qd.population import QDPopulation

            cat = OperatorCatalog()
            seed_val = kwargs.get("seed", 42)
            resolved_base_evaluator: Any
            if isinstance(fitness_spec, str):
                if "seed=" not in fitness_spec:
                    fitness_spec = (
                        f"{fitness_spec}:seed={seed_val}"
                        if ":" not in fitness_spec
                        else f"{fitness_spec},seed={seed_val}"
                    )
                resolved_base_evaluator = cat.get(fitness_spec)
            elif fitness_spec is not None:
                resolved_base_evaluator = fitness_spec
            else:
                resolved_base_evaluator = cat.get(
                    f"sphere:dim={kwargs.get('genome_length', 10)},maximize={maximize},seed={seed_val}"
                )

            bounds = kwargs.get("bounds", (-5.0, 5.0))
            num_desc = strategy.num_descriptors

            class ComposedQDEvaluator(BaseQDEvaluator[Any, Any, Any]):
                def evaluate_qd(self, genome):
                    raise NotImplementedError("Use evaluate_population directly")

                def evaluate_population(self, population):
                    updated_pop = resolved_base_evaluator.evaluate_population(population)
                    genotypes = getattr(population.genes, "values", population.genes)
                    desc_dims = genotypes[:, :num_desc]
                    lo, hi = float(bounds[0]), float(bounds[1])
                    descriptors = (desc_dims - lo) / (hi - lo)
                    new_info = dict(population.info) if population.info else {}
                    new_info["descriptors"] = descriptors

                    fitness = updated_pop.fitness

                    return QDPopulation(
                        genes=population.genes,
                        fitness=fitness,
                        config=population.config,
                        info=new_info,
                    )

            evaluator = ComposedQDEvaluator(
                config=BaseEvaluatorConfig(maximize=maximize), data=None
            )

        emitter = strategy.emitter
        if isinstance(emitter, str):
            bounds = kwargs.get("bounds", (-5.0, 5.0))
            genome_length = kwargs.get("genome_length", 10)

            if emitter.lower() == "mixing":
                sigma = getattr(strategy, "mutation_sigma", 0.1)
                emitter_spec = f"qdax_replica:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},genome_length={genome_length}"
            else:
                if ":" in emitter:
                    emitter_spec = f"{emitter},batch_size={pop_size},genome_length={genome_length}"
                else:
                    emitter_spec = f"{emitter}:batch_size={pop_size},genome_length={genome_length}"

            from malthusjax.composer.catalog import OperatorCatalog

            catalog = OperatorCatalog()
            emitter = catalog.get(emitter_spec)
        elif emitter is None:
            raise ValueError("MapElitesStrategy requires an explicit emitter.")

        centroids = strategy.centroids
        if centroids is None:
            centroids = compute_cvt_centroids(
                num_descriptors=strategy.num_descriptors,
                num_init_cvt_samples=50000,
                num_centroids=strategy.num_centroids,
                minval=0.0,
                maxval=1.0,
                key=jr.PRNGKey(42),
            )

        engine: Any = MapElitesEngine(
            emitter=emitter,
            evaluator=evaluator,
            engine_params=MapElitesEngineParams(pop_size=pop_size, num_generations=generations),
        )
        return MapElitesEngineAdapter(
            engine=engine,
            pop_size=pop_size,
            maximize=maximize,
            history_metrics=history_metrics,
            initial_population=kwargs.get("initial_population", None),
            centroids=centroids,
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
