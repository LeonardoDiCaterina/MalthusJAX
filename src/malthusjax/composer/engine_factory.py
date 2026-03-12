"""Helpers for constructing engines from string/catalog specifications.

This module contains adapters and factory functions that bridge the
low-level :class:`~malthusjax.engine.GeneticEngine` implementation with the
higher-level Composer infrastructure.  The primary entry points are
``build_engine`` and ``build_engine_from_catalog`` which take resolved
operator instances (or strings) along with configuration parameters and
produce a :class:`GeneticEngineAdapter` conforming to the
:class:`~malthusjax.benchmarking.runner.Engine` protocol.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, Union, cast

import chex
import jax.numpy as jnp

from ..core.genome.binary_genome import BinaryGenomeConfig

# for now it only supports real genomes but it will be extended in the future
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..core.random import resolve_prng_impl
from ..engine.base import _get_evolution_kernel
from ..engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from ..engine.schedules import TrackBest


class GeneticEngineAdapter:
    """Adapter to make GeneticEngine compatible with BenchmarkRunner.Engine protocol.

    Accepts an optional `initial_population` (array-like) to override the
    engine-initialised population for reproducible cross-engine comparisons.
    """

    def __init__(
        self,
        genetic_engine: GeneticEngine,
        genome_config: Any,
        initial_population: Any = None,
        prng_impl: Optional[str] = None,
    ):
        self.genetic_engine = genetic_engine
        self.genome_config = genome_config
        self.initial_population = initial_population
        self.prng_impl = prng_impl

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results.

        The returned dictionary contains three entries:
        ``'history'`` (list of per-generation statistics), ``'summary'``
        (final metrics), and ``'timings'`` (initialization/compile/evolution
        durations).
        """
        t_init_start = time.perf_counter()
        state = self.genetic_engine.init_state(key)
        state.best_fitness.block_until_ready()
        initial_best = float(state.best_fitness)
        if hasattr(self, "maximize") and self.maximize:
            initial_best = -initial_best
        t_init_end = time.perf_counter()

        if self.initial_population is not None:



            arr = jnp.asarray(self.initial_population)
            pop = RealPopulation.from_array(arr, self.genome_config, axis=0)
            evaluated_pop = self.genetic_engine.evaluator.evaluate_population(pop)

            fitness = evaluated_pop.fitness
            best_idx = int(jnp.argmax(fitness))
            best_fitness = fitness[best_idx]
            best_genome = evaluated_pop.genes[best_idx]

            state = cast(Any, state).replace(
                population=evaluated_pop,
                best_genome=best_genome,
                best_fitness=best_fitness,
            )

        # ------------------------------------------------------------------
        # Compile: warmup step + XLA compilation of the scan kernel.
        # The eager step triggers JAX tracing / dispatch warmup, then
        # lower().compile() forces full XLA optimisation.  Both costs are
        # captured in the ``compile`` timing bucket — matching the evosax
        # adapter which runs a 1-iteration warmup scan in the same bucket.
        # ------------------------------------------------------------------
        t_compile_start = time.perf_counter()

        _ws, _ = self.genetic_engine.step(state)
        _ws.best_fitness.block_until_ready()

        ep = self.genetic_engine.engine_params
        jit_fn = _get_evolution_kernel(ep, compile_jit=True, unroll_num=ep.unroll_num)
        _ = jit_fn.lower(self.genetic_engine, state).compile()
        t_compile_end = time.perf_counter()

        # ------------------------------------------------------------------
        # Evolution: run the pre-compiled scan.  Since we already warmed up
        # and compiled above, `compile=True` will hit JAX's JIT cache and
        # execute the cached kernel (no re-compilation).
        # ------------------------------------------------------------------
        t_evo_start = time.perf_counter()
        final_state, scan_history, _ = self.genetic_engine.run(
            state, time_it=True, compile=True
        )
        t_evo_end = time.perf_counter()

        num_gens = int(self.genetic_engine.engine_params.num_generations)
        history = []
        for g in range(num_gens):
            history.append(
                {
                    "generation": g + 1,
                    "best_fitness": float(scan_history.best_fitness[g]),
                    "mean_fitness": float(scan_history.mean_fitness[g]),
                    "std_fitness": float(scan_history.std_fitness[g]),
                }
            )

        total_evals = int(final_state.generation * self.genetic_engine.engine_params.pop_size)
        summary = {
            "initial_fitness": initial_best,
            "best_fitness": float(final_state.best_fitness),
            "final_generation": int(final_state.generation),
            "total_evaluations": total_evals,
        }

        timings = {
            "initialization": t_init_end - t_init_start,
            "compile": t_compile_end - t_compile_start,
            "evolution": t_evo_end - t_evo_start,
        }

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
        }


def build_engine(
    fitness_evaluator: Any,
    selection_op: Any,
    crossover_op: Any,
    mutation_op: Any,
    genome_type: str = "real",
    pop_size: int = 50,
    generations: int = 100,
    elitism: int = 2,
    genome_shape: Tuple[int, ...] = (10,),
    bounds: Tuple[float, float] = (-5.0, 5.0),
    unroll_factor: int = 1,
    **kwargs: Any,
) -> GeneticEngineAdapter:
    """Build a :class:`GeneticEngine` from concrete operator instances.

    The caller must supply a fitness evaluator plus selection, crossover and
    mutation operators. Optional parameters control population size,
    generations, elitism, genome shape/type and real bounds; any additional
    keyword arguments are forwarded to the engine constructor (e.g.
    ``prng_impl`` or strength schedules).  The result is wrapped in a
    :class:`GeneticEngineAdapter` suitable for use with
    :class:`~.benchmarking.BenchmarkRunner`.
    """
    genome_config: Union[RealGenomeConfig, BinaryGenomeConfig]
    if "genome_length" in kwargs:
        genome_shape = (int(kwargs.pop("genome_length")),)

    if genome_type == "real":
        genome_config = RealGenomeConfig(
            shape=genome_shape, bounds=bounds, dtype=kwargs.get("dtype", "float32")
        )
    elif genome_type == "binary":
        genome_config = BinaryGenomeConfig(shape=genome_shape)
    else:
        raise ValueError(f"Unsupported genome type: {genome_type}")

    OperatorCatalog: Any = None
    try:
        from .catalog import OperatorCatalog as _OperatorCatalog
        OperatorCatalog = _OperatorCatalog
    except Exception:
        OperatorCatalog = None

    if isinstance(selection_op, str):
        if OperatorCatalog is None:
            raise TypeError("selection_op provided as string but OperatorCatalog is unavailable")
        selection_op = OperatorCatalog().get(selection_op)

    if isinstance(crossover_op, str):
        if OperatorCatalog is None:
            raise TypeError("crossover_op provided as string but OperatorCatalog is unavailable")
        crossover_op = OperatorCatalog().get(crossover_op)

    if isinstance(mutation_op, str):
        if OperatorCatalog is None:
            raise TypeError("mutation_op provided as string but OperatorCatalog is unavailable")
        mutation_op = OperatorCatalog().get(mutation_op)

    for name, op in [
        ("selection", selection_op),
        ("crossover", crossover_op),
        ("mutation", mutation_op),
    ]:
        if not hasattr(op, "replace") or not callable(getattr(op, "replace")):
            raise TypeError(
                f"Operator '{name}' lacks required 'replace' method (type {type(op)}). "
                "Provide operator instance from OperatorCatalog.get(spec)"
                " or a proper implementation."
            )
    prng_impl_str = kwargs.pop("prng_impl", None)
    prng_extra: Dict[str, Any] = {}
    if prng_impl_str is not None:
        prng_extra["prng_impl"] = resolve_prng_impl(prng_impl_str)

    _schedule_keys = [
        "schedule_type",
        "initial_strength",
        "final_strength",
        "track_best",
    ]
    schedule_extra: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in _schedule_keys
    }

    if "track_best" not in schedule_extra:
        schedule_extra["track_best"] = TrackBest.NONE

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=generations,
        elitism=elitism,
        unroll_num=unroll_factor,
        **prng_extra,
        **schedule_extra,
    )

    genetic_engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=fitness_evaluator,
        selection=selection_op,
        crossover=crossover_op,
        mutation=mutation_op,
        enable_progress_bar=kwargs.get("enable_progress_bar", False),
    )

    initial_population = kwargs.get("initial_population", None)

    return GeneticEngineAdapter(
        genetic_engine,
        genome_config,
        initial_population=initial_population,
        prng_impl=prng_impl_str,
    )


def build_engine_from_catalog(
    catalog_operators: Dict[str, Any], config: Dict[str, Any]
) -> GeneticEngineAdapter:
    """Convenience wrapper that calls :func:`build_engine` using a dict of
    catalog operator instances and a separate configuration dictionary.

    The *catalog_operators* mapping must contain ``'fitness'``,
    ``'selection'``, ``'crossover'`` and ``'mutation'`` entries; remaining
    engine parameters are read from *config*.  Returns a
    :class:`GeneticEngineAdapter` prepared for benchmarking.
    """
    return build_engine(
        fitness_evaluator=catalog_operators["fitness"],
        selection_op=catalog_operators["selection"],
        crossover_op=catalog_operators["crossover"],
        mutation_op=catalog_operators["mutation"],
        **config,
    )
