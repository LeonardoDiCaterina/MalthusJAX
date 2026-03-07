from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, cast

import chex

from ..core.genome.binary_genome import BinaryGenomeConfig
from ..core.genome.real_genome import RealGenomeConfig
from ..core.random import resolve_prng_impl
from ..engine.base import compute_unroll_num
from ..engine.genetic_fastengine import GeneticEngine, GeneticEngineParams


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
        Returns:
            dict with keys:
            - 'history': List[Dict[str, Any]] - per-generation stats
            - 'summary': Dict[str, Any] - final summary metrics
            - 'timings': Dict[str, float] - timing info
        """
        import time

        # time initialization (includes any compilation costs on first call)
        t_init_start = time.perf_counter()
        state = self.genetic_engine.init_state(key)
        # record starting best fitness
        initial_best = float(state.best_fitness)
        if hasattr(self, "maximize") and self.maximize:
            initial_best = -initial_best
        t_init_end = time.perf_counter()

        # If an explicit initial population is provided, construct an evaluated
        # population object and replace the state's population with it.
        if self.initial_population is not None:
            import jax.numpy as jnp

            from ..core.genome.real_genome import RealPopulation

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

        history = []
        final_state = state

        t_evo_start = time.perf_counter()
        for _ in range(self.genetic_engine.engine_params.num_generations):
            final_state, metrics = self.genetic_engine.step(final_state)

            history.append(
                {
                    "generation": int(final_state.generation),
                    "best_fitness": float(final_state.best_fitness),
                    "mean_fitness": (
                        float(metrics.mean_fitness) if hasattr(metrics, "mean_fitness") else 0.0
                    ),
                    "std_fitness": 0.0,  # Could compute if needed
                }
            )
        t_evo_end = time.perf_counter()

        total_evals = int(final_state.generation * self.genetic_engine.engine_params.pop_size)
        summary = {
            "initial_fitness": initial_best,
            "best_fitness": float(final_state.best_fitness),
            "final_generation": int(final_state.generation),
            "total_evaluations": total_evals,
        }

        timings = {
            "initialization": t_init_end - t_init_start,
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
    **kwargs: Any,
) -> GeneticEngineAdapter:
    """Build a GeneticEngine from catalog operators.
    Args:
        fitness_evaluator: Fitness evaluator instance
        selection_op: Selection operator instance
        crossover_op: Crossover operator instance
        mutation_op: Mutation operator instance
        genome_type: "real" or "binary"
        pop_size: Population size
        generations: Number of generations
        elitism: Number of elite individuals
        genome_shape: Shape of real genomes
        bounds: Bounds for real genomes (min, max)
        **kwargs: Additional engine parameters
    Returns:
        GeneticEngineAdapter wrapping configured GeneticEngine
    """
    genome_config: Union[RealGenomeConfig, BinaryGenomeConfig]
    # Backwards-compatibility: accept `genome_length` (scalar) as an alias
    # for the single-dimension `genome_shape` argument used elsewhere in the API.
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

    # Coerce operator spec strings into actual operator instances if needed
    OperatorCatalog: Any = None
    try:
        from .catalog import OperatorCatalog as _OperatorCatalog

        OperatorCatalog = _OperatorCatalog
    except Exception:
        # Avoid circular imports breaking; if it fails, user must pass operator instances
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

    # Defensive validation: ensure operators implement required methods
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
    # Resolve PRNG implementation if provided
    prng_impl_str = kwargs.pop("prng_impl", None)
    prng_extra: Dict[str, Any] = {}
    if prng_impl_str is not None:
        prng_extra["prng_impl"] = resolve_prng_impl(prng_impl_str)

    # Pass through schedule fields (new API) and legacy callable (deprecated)
    _schedule_keys = [
        "schedule_type",
        "initial_strength",
        "final_strength",
        "mutation_strength_schedule",
    ]
    schedule_extra: Dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in _schedule_keys
    }

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=generations,
        elitism=elitism,
        unroll_num=compute_unroll_num(generations),
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
    """Build engine from catalog operator instances and config.
    Args:
        catalog_operators: Dict with keys 'fitness', 'selection', 'crossover', 'mutation'
        config: Configuration dict with engine parameters
    Returns:
        GeneticEngineAdapter ready for BenchmarkRunner
    """
    return build_engine(
        fitness_evaluator=catalog_operators["fitness"],
        selection_op=catalog_operators["selection"],
        crossover_op=catalog_operators["crossover"],
        mutation_op=catalog_operators["mutation"],
        **config,
    )
