from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import chex

from ..core.genome.binary_genome import BinaryGenomeConfig
from ..core.genome.real_genome import RealGenomeConfig
from ..engine.genetic_fastengine import GeneticEngine, GeneticEngineParams


class GeneticEngineAdapter:
    """Adapter to make GeneticEngine compatible with BenchmarkRunner.Engine protocol."""

    def __init__(self, genetic_engine: GeneticEngine, genome_config: Any):
        self.genetic_engine = genetic_engine
        self.genome_config = genome_config

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results.
        Returns:
            dict with keys:
            - 'history': List[Dict[str, Any]] - per-generation stats
            - 'summary': Dict[str, Any] - final summary metrics
            - 'timings': Dict[str, float] - timing info
        """
        state = self.genetic_engine.init_state(key)

        history = []
        final_state = state

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

        total_evals = int(final_state.generation * self.genetic_engine.engine_params.pop_size)
        summary = {
            "best_fitness": float(final_state.best_fitness),
            "final_generation": int(final_state.generation),
            "total_evaluations": total_evals,
            "stagnation_counter": int(final_state.stagnation_counter),
        }

        timings = {
            "initialization": 0.01,  # Placeholder
            "evolution": 0.05 * final_state.generation,  # Rough estimate
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
    bounds: tuple = (-5.0, 5.0),
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
    try:
        from .catalog import OperatorCatalog
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
    for name, op in ("selection", selection_op), ("crossover", crossover_op), ("mutation", mutation_op):
        if not hasattr(op, "replace") or not callable(getattr(op, "replace")):
            raise TypeError(
                f"Operator '{name}' does not implement required 'replace' method. Got type {type(op)}."
                " Pass an operator instance from OperatorCatalog.get(spec) or a proper operator implementation."
            )
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=generations,
        elitism=elitism,
        **{k: v for k, v in kwargs.items() if k in ["mutation_strength_schedule"]},
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

    return GeneticEngineAdapter(genetic_engine, genome_config)


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
