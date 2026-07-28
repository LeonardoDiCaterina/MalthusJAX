from typing import Any

from .base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    compute_unroll_num,
)

from malthusjax.engine.island_model.base import BaseIslandModel
from malthusjax.engine.island_model.topologies import FullyConnectedIsland, RingTopologyIsland

from .genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
from .schedules import ScheduleType, TrackBest, compute_scheduled_strength

# from .diversity_engine import DiversityAwareEngine

__all__ = [
    "AbstractEngine",
    "AbstractEvolutionState",
    "AbstractEngineParams",
    "AbstractGenerationOutput",
    "BaseIslandModel",
    "FullyConnectedIsland",
    "RingTopologyIsland",
    "GeneticEngine",
    "GeneticEngineParams",
    "GeneticGenerationOutput",
    "ScheduleType",
    "TrackBest",
    "compute_scheduled_strength",
    "compute_unroll_num",
    # "DiversityAwareEngine",
]


# ---------------------------------------------------------------------------
# Engine catalog registration
# ---------------------------------------------------------------------------


def _register_engines() -> None:
    """Register built-in engines with the global engine registry."""
    from ..composer.engine_registry import register_table

    def _ga_factory(
        evaluator: Any,
        selection: Any,
        crossover: Any,
        mutation: Any,
        genome_type: str = "real",
        pop_size: int = 50,
        generations: int = 100,
        genome_shape: tuple[int, ...] = (10,),
        bounds: tuple[float, float] = (-5.0, 5.0),
        elitism: int = 2,
        **kwargs: Any,
    ) -> Any:
        """Standard genetic algorithm (GeneticEngine).

        Wraps :class:`GeneticEngine` in a
        :class:`~malthusjax.composer.engine_factory.GeneticEngineAdapter`
        compatible with the BenchmarkRunner protocol.
        """
        from ..composer.engine_factory import build_engine

        return build_engine(
            fitness_evaluator=evaluator,
            selection_op=selection,
            crossover_op=crossover,
            mutation_op=mutation,
            genome_type=genome_type,
            pop_size=pop_size,
            generations=generations,
            genome_shape=genome_shape,
            bounds=bounds,
            elitism=elitism,
            **kwargs,
        )

    register_table(
        [
            (
                "ga",
                _ga_factory,
                {
                    "pop_size": 50,
                    "generations": 100,
                    "elitism": 2,
                    "genome_type": "real",
                },
            ),
        ],
        override=True,
    )


_register_engines()
