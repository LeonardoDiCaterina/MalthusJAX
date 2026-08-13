from typing import Any

from malthusjax.engine.island_model.base import BaseIslandModel
from malthusjax.engine.island_model.topologies import FullyConnectedIsland, RingTopologyIsland

from .base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    compute_unroll_num,
)
from .genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
from .native_fastengine import NativeFastEngine
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
    "NativeFastEngine",
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

    def _lightened_ga_factory(
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
        from ..composer.engine_factory import build_engine
        from .genetic_lightened import LightenedGeneticEngine

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
            engine_cls=LightenedGeneticEngine,
            use_vectorized_operators=False,
            **kwargs,
        )

    def _batched_ga_factory(
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
        from ..composer.engine_factory import build_engine
        from .genetic_lightened import LightenedGeneticEngine

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
            engine_cls=LightenedGeneticEngine,
            use_vectorized_operators=True,
            **kwargs,
        )

    def _native_fast_ga_factory(
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
        from ..composer.engine_factory import build_engine
        from .native_fastengine import NativeFastEngine

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
            engine_cls=NativeFastEngine,
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
            (
                "lightened_ga",
                _lightened_ga_factory,
                {
                    "pop_size": 50,
                    "generations": 100,
                    "elitism": 0,
                    "genome_type": "real",
                },
            ),
            (
                "batched_ga",
                _batched_ga_factory,
                {
                    "pop_size": 50,
                    "generations": 100,
                    "elitism": 0,
                    "genome_type": "real",
                },
            ),
            (
                "native_fast",
                _native_fast_ga_factory,
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
