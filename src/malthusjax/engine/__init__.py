from .base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    compute_unroll_num,
)
from .genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
from .schedules import ScheduleType, TrackBest, compute_scheduled_strength

# from .diversity_engine import DiversityAwareEngine

__all__ = [
    "AbstractEngine",
    "AbstractEvolutionState",
    "AbstractEngineParams",
    "AbstractGenerationOutput",
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
        evaluator,
        selection,
        crossover,
        mutation,
        genome_type="real",
        pop_size=50,
        generations=100,
        genome_shape=(10,),
        bounds=(-5.0, 5.0),
        elitism=2,
        **kwargs,
    ):
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
