from typing import Any

from malthusjax.engine.island_model.adapter import IslandEngineAdapter, IslandEvolutionState
from malthusjax.engine.island_model.base import BaseIslandModel
from malthusjax.engine.island_model.topologies import FullyConnectedIsland, RingTopologyIsland
from malthusjax.engine.qd.adapter import (
    IslandMapElitesAdapter,
    IslandMapElitesEvolutionState,
    MapElitesEngineAdapter,
    MapElitesEvolutionState,
)

from .base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    compute_unroll_num,
)
from .genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticFastEngine,
    GeneticGenerationOutput,
)
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
    "IslandEngineAdapter",
    "IslandEvolutionState",
    "MapElitesEngineAdapter",
    "IslandMapElitesAdapter",
    "MapElitesEvolutionState",
    "IslandMapElitesEvolutionState",
    "GeneticEngine",
    "GeneticFastEngine",
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

    def _island_factory(
        evaluator: Any,
        selection: Any = None,
        crossover: Any = None,
        mutation: Any = None,
        genome_type: str = "real",
        pop_size: int = 50,
        generations: int = 100,
        elitism: int = 2,
        num_islands: int = 4,
        migration_interval: int = 20,
        num_migrants: int = 1,
        topology: str = "ring",
        genome_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Distributed Island Model engine across isolated topologies."""
        from ..composer.engine_factory import build_engine
        from .island_model.adapter import IslandEngineAdapter
        from .island_model.topologies import FullyConnectedIsland, RingTopologyIsland

        base_adapter = build_engine(
            fitness_evaluator=evaluator,
            selection_op=selection,
            crossover_op=crossover,
            mutation_op=mutation,
            genome_type=genome_type,
            genome_config=genome_config,
            pop_size=pop_size,
            generations=migration_interval,
            elitism=elitism,
            **kwargs,
        )
        base_engine = base_adapter.genetic_engine

        island_model: BaseIslandModel[Any]
        if topology == "ring":
            island_model = RingTopologyIsland(
                engine=base_engine,
                num_islands=num_islands,
                migration_interval=migration_interval,
                num_migrants=num_migrants,
            )
        elif topology == "fully_connected":
            island_model = FullyConnectedIsland(
                engine=base_engine,
                num_islands=num_islands,
                migration_interval=migration_interval,
                num_migrants=num_migrants,
            )
        else:
            raise ValueError(
                f"Unknown island topology: '{topology}'. Expected 'ring' or 'fully_connected'."
            )

        return IslandEngineAdapter(
            island_model=island_model,
            generations=generations,
            genome_config=base_adapter.genome_config,
            maximize=base_adapter.maximize,
        )

    def _map_elites_factory(
        evaluator: Any,
        emitter: Any = None,
        genome_type: str = "real",
        pop_size: int = 20,
        generations: int = 100,
        num_centroids: int = 50,
        num_init_cvt_samples: int = 5000,
        minval: Any = (0.0, 0.0),
        maxval: Any = (1.0, 1.0),
        genome_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Quality-Diversity MAP-Elites engine factory."""
        from ..composer.genome_catalog import GenomeCatalog
        from .qd.adapter import MapElitesEngineAdapter
        from .qd.map_elites import MapElitesEngine, MapElitesEngineParams

        if genome_config is None:
            genome_config = GenomeCatalog().get(genome_type)

        if emitter is None:
            if "emitter_spec" in kwargs:
                from ..composer.catalog import OperatorCatalog

                emitter = OperatorCatalog().get(kwargs.pop("emitter_spec"))
            else:
                raise ValueError(
                    "map_elites engine requires an 'emitter' instance (or 'emitter_spec' in parameters). "
                    f"No emitter was provided for genome_type '{genome_type}'."
                )

        maximize = (
            getattr(evaluator.config, "maximize", False) if hasattr(evaluator, "config") else False
        )
        engine_params = MapElitesEngineParams(
            pop_size=pop_size,
            num_generations=generations,
            maximize=maximize,
        )
        base_engine: Any = MapElitesEngine(
            emitter=emitter,
            evaluator=evaluator,
            engine_params=engine_params,
        )
        return MapElitesEngineAdapter(
            engine=base_engine,
            genome_config=genome_config,
            generations=generations,
            num_centroids=num_centroids,
            num_init_cvt_samples=num_init_cvt_samples,
            minval=minval,
            maxval=maxval,
            maximize=maximize,
        )

    def _island_map_elites_factory(
        evaluator: Any,
        emitter: Any = None,
        genome_type: str = "real",
        pop_size: int = 20,
        generations: int = 100,
        num_islands: int = 4,
        migration_interval: int = 20,
        num_migrants: int = 2,
        num_centroids: int = 50,
        num_init_cvt_samples: int = 5000,
        minval: Any = (0.0, 0.0),
        maxval: Any = (1.0, 1.0),
        topology: str = "ring",
        genome_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Distributed Quality-Diversity Island MAP-Elites engine factory."""
        from ..composer.genome_catalog import GenomeCatalog
        from .qd.adapter import IslandMapElitesAdapter
        from .qd.map_elites import MapElitesEngine, MapElitesEngineParams

        if genome_config is None:
            genome_config = GenomeCatalog().get(genome_type)

        if emitter is None:
            if "emitter_spec" in kwargs:
                from ..composer.catalog import OperatorCatalog

                emitter = OperatorCatalog().get(kwargs.pop("emitter_spec"))
            else:
                raise ValueError(
                    "island_map_elites engine requires an 'emitter' instance (or 'emitter_spec' in parameters). "
                    f"No emitter was provided for genome_type '{genome_type}'."
                )

        maximize = (
            getattr(evaluator.config, "maximize", False) if hasattr(evaluator, "config") else False
        )
        engine_params = MapElitesEngineParams(
            pop_size=pop_size,
            num_generations=migration_interval,
            maximize=maximize,
        )
        base_engine: Any = MapElitesEngine(
            emitter=emitter,
            evaluator=evaluator,
            engine_params=engine_params,
        )
        return IslandMapElitesAdapter(
            engine=base_engine,
            genome_config=genome_config,
            num_islands=num_islands,
            migration_interval=migration_interval,
            num_migrants=num_migrants,
            generations=generations,
            num_centroids=num_centroids,
            num_init_cvt_samples=num_init_cvt_samples,
            minval=minval,
            maxval=maxval,
            maximize=maximize,
            topology=topology,
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
                "island",
                _island_factory,
                {
                    "pop_size": 50,
                    "generations": 100,
                    "elitism": 2,
                    "num_islands": 4,
                    "migration_interval": 20,
                    "num_migrants": 1,
                    "topology": "ring",
                    "genome_type": "real",
                },
            ),
            (
                "map_elites",
                _map_elites_factory,
                {
                    "pop_size": 20,
                    "generations": 100,
                    "num_centroids": 50,
                    "genome_type": "real",
                },
            ),
            (
                "island_map_elites",
                _island_map_elites_factory,
                {
                    "pop_size": 20,
                    "generations": 100,
                    "num_islands": 4,
                    "migration_interval": 20,
                    "num_migrants": 2,
                    "num_centroids": 50,
                    "topology": "ring",
                    "genome_type": "real",
                },
            ),
        ],
        override=True,
    )


_register_engines()
