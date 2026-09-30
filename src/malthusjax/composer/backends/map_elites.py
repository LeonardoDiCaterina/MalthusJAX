"""MAP-Elites backend provider."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


class MapElitesProvider:
    """Backend provider for MAP-Elites quality diversity algorithms."""

    @property
    def name(self) -> str:
        return "map_elites"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        from malthusjax.composer.strategies.core import MapElitesStrategy

        return MapElitesStrategy(
            emitter=kwargs.get("map_elites_emitter", kwargs.get("emitter", "mixing")),
            num_descriptors=kwargs.get("qdax_num_descriptors", kwargs.get("num_descriptors", 2)),
            num_centroids=kwargs.get("qdax_num_centroids", kwargs.get("num_centroids", 100)),
        )

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base

        return resolve_evaluator_base(
            fitness_spec,
            maximize=maximize,
            seed=seed,
            num_dims=num_dims,
            bounds=bounds,
        )

    def build_engine(
        self,
        strategy: BaseStrategy,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Any:
        from malthusjax.composer.factory import build_map_elites_engine

        engine_kwargs = dict(kwargs)
        for ignored in (
            "genome_type",
            "engine_type",
            "elitism",
            "data_config",
            "genome_length",
            "genome_shape",
        ):
            engine_kwargs.pop(ignored, None)

        return build_map_elites_engine(
            strategy=strategy,
            fitness_spec=evaluator,
            pop_size=pop_size,
            generations=generations,
            maximize=maximize,
            history_metrics=history_metrics,
            genome_length=kwargs.get("genome_length", kwargs.get("genome_shape", 10)),
            bounds=bounds,
            step_logging=step_logging,
            **engine_kwargs,
        )

    def generate_initial_population(
        self,
        config: Dict[str, Any],
        pop_seed: int,
    ) -> Optional[Any]:
        from malthusjax.composer.config import infer_genome_length

        pop_size = int(config.get("pop_size", 50))
        genome_length = infer_genome_length(config)
        bounds = config.get("bounds", (-5.0, 5.0))
        return jr.uniform(
            jr.PRNGKey(pop_seed),
            (pop_size, genome_length),
            minval=float(bounds[0]),
            maxval=float(bounds[1]),
        )


_map_elites_provider = MapElitesProvider()
register_backend("map_elites", _map_elites_provider)
