"""QDAX backend provider — quality diversity algorithms."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


class QdaxProvider:
    """Backend provider for QDAX algorithms."""

    @property
    def name(self) -> str:
        return "qdax"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        from malthusjax.composer.strategies.core import QDAXStrategy

        algo_kwargs = kwargs.get("algorithm_kwargs")
        if algo_kwargs is None:
            exclude = {
                "qdax_strategy",
                "backend",
                "fitness",
                "pop_size",
                "generations",
                "seeds",
                "maximize",
                "bounds",
                "genome_length",
                "prng_impl",
                "step_logging",
                "history_metrics",
                "strategy",
                "qdax_num_descriptors",
                "qdax_num_centroids",
                "qdax_mutation_sigma",
            }
            algo_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}

        return QDAXStrategy(
            strategy_cls=kwargs.get("qdax_strategy", "MAPElites"),
            num_descriptors=kwargs.get("qdax_num_descriptors", 2),
            num_centroids=kwargs.get("qdax_num_centroids", 100),
            mutation_sigma=kwargs.get("qdax_mutation_sigma", 0.05),
            algorithm_kwargs=algo_kwargs,
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
        from malthusjax.composer.factory import resolve_qdax_evaluator

        return resolve_qdax_evaluator(fitness_spec, bounds=bounds, maximize=maximize)

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
        from malthusjax.composer.factory import build_qdax_engine

        engine_kwargs = dict(kwargs)
        for k in (
            "genome_type",
            "engine_type",
            "elitism",
            "data_config",
            "genome_shape",
            "genome_length",
        ):
            engine_kwargs.pop(k, None)

        return build_qdax_engine(
            strategy=strategy,
            fitness_spec=evaluator,
            pop_size=pop_size,
            generations=generations,
            genome_length=kwargs.get("genome_length", 10),
            bounds=bounds,
            maximize=maximize,
            history_metrics=history_metrics,
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


_qdax_provider = QdaxProvider()
register_backend("qdax", _qdax_provider)
