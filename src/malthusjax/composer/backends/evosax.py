"""Evosax backend provider — wraps EvoSAX strategies and composable variant."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


def build_evosax_engine(
    strategy_name: str,
    fitness_spec: Optional[Any],
    pop_size: int,
    generations: int,
    num_dims: int,
    bounds: Tuple[float, float],
    maximize: bool,
    prng_impl: Optional[str] = None,
    history_metrics: Optional[Sequence[str]] = None,
    step_logging: Any = None,
    **kwargs: Any,
) -> Any:
    from malthusjax.composer.evosax_adapter import (
        build_evosax_engine as adapter_build_evosax_engine,
    )

    if isinstance(fitness_spec, (str, dict)):
        from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base

        fitness_spec = resolve_evaluator_base(
            fitness_spec,
            maximize=maximize,
            seed=kwargs.get("seed", 42),
            num_dims=num_dims,
            bounds=bounds,
        )

    return adapter_build_evosax_engine(
        strategy_name=strategy_name,
        evaluator=fitness_spec,
        pop_size=pop_size,
        generations=generations,
        num_dims=num_dims,
        bounds=bounds,
        maximize=maximize,
        prng_impl=prng_impl,
        history_metrics=history_metrics,
        step_logging=step_logging,
        **kwargs,
    )


def build_composable_evosax_engine(
    strategy_name: str = "SimpleGA",
    fitness_spec: Optional[Any] = None,
    pop_size: int = 50,
    generations: int = 100,
    num_dims: int = 1,
    bounds: Optional[Tuple[float, float]] = None,
    maximize: bool = False,
    prng_impl: Optional[str] = None,
    history_metrics: Optional[Sequence[str]] = None,
    step_logging: Any = None,
    **kwargs: Any,
) -> Any:
    from malthusjax.composer.composable_evosax_adapter import (
        build_composable_evosax_engine as adapter_build_composable_evosax_engine,
    )

    if isinstance(fitness_spec, (str, dict)):
        from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base

        fitness_spec = resolve_evaluator_base(
            fitness_spec,
            maximize=maximize,
            seed=kwargs.get("seed", 42),
            num_dims=num_dims,
            bounds=bounds or (-5.0, 5.0),
        )

    return adapter_build_composable_evosax_engine(
        strategy_name=strategy_name,
        evaluator=fitness_spec,
        pop_size=pop_size,
        generations=generations,
        num_dims=num_dims,
        bounds=bounds,
        maximize=maximize,
        strategy_params=kwargs.get("strategy_params"),
        prng_impl=prng_impl,
        history_metrics=history_metrics,
        step_logging=step_logging,
        **kwargs,
    )


class EvosaxProvider:
    """Backend provider for EvoSAX evolutionary strategies."""

    @property
    def name(self) -> str:
        return "evosax"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        from malthusjax.composer.strategies.core import EvoSAXStrategy

        algo_kwargs = kwargs.get("algorithm_kwargs")
        if algo_kwargs is None:
            exclude = {
                "evosax_strategy",
                "backend",
                "fitness",
                "pop_size",
                "generations",
                "seeds",
                "maximize",
                "bounds",
                "num_dims",
                "genome_length",
                "prng_impl",
                "step_logging",
                "history_metrics",
                "strategy",
                "data_config",
                "composable",
            }
            algo_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}

        return EvoSAXStrategy(
            algorithm_name=kwargs.get("evosax_strategy", "SimpleGA"),
            algorithm_kwargs=algo_kwargs,
        )

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        from malthusjax.composer.strategies.core import EvoSAXStrategy

        return isinstance(strategy, EvoSAXStrategy)

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
        composable: bool = False,
        **kwargs: Any,
    ) -> Any:
        if composable:
            builder = build_composable_evosax_engine
        else:
            builder = build_evosax_engine

        algo_kwargs = getattr(strategy, "algorithm_kwargs", {}) or {}
        num_dims = kwargs.get("num_dims", kwargs.get("genome_length", 10))

        engine_kwargs = {**algo_kwargs, **kwargs}
        for ignored in (
            "composable",
            "genome_length",
            "genome_shape",
            "engine_type",
            "genome_type",
            "elitism",
            "data_config",
            "num_dims",
            "pop_size",
            "generations",
            "bounds",
            "maximize",
            "prng_impl",
            "history_metrics",
            "step_logging",
            "strategy_name",
            "fitness_spec",
            "evaluator",
        ):
            engine_kwargs.pop(ignored, None)

        return builder(
            strategy_name=getattr(strategy, "algorithm_name", "SimpleGA"),
            fitness_spec=evaluator,
            pop_size=pop_size,
            generations=generations,
            num_dims=num_dims,
            bounds=bounds,
            maximize=maximize,
            prng_impl=kwargs.get("prng_impl"),
            history_metrics=history_metrics,
            step_logging=step_logging,
            **engine_kwargs,
        )

    def generate_initial_population(
        self,
        config: Any,
        pop_seed: int,
    ) -> Optional[Any]:
        from malthusjax.composer.backends._population_init import (
            generate_initial_population as _gen_init_pop,
        )

        return _gen_init_pop(config, pop_seed)


_evosax_provider = EvosaxProvider()
register_backend("evosax", _evosax_provider)
register_backend("composable_evosax", _evosax_provider, defaults={"composable": True})
