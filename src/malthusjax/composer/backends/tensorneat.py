"""TensorNEAT backend provider — topology and neural evolution."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


class TensorneatProvider:
    """Backend provider for TensorNEAT neuroevolution algorithms."""

    @property
    def name(self) -> str:
        return "tensorneat"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        from malthusjax.composer.strategies.core import TensorNEATStrategy

        algo_kwargs = kwargs.get("algorithm_kwargs")
        if algo_kwargs is None:
            exclude = {
                "tensorneat_algorithm",
                "tensorneat_genome",
                "tensorneat_problem",
                "tensorneat_num_inputs",
                "tensorneat_num_outputs",
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
                "data_config",
                "composable",
            }
            algo_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}

        return TensorNEATStrategy(
            algorithm_name=kwargs.get("tensorneat_algorithm", "NEAT"),
            genome_name=kwargs.get("tensorneat_genome", "DefaultGenome"),
            problem_name=kwargs.get("tensorneat_problem", None),
            num_inputs=kwargs.get("tensorneat_num_inputs", kwargs.get("num_inputs", 2)),
            num_outputs=kwargs.get("tensorneat_num_outputs", kwargs.get("num_outputs", 1)),
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
        if kwargs.get("composable", False):
            try:
                from malthusjax.composer.backends._evaluator_resolver import (
                    resolve_evaluator_base,
                )

                return resolve_evaluator_base(
                    fitness_spec,
                    maximize=maximize,
                    seed=seed,
                    num_dims=num_dims,
                    bounds=bounds,
                )
            except Exception:
                return fitness_spec
        return fitness_spec

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
            from malthusjax.composer.factory import build_composable_tensorneat_engine

            builder = build_composable_tensorneat_engine
        else:
            from malthusjax.composer.factory import build_tensorneat_engine

            builder = build_tensorneat_engine

        engine_kwargs = dict(kwargs)
        for ignored in (
            "composable",
            "genome_length",
            "genome_shape",
            "engine_type",
            "genome_type",
            "elitism",
            "data_config",
            "bounds",
            "prng_impl",
        ):
            engine_kwargs.pop(ignored, None)

        return builder(
            strategy=strategy,
            fitness_spec=evaluator,
            pop_size=pop_size,
            generations=generations,
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
        import tensorneat.algorithm
        import tensorneat.genome
        from tensorneat.common import State

        pop_size = int(config.get("pop_size", 50))
        num_inputs = config.get("num_inputs", 2)
        num_outputs = config.get("num_outputs", 1)

        genome = tensorneat.genome.DefaultGenome(num_inputs=num_inputs, num_outputs=num_outputs)
        algorithm = tensorneat.algorithm.NEAT(pop_size=pop_size, genome=genome)

        state = State(randkey=jr.PRNGKey(pop_seed))
        state = algorithm.setup(state)

        pop_nodes = getattr(state, "pop_nodes", state.state_dict.get("pop_nodes"))
        pop_conns = getattr(state, "pop_conns", state.state_dict.get("pop_conns"))
        return (pop_nodes, pop_conns)


_tensorneat_provider = TensorneatProvider()
register_backend("tensorneat", _tensorneat_provider)
register_backend("composable_tensorneat", _tensorneat_provider, defaults={"composable": True})
