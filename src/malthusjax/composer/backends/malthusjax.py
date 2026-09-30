"""MalthusJAX native backend provider."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


class MalthusJAXProvider:
    """Backend provider for native MalthusJAX engines."""

    @property
    def name(self) -> str:
        return "malthusjax"

    def default_strategy(self, **kwargs: Any) -> BaseStrategy:
        from malthusjax.composer.factory import has_real_operators
        from malthusjax.composer.strategies.core import GeneticStrategy, MapElitesStrategy

        engine_type = kwargs.get("engine_type")
        if engine_type == "map_elites":
            return MapElitesStrategy(
                emitter=kwargs.get("map_elites_emitter", kwargs.get("emitter", "mixing")),
                num_descriptors=kwargs.get("qdax_num_descriptors", 2),
                num_centroids=kwargs.get("qdax_num_centroids", 100),
            )

        genome = kwargs.get("genome")
        fitness = kwargs.get("fitness")
        selection = kwargs.get("selection")
        crossover = kwargs.get("crossover")
        mutation = kwargs.get("mutation")

        if has_real_operators(genome, fitness, selection, crossover, mutation):
            return GeneticStrategy(
                selection=selection,
                crossover=crossover,
                mutation=mutation,
            )

        return BaseStrategy()

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
        from malthusjax.composer.factory import build_data_registry

        data_config = kwargs.get("data_config")
        data_registry = build_data_registry(data_config) if data_config else None

        return resolve_evaluator_base(
            fitness_spec,
            maximize=maximize,
            seed=seed,
            num_dims=num_dims,
            bounds=bounds,
            data_registry=data_registry,
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
        from malthusjax.composer.strategies.core import GeneticStrategy, MapElitesStrategy

        if isinstance(strategy, MapElitesStrategy):
            from malthusjax.composer.factory import build_map_elites_engine

            engine_kwargs = dict(kwargs)
            for k in (
                "genome_length",
                "genome_shape",
                "pop_size",
                "generations",
                "maximize",
                "bounds",
                "history_metrics",
                "step_logging",
            ):
                engine_kwargs.pop(k, None)

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

        if isinstance(strategy, GeneticStrategy):
            from malthusjax.composer.factory import build_real_engine

            engine_kwargs = dict(kwargs)
            for k in (
                "genome",
                "fitness",
                "engine_type",
                "genome_type",
                "pop_size",
                "generations",
                "genome_shape",
                "genome_length",
                "bounds",
                "elitism",
                "maximize",
                "prng_impl",
                "data_config",
                "history_metrics",
                "step_logging",
            ):
                engine_kwargs.pop(k, None)

            return build_real_engine(
                strategy=strategy,
                genome=kwargs.get("genome"),
                fitness=evaluator,
                engine_type=kwargs.get("engine_type", "ga"),
                genome_type=kwargs.get("genome_type", "real"),
                pop_size=pop_size,
                generations=generations,
                genome_shape=kwargs.get("genome_length", kwargs.get("genome_shape", 10)),
                bounds=bounds,
                elitism=kwargs.get("elitism", True),
                maximize=maximize,
                prng_impl=kwargs.get("prng_impl"),
                data_config=kwargs.get("data_config"),
                history_metrics=history_metrics,
                step_logging=step_logging,
                **engine_kwargs,
            )

        from malthusjax.composer.factory import build_stub_engine

        return build_stub_engine(generations, **kwargs)

    def generate_initial_population(
        self,
        config: Dict[str, Any],
        pop_seed: int,
    ) -> Optional[Any]:
        from malthusjax.composer.catalog import OperatorCatalog
        from malthusjax.composer.config import infer_genome_length
        from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
        from malthusjax.core.fitness.composable.environments import BBOBEnv
        from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
        from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

        pop_size = int(config.get("pop_size", 50))
        genome_length = infer_genome_length(config)
        bounds = config.get("bounds", (-5.0, 5.0))
        fitness_spec = config.get("fitness")

        if fitness_spec and isinstance(fitness_spec, str) and "bbob" in fitness_spec.lower():
            cat = OperatorCatalog()
            parsed_name, parsed_params = cat.parse_spec(fitness_spec)
            if parsed_name == "bbob":
                fn = parsed_params.get("fn_name", parsed_params.get("fn", "rosenbrock"))
                dims = parsed_params.get("dim", parsed_params.get("num_dims", genome_length))
                bbob_seed = parsed_params.get("seed", 0)
                bbob_eval = OptimizationEvaluator(
                    env=BBOBEnv.create(fn_name=fn, num_dims=dims, seed=bbob_seed),
                    transform=IdentityTransform(),
                    interpreter=IdentityInterpreter(),
                    output=ScalarOutput(maximize=config.get("maximize", False)),
                )
                pop_key = jr.PRNGKey(pop_seed)
                sample_keys = jr.split(pop_key, pop_size)
                return jax.vmap(bbob_eval.env._problem.sample)(sample_keys)

        return jr.uniform(
            jr.PRNGKey(pop_seed),
            (pop_size, genome_length),
            minval=float(bounds[0]),
            maxval=float(bounds[1]),
        )


_malthusjax_provider = MalthusJAXProvider()
register_backend("malthusjax", _malthusjax_provider)
