"""QDAX backend provider — quality diversity algorithms."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


def resolve_qdax_evaluator(
    fitness_spec: Optional[Any], bounds: Tuple[float, float], maximize: bool
) -> Any:
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.core.genome.real_genome import RealGenome, RealPopulation

    class QDAXNativeEvaluator:
        def __init__(self, fitness_fn, evaluator=None, num_descriptors=2):
            self._fitness_fn = fitness_fn
            self._evaluator = evaluator
            self._num_descriptors = num_descriptors

        def scoring_function(self, genotypes, random_key):
            import jax
            import jax.numpy as jnp

            if self._evaluator is not None:
                genes = RealGenome(values=genotypes)
                pop = RealPopulation(
                    genes=genes, fitness=jnp.zeros(genotypes.shape[0]), config=None
                )
                updated_pop = self._evaluator.evaluate_population(pop)
                fitnesses = updated_pop.fitness
                if not maximize:
                    fitnesses = -fitnesses
                if hasattr(updated_pop, "descriptors"):
                    descriptors = updated_pop.descriptors
                else:
                    desc_dims = genotypes[:, : self._num_descriptors]
                    lo, hi = float(bounds[0]), float(bounds[1])
                    descriptors = (desc_dims - lo) / (hi - lo)
            else:
                fitnesses = jax.vmap(self._fitness_fn)(genotypes)
                desc_dims = genotypes[:, : self._num_descriptors]
                lo, hi = float(bounds[0]), float(bounds[1])
                descriptors = (desc_dims - lo) / (hi - lo)
            return fitnesses, descriptors, {}

    if isinstance(fitness_spec, str):
        cat = OperatorCatalog()
        resolved = cat.get(fitness_spec)
        return QDAXNativeEvaluator(None, evaluator=resolved)
    elif fitness_spec is not None:
        return QDAXNativeEvaluator(None, evaluator=fitness_spec)
    else:
        import jax.numpy as jnp

        def fn(x):
            return -jnp.sum(jnp.square(x))

        return QDAXNativeEvaluator(fn)


def build_qdax_engine(
    strategy: Any,
    fitness_spec: Optional[Any],
    pop_size: int,
    generations: int,
    genome_length: int,
    bounds: Tuple[float, float],
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    import functools

    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
    from qdax.utils.metrics import default_qd_metrics

    from malthusjax.composer.qdax_adapter import build_qdax_engine as adapter_build_qdax_engine

    # 1. Resolve strategy class
    strategy_cls = getattr(strategy, "strategy_cls", "MAPElites")
    if isinstance(strategy_cls, str):
        import importlib
        import pkgutil

        import qdax.core

        resolved_cls = None
        for _, name, is_pkg in pkgutil.iter_modules(qdax.core.__path__):
            if not is_pkg:
                mod = importlib.import_module(f"qdax.core.{name}")
                if hasattr(mod, strategy_cls):
                    resolved_cls = getattr(mod, strategy_cls)
                    break
        if resolved_cls is None:
            raise ValueError(f"Unknown QDAX strategy: {strategy_cls}")
        strategy_cls = resolved_cls
    # 2. Auto-create emitter if not provided
    emitter = getattr(strategy, "emitter", None)
    if isinstance(emitter, str):
        if emitter.lower() == "mixing":
            sigma = getattr(strategy, "mutation_sigma", 0.05)
            var_pct = kwargs.pop("qdax_variation_percentage", 0.5)
            emitter_spec = f"qdax_native:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},variation_percentage={var_pct}"
        else:
            if ":" in emitter:
                emitter_spec = f"{emitter},batch_size={pop_size}"
            else:
                emitter_spec = f"{emitter}:batch_size={pop_size}"
        from malthusjax.composer.catalog import OperatorCatalog

        catalog = OperatorCatalog()
        emitter = catalog.get(emitter_spec)
    elif emitter is None:
        sigma = getattr(strategy, "mutation_sigma", 0.05)
        var_pct = kwargs.pop("qdax_variation_percentage", 0.5)
        emitter_spec = f"qdax_native:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},variation_percentage={var_pct}"
        from malthusjax.composer.catalog import OperatorCatalog

        catalog = OperatorCatalog()
        emitter = catalog.get(emitter_spec)
    # 3. Auto-create metrics function if not provided
    metrics_fn = getattr(strategy, "metrics_function", None)
    if metrics_fn is None:
        metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)
    # 4. Auto-compute centroids if not provided
    centroids = getattr(strategy, "centroids", None)
    if centroids is None:
        centroids = compute_cvt_centroids(
            num_descriptors=getattr(strategy, "num_descriptors", 2),
            num_init_cvt_samples=50000,
            num_centroids=getattr(strategy, "num_centroids", 100),
            minval=0.0,
            maxval=1.0,
            key=jr.PRNGKey(42),
        )
    init_variables = getattr(strategy, "init_variables", None)
    evaluator = resolve_qdax_evaluator(fitness_spec, bounds, maximize)
    return adapter_build_qdax_engine(
        strategy_cls=strategy_cls,
        emitter=emitter,
        metrics_function=metrics_fn,
        evaluator=evaluator,
        init_variables=init_variables,
        centroids=centroids,
        pop_size=pop_size,
        generations=generations,
        maximize=maximize,
        eval_mode="native",
        history_metrics=history_metrics or ["qd_score", "coverage"],
        bounds=bounds,
        genome_length=genome_length,
        **kwargs,
    )


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

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        from malthusjax.composer.strategies.core import QDAXStrategy

        return isinstance(strategy, QDAXStrategy)

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
