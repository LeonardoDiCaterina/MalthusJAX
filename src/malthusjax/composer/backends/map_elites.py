"""MAP-Elites backend provider."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, cast

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


def build_map_elites_engine(
    strategy: Any,
    fitness_spec: Optional[Any],
    pop_size: int,
    generations: int,
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

    from malthusjax.composer.backends.tensorneat import resolve_tensorneat_problem
    from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
    from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter

    # 1. Instantiate Emitter first
    emitter_obj = getattr(strategy, "emitter", None)
    if isinstance(emitter_obj, str):
        genome_length = kwargs.get("genome_length", 10)
        if emitter_obj.lower() == "mixing":
            sigma = getattr(strategy, "mutation_sigma", 0.1)
            emitter_spec = f"qdax_replica:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},genome_length={genome_length}"
        else:
            if ":" in emitter_obj:
                emitter_spec = f"{emitter_obj},batch_size={pop_size},genome_length={genome_length}"
            else:
                emitter_spec = f"{emitter_obj}:batch_size={pop_size},genome_length={genome_length}"
        from malthusjax.composer.catalog import OperatorCatalog

        catalog = OperatorCatalog()
        emitter_obj = catalog.get(emitter_spec)
    elif emitter_obj is None:
        raise ValueError("MapElitesStrategy requires an explicit emitter.")
    # 2. Resolve Evaluator
    evaluator: Any = None
    if isinstance(emitter_obj, TensorNeatEmitter):
        from malthusjax.core.fitness.composable.base import ScalarOutput
        from malthusjax.core.fitness.composable.environments import TensorNEATProblemWrapper
        from malthusjax.core.fitness.composable.evaluators import TensorNeatEvaluator
        from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
        from plugins.tensor_neat_transform import TensorNeatTransform

        problem_name = kwargs.get("tensorneat_problem", None)
        objective_fn = kwargs.get("objective_function")
        if objective_fn is None:
            problem, problem_state = resolve_tensorneat_problem(
                problem_name, fitness_spec if isinstance(fitness_spec, str) else None
            )
            import inspect

            import tensorneat.genome

            target_genome = kwargs.get("tensorneat_genome", "default").lower()
            genome_cls: Any = None
            for cls_name, cls_obj in inspect.getmembers(tensorneat.genome, inspect.isclass):
                name_lower = cls_name.lower()
                if name_lower == target_genome or name_lower == f"{target_genome}genome":
                    genome_cls = cls_obj
                    break
            if genome_cls is None:
                genome_cls = tensorneat.genome.DefaultGenome
            genome_obj = genome_cls(
                num_inputs=kwargs.get("tensorneat_num_inputs", 2),
                num_outputs=kwargs.get("tensorneat_num_outputs", 1),
            )
            emitter_obj = cast(Any, emitter_obj).replace(genome=genome_obj)

            evaluator = TensorNeatEvaluator(
                env=TensorNEATProblemWrapper(problem=problem),
                transform=TensorNeatTransform(algorithm=genome_obj),
                interpreter=IdentityInterpreter(),
                output=ScalarOutput(),
                forward_fn=genome_obj.forward,
                maximize=maximize,
            )
        else:
            from malthusjax.core.fitness.base import BaseEvaluatorConfig
            from malthusjax.core.fitness.tensorneat import TensorNeatQDEvaluator

            evaluator = TensorNeatQDEvaluator(
                objective_function=objective_fn,
                config=BaseEvaluatorConfig(maximize=maximize),
                data=None,
            )
    else:
        from malthusjax.composer.catalog import OperatorCatalog
        from malthusjax.core.fitness.base import BaseEvaluatorConfig
        from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
        from malthusjax.core.genome.qd.population import QDPopulation

        cat = OperatorCatalog()
        seed_val = kwargs.get("seed", 42)
        resolved_base_evaluator: Any
        if isinstance(fitness_spec, str):
            if "seed=" not in fitness_spec:
                fitness_spec = (
                    f"{fitness_spec}:seed={seed_val}"
                    if ":" not in fitness_spec
                    else f"{fitness_spec},seed={seed_val}"
                )
            resolved_base_evaluator = cat.get(fitness_spec)
        elif fitness_spec is not None:
            resolved_base_evaluator = fitness_spec
        else:
            resolved_base_evaluator = cat.get(
                f"sphere:dim={kwargs.get('genome_length', 10)},maximize={maximize},seed={seed_val}"
            )
        bounds = kwargs.get("bounds", (-5.0, 5.0))
        num_desc = getattr(strategy, "num_descriptors", 2)

        class ComposedQDEvaluator(BaseQDEvaluator[Any, Any, Any]):
            def evaluate_qd(genome):
                raise NotImplementedError("Use evaluate_population directly")

            def evaluate_population(population):
                updated_pop = resolved_base_evaluator.evaluate_population(population)
                genotypes = getattr(population.genes, "values", population.genes)
                desc_dims = genotypes[:, :num_desc]
                lo, hi = float(bounds[0]), float(bounds[1])
                descriptors = (desc_dims - lo) / (hi - lo)
                new_info = dict(population.info) if population.info else {}
                new_info["descriptors"] = descriptors
                fitness = updated_pop.fitness
                return QDPopulation(
                    genes=population.genes,
                    fitness=fitness,
                    config=population.config,
                    info=new_info,
                )

        evaluator = ComposedQDEvaluator(config=BaseEvaluatorConfig(maximize=maximize), data=None)
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
    engine: Any = MapElitesEngine(
        emitter=emitter_obj,
        evaluator=evaluator,
        engine_params=MapElitesEngineParams(
            pop_size=pop_size, num_generations=generations, maximize=maximize
        ),
    )
    from malthusjax.composer.adapters.map_elites_adapter import MapElitesEngineAdapter

    return MapElitesEngineAdapter(
        engine=engine,
        pop_size=pop_size,
        maximize=maximize,
        history_metrics=history_metrics,
        initial_population=kwargs.get("initial_population", None),
        centroids=centroids,
    )


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

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        from malthusjax.composer.strategies.core import MapElitesStrategy

        return isinstance(strategy, MapElitesStrategy)

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
