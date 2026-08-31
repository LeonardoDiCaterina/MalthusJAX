from typing import Any, Dict, Optional, Sequence, Tuple, cast

import jax
import jax.random as jr

from malthusjax.benchmarking import StubEngine
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.engine_catalog import EngineRegistry
from malthusjax.composer.strategies.base import BaseStrategy
from malthusjax.composer.strategies.core import (
    GeneticStrategy,
    MapElitesStrategy,
    QDAXStrategy,
    TensorNEATStrategy,
)
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


def has_real_operators(
    genome: Optional[str],
    fitness: Optional[str],
    selection: Optional[str],
    crossover: Optional[str],
    mutation: Optional[str],
) -> bool:
    """Check if any real operator specs are provided."""
    return any([genome, fitness, selection, crossover, mutation])


def build_data_registry(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a configuration of data sources into a resolved data registry."""
    from malthusjax.benchmarking.registry import DataRegistry

    reg = DataRegistry()
    for data_id, data_spec in data_config.items():
        reg.register(data_id, data_spec)
    resolved: Dict[str, Any] = {}
    for data_id in data_config.keys():
        resolved[data_id] = reg.resolve(data_id)
    return resolved


def build_real_engine(
    strategy: BaseStrategy,
    fitness: Optional[str],
    engine_type: str = "ga",
    data_config: Optional[Dict[str, Any]] = None,
    **config: Any,
) -> Any:
    """Build engine from operator specs and config via EngineRegistry."""
    from malthusjax.engine.schedules import TrackBest

    catalog = OperatorCatalog()
    data_registry = None
    if data_config:
        data_registry = build_data_registry(data_config)
    seed_val = config.get("seed", 42)
    maximize_flag = config.get("maximize", False)
    # We append seed to fitness strings if missing so BBOB etc uses the right seed
    if isinstance(fitness, str):
        if "seed=" not in fitness:
            if ":" in fitness:
                fitness = f"{fitness},seed={seed_val}"
            else:
                fitness = f"{fitness}:seed={seed_val}"
        resolved_evaluator = catalog.get(
            fitness,
            data_registry=data_registry,
        )
    elif fitness is not None:
        resolved_evaluator = fitness
    else:
        resolved_evaluator = catalog.get(
            f"sphere:dim=10,maximize={maximize_flag},seed={seed_val}",
            data_registry=data_registry,
        )
    if isinstance(strategy, GeneticStrategy):
        resolved_selection = (
            catalog.get(
                strategy.selection
                or f"tournament:num_selections={config.get('pop_size', 50) // 2},tournament_size=3",
                data_registry=data_registry,
            )
            if isinstance(strategy.selection, str) or strategy.selection is None
            else strategy.selection
        )
        resolved_crossover = (
            catalog.get(strategy.crossover or "blend:alpha=0.5", data_registry=data_registry)
            if isinstance(strategy.crossover, str) or strategy.crossover is None
            else strategy.crossover
        )
        resolved_mutation = (
            catalog.get(
                strategy.mutation or "gaussian:mutation_rate=0.1", data_registry=data_registry
            )
            if isinstance(strategy.mutation, str) or strategy.mutation is None
            else strategy.mutation
        )
    else:
        resolved_selection = None
        resolved_crossover = None
        resolved_mutation = None
    # Ensure we use LIGHT tracking for monotonic convergence curves
    if "track_best" not in config:
        config["track_best"] = TrackBest.LIGHT
    engine_registry = EngineRegistry()
    return engine_registry.get(
        engine_type,
        evaluator=resolved_evaluator,
        selection=resolved_selection,
        crossover=resolved_crossover,
        mutation=resolved_mutation,
        **config,
    )


def build_evosax_engine(
    strategy_name: str,
    fitness_spec: Optional[str],
    pop_size: int,
    generations: int,
    num_dims: int,
    bounds: Tuple[float, float],
    maximize: bool,
    prng_impl: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Build EvosaxEngineAdapter from strategy name and config."""
    from .evosax_adapter import build_evosax_engine as adapter_build_evosax_engine

    if fitness_spec is not None:
        if isinstance(fitness_spec, str):
            from .catalog import OperatorCatalog

            cat = OperatorCatalog()
            parsed_name, parsed_params = cat.parse_spec(fitness_spec)
            if parsed_name == "bbob":
                fn_param = parsed_params.get("fn_name", parsed_params.get("fn", None))
                if isinstance(fn_param, int):
                    import evosax.problems.bbob.meta_bbob as mb

                    bbob_keys = list(mb.bbob_fns.keys())
                    if fn_param < 1 or fn_param > len(bbob_keys):
                        raise ValueError(
                            f"BBOB function index {fn_param} is out of range (1-{len(bbob_keys)})"
                        )
                    fn = bbob_keys[fn_param - 1]
                elif fn_param is None:
                    raise ValueError(
                        "BBOB fitness specification requires either fn_name or fn index"
                    )
                else:
                    fn = fn_param
            else:
                fn = parsed_params.get("fn_name", parsed_name)
            dims = parsed_params.get("dim", parsed_params.get("num_dims", num_dims))
            seed = parsed_params.get("seed", kwargs.get("seed", 42))
            maxim = parsed_params.get("maximize", maximize)
            evalr = BBOBEvaluator.create(
                BBOBConfig(fn_name=fn, num_dims=dims, seed=seed, maximize=maxim)
            )
        else:
            evalr = fitness_spec
            maxim = (
                getattr(evalr.config, "maximize", maximize)
                if hasattr(evalr, "config")
                else maximize
            )
    else:
        fn = "sphere"
        dims = num_dims
        seed = kwargs.get("seed", 42)
        maxim = maximize
        evalr = BBOBEvaluator.create(
            BBOBConfig(fn_name=fn, num_dims=dims, seed=seed, maximize=maxim)
        )
    # If no initial_population provided and evaluator has sample() method,
    # use it for consistent initialization across backends
    init_pop = kwargs.get("initial_population")
    if init_pop is None and hasattr(evalr, "evosax_problem"):
        # Use same seed as BBOB problem for consistent initialization
        pop_key = jr.PRNGKey(seed)
        sample_keys = jr.split(pop_key, pop_size)
        init_pop = jax.vmap(evalr.evosax_problem.sample)(sample_keys)
    return adapter_build_evosax_engine(
        strategy_name=strategy_name,
        evaluator=evalr,
        pop_size=pop_size,
        generations=generations,
        num_dims=num_dims,
        bounds=bounds,
        maximize=maxim,
        strategy_params=kwargs.get("strategy_params"),
        initial_population=init_pop,
        prng_impl=prng_impl,
    )


def build_qdax_engine(
    strategy: QDAXStrategy,
    fitness_spec: Optional[str],
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

    from .qdax_adapter import build_qdax_engine as adapter_build_qdax_engine

    # 1. Resolve strategy class
    strategy_cls = strategy.strategy_cls
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
    emitter = strategy.emitter
    if isinstance(emitter, str):
        if emitter.lower() == "mixing":
            sigma = strategy.mutation_sigma
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
        sigma = strategy.mutation_sigma
        var_pct = kwargs.pop("qdax_variation_percentage", 0.5)
        emitter_spec = f"qdax_native:mutation=gaussian:sigma={sigma},crossover=none,batch_size={pop_size},variation_percentage={var_pct}"
        from malthusjax.composer.catalog import OperatorCatalog

        catalog = OperatorCatalog()
        emitter = catalog.get(emitter_spec)
    # 3. Auto-create metrics function if not provided
    metrics_fn = strategy.metrics_function
    if metrics_fn is None:
        metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)
    # 4. Auto-compute centroids if not provided
    centroids = strategy.centroids
    if centroids is None:
        centroids = compute_cvt_centroids(
            num_descriptors=strategy.num_descriptors,
            num_init_cvt_samples=50000,
            num_centroids=strategy.num_centroids,
            minval=0.0,
            maxval=1.0,
            key=jr.PRNGKey(42),
        )
    # 5. Auto-generate init_variables if not provided
    # Note: We do NOT generate it here statically with PRNGKey(0) anymore.
    # It is handled dynamically by QDaxEngineAdapter._adapter_init using the runtime seed
    # or it is overridden by the shared_initial_population feature.
    init_variables = strategy.init_variables
    # 6. Resolve evaluator
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


def resolve_qdax_evaluator(
    fitness_spec: Optional[str], bounds: Tuple[float, float], maximize: bool
) -> Any:
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


def build_tensorneat_engine(
    strategy: TensorNEATStrategy,
    fitness_spec: Optional[str],
    pop_size: int,
    generations: int,
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    import inspect

    # 1. Resolve algorithm
    import tensorneat.algorithm

    from .tensorneat_adapter import build_tensorneat_engine as adapter_build_tensorneat_engine

    algorithm_cls: Any = None
    for name, obj in inspect.getmembers(tensorneat.algorithm, inspect.isclass):
        if name.lower() == strategy.algorithm_name.lower():
            algorithm_cls = obj
            break
    if algorithm_cls is None:
        raise ValueError(f"Unknown TensorNEAT algorithm: {strategy.algorithm_name}")
    # 2. Resolve genome
    import tensorneat.genome

    genome_cls: Any = None
    # In TensorNEAT, genome classes often end with "Genome" (e.g. DefaultGenome, RecurrentGenome)
    # So if user passes "default", we check for "default" or "defaultgenome"
    target_genome = strategy.genome_name.lower()
    for name, obj in inspect.getmembers(tensorneat.genome, inspect.isclass):
        name_lower = name.lower()
        if name_lower == target_genome or name_lower == f"{target_genome}genome":
            genome_cls = obj
            break
    if genome_cls is None:
        raise ValueError(f"Unknown TensorNEAT genome: {strategy.genome_name}")
    genome = genome_cls(num_inputs=strategy.num_inputs, num_outputs=strategy.num_outputs)
    # initial_population may have leaked into algorithm_kwargs via kwargs
    alg_kwargs = strategy.algorithm_kwargs.copy()
    init_pop = alg_kwargs.pop("initial_population", kwargs.get("initial_population", None))
    algorithm = algorithm_cls(pop_size=pop_size, genome=genome, **alg_kwargs)
    problem, problem_state = resolve_tensorneat_problem(strategy.problem_name, fitness_spec)
    return adapter_build_tensorneat_engine(
        algorithm=algorithm,
        evaluator=(problem, problem_state),
        generations=generations,
        pop_size=pop_size,
        maximize=maximize,
        history_metrics=history_metrics,
        initial_population=init_pop,
    )


def resolve_tensorneat_problem(
    problem_name: Optional[str], fitness_spec: Optional[str]
) -> Tuple[Any, Any]:
    import inspect

    import tensorneat.problem

    name = problem_name or (fitness_spec if isinstance(fitness_spec, str) else "xor")
    base_name = name.split(":")[0].lower()
    problem_cls: Any = None
    for cls_name, cls_obj in inspect.getmembers(tensorneat.problem, inspect.isclass):
        if cls_name.lower() == base_name:
            problem_cls = cls_obj
            break
    if problem_cls is None:
        raise ValueError(f"Unknown TensorNEAT problem: {base_name}.")
    kwargs: Dict[str, Any] = {}
    if ":" in name:
        args_part = name.split(":", 1)[1]
        for kv in args_part.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k] = v
    if base_name == "gymnaxenv" and "action_policy" not in kwargs:
        import jax.numpy as jnp

        def default_discrete_policy(randkey, forward_func, obs):
            logits = forward_func(obs)
            if logits.ndim > 0 and logits.shape[-1] > 1:
                return jnp.reshape(jnp.argmax(logits, axis=-1), ())
            return logits

        kwargs["action_policy"] = default_discrete_policy
    problem = problem_cls(**kwargs)
    return problem, problem.setup()


def build_map_elites_engine(
    strategy: MapElitesStrategy,
    fitness_spec: Optional[str],
    pop_size: int,
    generations: int,
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    import jax.random as jr
    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

    # Resolve Evaluator using EngineRegistry or direct resolution (for MAP-Elites parity)
    from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams
    from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter

    # 1. Instantiate Emitter first
    emitter_obj = strategy.emitter
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
        from malthusjax.core.fitness.base import BaseEvaluatorConfig
        from malthusjax.core.fitness.tensorneat import TensorNeatQDEvaluator

        problem_name = kwargs.get("tensorneat_problem", None)
        # Extract objective_function if passed directly
        objective_fn = kwargs.get("objective_function")
        # If not passed directly, try to resolve from fitness_spec
        if objective_fn is None:
            # Resolve using tensorneat problem registry
            problem, problem_state = resolve_tensorneat_problem(
                problem_name, fitness_spec if isinstance(fitness_spec, str) else None
            )
            # Create genome to get forward function
            import tensorneat.genome

            target_genome = kwargs.get("tensorneat_genome", "default").lower()
            genome_cls: Any = None
            import inspect

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
            # Assign genome to emitter
            emitter_obj = cast(Any, emitter_obj).replace(genome=genome_obj)

            # Create a wrapper objective function that calls problem.evaluate
            def obj_fn(nodes, conns):
                import jax
                import jax.numpy as jnp
                from tensorneat.common import State

                # Ensure nodes and conns have batch dimension
                if nodes.ndim == 2:  # Single network
                    nodes = jnp.expand_dims(nodes, 0)
                    conns = jnp.expand_dims(conns, 0)
                batch_size = nodes.shape[0]
                state = State(randkey=jax.random.PRNGKey(0))
                state = genome_obj.setup(state)
                # 1. Transform population
                # genome_obj.transform takes (state, nodes, conns)
                # We map over nodes and conns
                transformed_pop = jax.vmap(genome_obj.transform, in_axes=(None, 0, 0))(
                    state, nodes, conns
                )
                # 2. Evaluate population
                keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
                fitness = jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(
                    state, keys, genome_obj.forward, transformed_pop
                )
                # FIX: TensorNEAT problems natively return inverted fitness (-loss) for
                # its internal maximization loop. However, MalthusJAX engines expect the
                # raw objective value (e.g., positive loss) if `maximize=False`.
                if not maximize:
                    fitness = -fitness
                # TensorNEAT problems don't return descriptors, so we use dummy ones for QD
                descriptors = jnp.zeros((batch_size, kwargs.get("qdax_num_descriptors", 2)))
                return fitness, descriptors

            objective_fn = obj_fn
        evaluator = TensorNeatQDEvaluator(
            objective_function=objective_fn,
            config=BaseEvaluatorConfig(maximize=maximize),
            data=None,
        )
    else:
        # We use the BaseQDEvaluator composition to match standard evaluation
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
        num_desc = strategy.num_descriptors

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
    centroids = strategy.centroids
    if centroids is None:
        centroids = compute_cvt_centroids(
            num_descriptors=strategy.num_descriptors,
            num_init_cvt_samples=50000,
            num_centroids=strategy.num_centroids,
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
    from .adapters.map_elites_adapter import MapElitesEngineAdapter

    return MapElitesEngineAdapter(
        engine=engine,
        pop_size=pop_size,
        maximize=maximize,
        history_metrics=history_metrics,
        initial_population=kwargs.get("initial_population", None),
        centroids=centroids,
    )


def build_stub_engine(generations: int, **kwargs: Any) -> StubEngine:
    """Build StubEngine with legacy behavior for backward compatibility."""
    base_fitness = kwargs.get("base_fitness", 1.0)
    improvement_rate = kwargs.get("improvement_rate", 0.1)
    return StubEngine(
        generations=generations,
        base_fitness=base_fitness,
        improvement_rate=improvement_rate,
    )
