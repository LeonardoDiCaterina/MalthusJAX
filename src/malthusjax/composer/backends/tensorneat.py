"""TensorNEAT backend provider — topology and neural evolution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax.random as jr

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.strategies.base import BaseStrategy


def resolve_tensorneat_problem(
    problem_name: Optional[str], fitness_spec: Optional[str]
) -> Tuple[Any, Any]:
    import inspect

    import tensorneat.problem

    name = problem_name or (fitness_spec if isinstance(fitness_spec, str) else "xor")
    if name.startswith("tensorneat:"):
        name = name[len("tensorneat:") :]
    if name.startswith("problem="):
        name = name[len("problem=") :]
    base_name = name.split(":")[0].lower()
    problem_cls: Any = None
    for cls_name, cls_obj in inspect.getmembers(tensorneat.problem, inspect.isclass):
        if cls_name.lower() == base_name:
            problem_cls = cls_obj
            break
    if problem_cls is None:
        try:
            from lsp.evaluator.tensorneat_bridge import TensorNEATSupervisedProblem

            from malthusjax.composer.catalog import OperatorCatalog

            if not isinstance(fitness_spec, str):
                raise ValueError(
                    "fitness_spec must be a string to resolve a MalthusJAX evaluator fallback"
                )
            mjax_evaluator = OperatorCatalog().get(fitness_spec)
            problem = TensorNEATSupervisedProblem(mjax_evaluator)
            return problem, problem.setup()
        except Exception as fallback_e:
            raise ValueError(
                f"Unknown TensorNEAT problem: {base_name}. Fallback failed: {fallback_e}"
            )
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


def build_tensorneat_engine(
    strategy: Any,
    fitness_spec: Optional[Any],
    pop_size: int,
    generations: int,
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    import inspect

    import tensorneat.algorithm
    import tensorneat.genome

    from malthusjax.composer.tensorneat_adapter import (
        build_tensorneat_engine as adapter_build_tensorneat_engine,
    )

    algorithm_cls: Any = None
    for name, obj in inspect.getmembers(tensorneat.algorithm, inspect.isclass):
        if name.lower() == strategy.algorithm_name.lower():
            algorithm_cls = obj
            break
    if algorithm_cls is None:
        raise ValueError(f"Unknown TensorNEAT algorithm: {strategy.algorithm_name}")

    genome_cls: Any = None
    target_genome = strategy.genome_name.lower()
    for name, obj in inspect.getmembers(tensorneat.genome, inspect.isclass):
        name_lower = name.lower()
        if name_lower == target_genome or name_lower == f"{target_genome}genome":
            genome_cls = obj
            break
    if genome_cls is None:
        raise ValueError(f"Unknown TensorNEAT genome: {strategy.genome_name}")

    alg_kwargs = strategy.algorithm_kwargs.copy()
    init_pop = alg_kwargs.pop("initial_population", kwargs.get("initial_population", None))
    pop_size = alg_kwargs.pop("pop_size", pop_size)
    genome_sig = inspect.signature(genome_cls.__init__)
    genome_params = set(genome_sig.parameters.keys()) - {"self", "num_inputs", "num_outputs"}
    genome_kwargs = {p: alg_kwargs.pop(p) for p in list(alg_kwargs.keys()) if p in genome_params}
    genome = genome_cls(
        num_inputs=strategy.num_inputs, num_outputs=strategy.num_outputs, **genome_kwargs
    )
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


def build_composable_tensorneat_engine(
    strategy: Any,
    fitness_spec: Optional[Any],
    pop_size: int,
    generations: int,
    maximize: bool,
    history_metrics: Optional[Sequence[str]],
    **kwargs: Any,
) -> Any:
    import inspect

    import tensorneat.algorithm
    import tensorneat.genome

    from malthusjax.composer.composable_tensor_neat_adapter import (
        build_composable_tensorneat_engine as adapter_build_composable_tensorneat_engine,
    )

    if not isinstance(fitness_spec, str) and fitness_spec is not None:
        evaluator = fitness_spec
        if hasattr(evaluator, "evosax_problem"):
            problem = evaluator.evosax_problem
            problem_state = evaluator.problem_state
        else:
            problem = None
            problem_state = None
    else:
        problem, problem_state = resolve_tensorneat_problem(strategy.problem_name, fitness_spec)
        evaluator = None

    num_inputs = strategy.num_inputs
    num_outputs = strategy.num_outputs

    if (
        (num_inputs is None or num_outputs is None)
        and evaluator is not None
        and hasattr(evaluator, "interpreter")
    ):
        num_inputs = num_inputs or evaluator.interpreter.input_dim
        num_outputs = num_outputs or evaluator.interpreter.output_dim

    if num_inputs is None or num_outputs is None:
        num_inputs = num_inputs or 2
        num_outputs = num_outputs or 1

    algorithm_cls: Any = None
    for name, obj in inspect.getmembers(tensorneat.algorithm, inspect.isclass):
        if name.lower() == strategy.algorithm_name.lower():
            algorithm_cls = obj
            break
    if algorithm_cls is None:
        raise ValueError(f"Unknown TensorNEAT algorithm: {strategy.algorithm_name}")

    genome_cls: Any = None
    target_genome = strategy.genome_name.lower()
    for name, obj in inspect.getmembers(tensorneat.genome, inspect.isclass):
        name_lower = name.lower()
        if name_lower == target_genome or name_lower == f"{target_genome}genome":
            genome_cls = obj
            break
    if genome_cls is None:
        raise ValueError(f"Unknown TensorNEAT genome: {strategy.genome_name}")

    alg_kwargs = strategy.algorithm_kwargs.copy()
    init_pop = alg_kwargs.pop("initial_population", kwargs.get("initial_population", None))
    pop_size = alg_kwargs.pop("pop_size", pop_size)
    genome_sig = inspect.signature(genome_cls.__init__)
    genome_params = set(genome_sig.parameters.keys()) - {"self", "num_inputs", "num_outputs"}
    genome_kwargs = {p: alg_kwargs.pop(p) for p in list(alg_kwargs.keys()) if p in genome_params}
    genome = genome_cls(num_inputs=num_inputs, num_outputs=num_outputs, **genome_kwargs)
    algorithm = algorithm_cls(pop_size=pop_size, genome=genome, **alg_kwargs)

    return adapter_build_composable_tensorneat_engine(
        algorithm=algorithm,
        evaluator=evaluator or (problem, problem_state),
        generations=generations,
        pop_size=pop_size,
        maximize=maximize,
        history_metrics=history_metrics,
        initial_population=init_pop,
    )


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
                "genome",
                "engine_type",
                "genome_type",
                "selection",
                "crossover",
                "mutation",
                "elitism",
                "experiment_name",
                "output_dir",
                "engine",
                "trace_dir",
                "eval_mode",
                "evosax_strategy",
                "evosax_es_params",
                "qdax_strategy",
                "qdax_num_descriptors",
                "qdax_num_centroids",
                "qdax_mutation_sigma",
                "seed",
                "num_dims",
                "num_parallel",
                "num_inputs",
                "num_outputs",
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

    def handles_strategy(self, strategy: Any) -> bool:
        from malthusjax.composer.strategies.core import TensorNEATStrategy

        return isinstance(strategy, TensorNEATStrategy) or (
            isinstance(strategy, str) and "tensorneat" in strategy
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
        if isinstance(strategy, str):
            parts = strategy.split(":")
            strat_kwargs = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    strat_kwargs[k] = v
            from malthusjax.composer.strategies.core import TensorNEATStrategy

            strat_pop_size = strat_kwargs.pop("pop_size", pop_size)
            strategy = TensorNEATStrategy(
                algorithm_name=strat_kwargs.pop("algorithm", "neat"),
                genome_name=strat_kwargs.pop("genome", "default"),
                problem_name=strat_kwargs.pop("problem", None),
                num_inputs=strat_kwargs.pop("num_inputs", 2),
                num_outputs=strat_kwargs.pop("num_outputs", 1),
                algorithm_kwargs=strat_kwargs,
            )
            pop_size = strat_pop_size
        elif hasattr(strategy, "algorithm_kwargs") and "pop_size" in getattr(
            strategy, "algorithm_kwargs", {}
        ):
            from dataclasses import replace

            strat_kwargs = dict(strategy.algorithm_kwargs)
            pop_size = strat_kwargs.pop("pop_size")
            strategy = replace(strategy, algorithm_kwargs=strat_kwargs)

        if composable:
            builder = build_composable_tensorneat_engine
        else:
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
