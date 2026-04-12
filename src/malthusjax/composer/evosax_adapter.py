"""Evosax strategy adapter for the BenchmarkRunner.Engine protocol.

Wraps any evosax population-based strategy (ask/tell interface) so it can
be used interchangeably with GeneticEngineAdapter through Composer.quick_run().

Usage::

    from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig

    # build a MalthusJAX evaluator (could be any BaseEvaluator)
    evalr = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=10, seed=0))
    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evalr,
        pop_size=100,
        generations=200,
    )
    result = adapter.run_once(jax.random.PRNGKey(42))
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import chex
import evosax
import jax
import jax.numpy as jnp
import jax.random as jr
from evosax.algorithms import MR15_GA, DifferentialEvolution, SimpleGA

from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

EVOSAX_STRATEGIES: Dict[str, type] = {
    "SimpleGA": SimpleGA,
    "MR15_GA": MR15_GA,
    "DifferentialEvolution": DifferentialEvolution,
}


def list_strategies() -> list[str]:
    """Return available evosax strategy names."""
    return sorted(EVOSAX_STRATEGIES.keys())


class EvosaxEngineAdapter:
    """Adapter to make evosax strategies compatible with BenchmarkRunner.Engine protocol.

    Implements the same ``run_once(key) -> Dict`` contract as
    :class:`GeneticEngineAdapter` so both can be used interchangeably
    with :class:`BenchmarkRunner`.
    """

    def __init__(
        self,
        strategy: evosax.algorithms.population_based.PopulationBasedAlgorithm,
        params: Any,  # struct.dataclass with strategy hyper-parameters
        problem: evosax.problems.problem.Problem,
        problem_state: Any,  # struct.dataclass with problem state
        pop_size: int,
        num_generations: int,
        num_dims: int,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        maximize: bool = False,
        initial_population: chex.Array = None,
        prng_impl: Optional[str] = None,
    ) -> None:
        self.strategy = strategy
        self.params = params
        self.problem = problem
        self.problem_state = problem_state
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_dims = num_dims
        self.bounds = bounds
        self.maximize = maximize
        self.initial_population = initial_population
        self.prng_impl = prng_impl

    def run_once(
        self, key: chex.Array, unroll_factor: int = 1, compile: bool = True
    ) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results.

        Returns
        -------
        dict
            ``history`` : List[Dict] - per-generation stats
            ``summary`` : Dict       - final summary metrics
            ``timings`` : Dict       - wall-clock timing breakdown
        """

        key, key_pop, key_eval = jax.random.split(key, 3)

        if self.initial_population is not None:
            population_init = self.initial_population
        else:
            keys = jax.random.split(key_pop, self.pop_size)
            population_init = jax.vmap(self.problem.sample)(keys)

        fitness_init, prob_state_init, _ = self.problem.eval(
            key_eval, population_init, self.problem_state
        )

        if self.maximize:
            initial_best_idx = jnp.argmax(fitness_init)
            initial_best_fitness = fitness_init[initial_best_idx]
        else:
            initial_best_idx = jnp.argmin(fitness_init)
            initial_best_fitness = fitness_init[initial_best_idx]

        # Evosax algorithms natively minimize. If maximize is True, we must negate the fitness
        tell_fitness_init = -fitness_init if self.maximize else fitness_init

        def scan_step(carry: Tuple[Any, Any, Any], _: Any) -> Tuple[Tuple[Any, Any, Any], Any]:
            rng, state, p_state = carry
            rng, key_ask, key_eval_step, key_tell = jax.random.split(rng, 4)

            population, state = self.strategy.ask(key_ask, state, self.params)

            fitness, p_state, _ = self.problem.eval(key_eval_step, population, p_state)

            mean_fit = jnp.mean(fitness)
            std_fit = jnp.std(fitness)

            tell_fitness = -fitness if self.maximize else fitness

            state, metrics = self.strategy.tell(
                key_tell, population, tell_fitness, state, self.params
            )

            metrics = dict(metrics)  # copy to allow mutation
            metrics["mean_fitness"] = mean_fit
            metrics["std_fitness"] = std_fit

            return (rng, state, p_state), metrics

        def run_loop(rng: Any, pop_init: Any, fit_init: Any, p_state: Any) -> Tuple[Any, Any]:
            rng, key_init = jax.random.split(rng)
            state = self.strategy.init(key_init, pop_init, fit_init, self.params)

            carry = (rng, state, p_state)
            carry, metrics = jax.lax.scan(
                scan_step, carry, None, length=self.num_generations, unroll=unroll_factor
            )
            return carry[1], metrics

        start_time = time.perf_counter()
        if compile:
            run_loop = jax.jit(run_loop)

        compile_start = time.perf_counter()
        final_state, metrics = run_loop(key, population_init, tell_fitness_init, prob_state_init)

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_state)
        end_time = time.perf_counter()

        if False:
            print("DEBUG before flip metrics type", type(metrics))
            try:
                print("DEBUG keys", list(metrics.keys()))
            except Exception as e:
                print("DEBUG keys error", e)
            print("DEBUG bf values before flip", metrics.get("best_fitness", None))

        if self.maximize:

            def flip(x: chex.Array) -> chex.Array:
                return -x

            for key_name in ("best_fitness", "best_fitness_in_generation", "mean_fitness"):
                if key_name in metrics:
                    metrics[key_name] = flip(metrics[key_name])

        if False:
            print("DEBUG after flip best_fitness", metrics.get("best_fitness", None))

        history = []
        for g in range(self.num_generations):
            gen_stats = {}
            for k, v in metrics.items():
                val = v[g]
                if val.ndim == 0:
                    gen_stats[k] = val.item()
                else:
                    gen_stats[k] = val.tolist()
            gen_stats.setdefault("generation", g + 1)
            history.append(gen_stats)

        if history:
            best_fitness_value = history[-1].get("best_fitness")
        else:
            best_fitness_value = float(initial_best_fitness)
        summary = {
            "best_fitness": best_fitness_value,
            "best_solution": self.strategy.get_best_solution(final_state).tolist(),
            "total_generations": self.num_generations,
            "final_generation": self.num_generations,
            "total_evaluations": self.num_generations * self.pop_size,
            "pop_size": self.pop_size,
        }

        timings = {
            "initialization": compile_start - start_time,
            "evolution": end_time - compile_start,
            "total_time": end_time - start_time,
        }

        return {"history": history, "summary": summary, "timings": timings}


def build_evosax_engine(
    strategy_name: str = "SimpleGA",
    *,
    evaluator: Optional[BaseEvaluator[Any, Any, Any]] = None,
    fitness_spec: Optional[str] = None,
    pop_size: int = 50,
    generations: int = 100,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    maximize: bool = False,
    seed: int = 42,
    strategy_params: Optional[Dict[str, Any]] = None,
    initial_population: Any = None,
    prng_impl: Optional[str] = None,
    **kwargs: Any,
) -> EvosaxEngineAdapter:
    """Build an :class:`EvosaxEngineAdapter` from high-level specs.

    Parameters
    ----------
    strategy_name
        Name of the evosax strategy (``SimpleGA``, ``MR15_GA``,
        ``DifferentialEvolution``).
    evaluator
        A MalthusJAX :class:`BaseEvaluator` instance describing the
        fitness function.  If the object is a :class:`BBOBEvaluator` the
        underlying evosax problem is unwrapped automatically.  Support for
        other evaluator types is not yet implemented and will raise
        ``NotImplementedError``.
    fitness_spec
        Optional catalog-style spec that may override the configuration of a
        ``BBOBEvaluator`` when one is provided.  Has no effect for other
        evaluator types.
    pop_size
        Population size.
    generations
        Number of generations.
    bounds
        Search domain as ``(min, max)``.
    maximize
        If ``True``, flip the sign so the adapter reports fitness in
        maximisation convention (matching MalthusJAX default).
    seed
        Seed for the BBOB problem rotation/shift.
    strategy_params
        Optional dict of strategy-specific hyper-parameters that will be
        merged into ``strategy.default_params`` via ``.replace()``.

    Returns
    -------
    EvosaxEngineAdapter
        Ready to call ``.run_once(key)``.
    """
    if "num_generations" in kwargs:
        generations = int(kwargs.pop("num_generations"))

    if evaluator is None:
        raise ValueError("build_evosax_engine requires an evaluator argument")

    if fitness_spec is not None and isinstance(evaluator, BBOBEvaluator):
        from .catalog import OperatorCatalog

        cat = OperatorCatalog()
        parsed_name, parsed_params = cat.parse_spec(fitness_spec)
        fn = parsed_params.get("fn_name", parsed_name)
        dims = parsed_params.get("dim", parsed_params.get("num_dims"))
        if dims is None:
            dims = evaluator.config.num_dims
        if "seed" in parsed_params:
            seed = parsed_params["seed"]
        if "maximize" in parsed_params:
            maximize = parsed_params["maximize"]
        evaluator = BBOBEvaluator.create(
            BBOBConfig(fn_name=fn, num_dims=dims, seed=seed, maximize=maximize)
        )

    if strategy_name not in EVOSAX_STRATEGIES:
        raise KeyError(f"Unknown evosax strategy '{strategy_name}'. Available: {list_strategies()}")

    rng = jr.PRNGKey(seed)

    if isinstance(evaluator, BBOBEvaluator):
        problem = evaluator.evosax_problem
        problem_state = evaluator.problem_state
        num_dims = evaluator.config.num_dims
    else:
        raise NotImplementedError(
            "Only BBOBEvaluator instances are currently supported by the "
            "evosax adapter. Generic BaseEvaluator support is not implemented yet."
        )

    init_solution = jr.uniform(rng, (num_dims,), minval=bounds[0], maxval=bounds[1])

    strategy_cls = EVOSAX_STRATEGIES[strategy_name]
    strategy = strategy_cls(population_size=pop_size, solution=init_solution)

    params = strategy.default_params
    if strategy_params:
        params = params.replace(**strategy_params)

    return EvosaxEngineAdapter(
        strategy=strategy,
        params=params,
        problem=problem,
        problem_state=problem_state,
        pop_size=pop_size,
        num_generations=generations,
        num_dims=num_dims,
        bounds=bounds,
        maximize=maximize,
        initial_population=initial_population,
        prng_impl=prng_impl,
    )
