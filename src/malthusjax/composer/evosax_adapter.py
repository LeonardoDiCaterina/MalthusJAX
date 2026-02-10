"""Evosax strategy adapter for the BenchmarkRunner.Engine protocol.

Wraps any evosax population-based strategy (ask/tell interface) so it can
be used interchangeably with GeneticEngineAdapter through Composer.quick_run().

Usage::

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec="sphere:dim=10",
        pop_size=100,
        generations=200,
    )
    result = adapter.run_once(jax.random.PRNGKey(42))
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from evosax.algorithms.population_based import (
    DifferentialEvolution,
    MR15_GA,
    SimpleGA,
)
from evosax.problems import BBOBProblem

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

EVOSAX_STRATEGIES: Dict[str, type] = {
    "SimpleGA": SimpleGA,
    "MR15_GA": MR15_GA,
    "DifferentialEvolution": DifferentialEvolution,
}


def list_strategies() -> list[str]:
    """Return available evosax strategy names."""
    return sorted(EVOSAX_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class EvosaxEngineAdapter:
    """Adapter to make evosax strategies compatible with BenchmarkRunner.Engine protocol.

    Implements the same ``run_once(key) -> Dict`` contract as
    :class:`GeneticEngineAdapter` so both can be used interchangeably
    with :class:`BenchmarkRunner`.
    """

    def __init__(
        self,
        strategy: Any,
        params: Any,
        problem: BBOBProblem,
        pop_size: int,
        num_generations: int,
        num_dims: int,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        maximize: bool = False,
        initial_population: Any = None,
    ) -> None:
        self.strategy = strategy
        self.params = params
        self.problem = problem
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_dims = num_dims
        self.bounds = bounds
        self.maximize = maximize
        self.initial_population = initial_population

    # -- BenchmarkRunner.Engine protocol ------------------------------------

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results.

        Returns
        -------
        dict
            ``history`` : List[Dict] - per-generation stats
            ``summary`` : Dict       - final summary metrics
            ``timings`` : Dict       - wall-clock timing breakdown
        """
        # ---- 1. Initialisation -------------------------------------------
        t_init_start = time.perf_counter()

        k_init, k_run = jr.split(key)
        p_state = self.problem.init(k_init)

        # Allow externally provided initial population for reproducible
        # comparisons. If not provided, sample uniformly from bounds.
        if self.initial_population is not None:
            init_x = jnp.asarray(self.initial_population)
        else:
            init_x = jr.uniform(
                k_init,
                (self.pop_size, self.num_dims),
                minval=self.bounds[0],
                maxval=self.bounds[1],
            )

        init_fit = jnp.full((self.pop_size,), jnp.inf)

        state = self.strategy.init(k_init, init_x, init_fit, self.params)

        # If an explicit initial population was supplied, evaluate it and
        # tell the strategy so the internal best_fitness is updated.
        if self.initial_population is not None:
            fitness, p_state, _ = self.problem.eval(k_init, init_x, p_state)
            state, _metrics = self.strategy.tell(k_init, init_x, fitness, state, self.params)

        t_init_end = time.perf_counter()

        # ---- 2. Evolution loop -------------------------------------------
        t_evo_start = time.perf_counter()

        history: list[Dict[str, Any]] = []
        rng = k_run

        for gen in range(self.num_generations):
            rng, rng_step = jr.split(rng)

            # ask  →  tell
            x, state = self.strategy.ask(rng_step, state, self.params)
            fitness, p_state, _ = self.problem.eval(rng_step, x, p_state)
            state, _metrics = self.strategy.tell(
                rng_step, x, fitness, state, self.params
            )

            # evosax minimises; flip sign when Composer expects maximisation
            best_f = float(state.best_fitness)
            mean_f = float(jnp.mean(fitness))
            std_f = float(jnp.std(fitness))

            if self.maximize:
                best_f = -best_f
                mean_f = -mean_f

            history.append(
                {
                    "generation": gen + 1,
                    "best_fitness": best_f,
                    "mean_fitness": mean_f,
                    "std_fitness": std_f,
                }
            )

        t_evo_end = time.perf_counter()

        # ---- 3. Assemble output ------------------------------------------
        final_best = float(state.best_fitness)
        if self.maximize:
            final_best = -final_best

        summary: Dict[str, Any] = {
            "best_fitness": final_best,
            "final_generation": self.num_generations,
            "total_evaluations": self.num_generations * self.pop_size,
            "stagnation_counter": 0,
        }

        timings: Dict[str, float] = {
            "initialization": t_init_end - t_init_start,
            "evolution": t_evo_end - t_evo_start,
        }

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_evosax_engine(
    strategy_name: str = "SimpleGA",
    fitness_spec: Optional[str] = None,
    problem_name: str = "sphere",
    num_dims: int = 10,
    pop_size: int = 50,
    generations: int = 100,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    maximize: bool = False,
    seed: int = 42,
    strategy_params: Optional[Dict[str, Any]] = None,
    initial_population: Any = None,
    **kwargs: Any,
) -> EvosaxEngineAdapter:
    """Build an :class:`EvosaxEngineAdapter` from high-level specs.

    Parameters
    ----------
    strategy_name
        Name of the evosax strategy (``SimpleGA``, ``MR15_GA``,
        ``DifferentialEvolution``).
    fitness_spec
        Optional catalog-style spec that overrides *problem_name* and
        *num_dims*, e.g. ``"sphere:dim=10"`` or ``"rastrigin:dim=5"``.
    problem_name
        BBOB problem name (used when *fitness_spec* is ``None``).
    num_dims
        Dimensionality of the search space.
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
    # ---- handle common aliases -----------------------------------------
    if "num_generations" in kwargs:
        generations = int(kwargs.pop("num_generations"))

    # ---- resolve fitness spec if provided --------------------------------
    if fitness_spec is not None:
        from .catalog import OperatorCatalog

        cat = OperatorCatalog()
        parsed_name, parsed_params = cat.parse_spec(fitness_spec)
        # Map catalog names to BBOB names
        problem_name = parsed_params.get("fn_name", parsed_name)
        num_dims = parsed_params.get("dim", parsed_params.get("num_dims", num_dims))
        if "seed" in parsed_params:
            seed = parsed_params["seed"]
        if "maximize" in parsed_params:
            maximize = parsed_params["maximize"]

    # ---- strategy --------------------------------------------------------
    if strategy_name not in EVOSAX_STRATEGIES:
        raise KeyError(
            f"Unknown evosax strategy '{strategy_name}'. "
            f"Available: {list_strategies()}"
        )

    rng = jr.PRNGKey(seed)
    problem = BBOBProblem(problem_name, num_dims=num_dims, seed=seed)
    init_solution = problem.sample(rng)

    strategy_cls = EVOSAX_STRATEGIES[strategy_name]
    strategy = strategy_cls(population_size=pop_size, solution=init_solution)

    params = strategy.default_params
    if strategy_params:
        params = params.replace(**strategy_params)

    return EvosaxEngineAdapter(
        strategy=strategy,
        params=params,
        problem=problem,
        pop_size=pop_size,
        num_generations=generations,
        num_dims=num_dims,
        bounds=bounds,
        maximize=maximize,
        initial_population=initial_population,
    )
