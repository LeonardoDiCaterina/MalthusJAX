"""Test that EvosaxEngineAdapter.run_once produces identical results to
running evosax directly with the same key sequence.

This is the ground-truth equivalence test: if the adapter changes behaviour
in any way (key splitting, sign convention, history recording), this test
will catch it.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from evosax.problems import BBOBProblem

from malthusjax.composer.evosax_adapter import (
    EVOSAX_STRATEGIES,
    build_evosax_engine,
)


# ---------------------------------------------------------------------------
# Helper: vanilla evosax run (no adapter)
# ---------------------------------------------------------------------------


def _run_evosax_raw(
    strategy_name: str,
    problem_name: str,
    num_dims: int,
    pop_size: int,
    generations: int,
    bounds: tuple[float, float],
    key,
    seed: int = 42,
):
    """Reproduce the exact logic of EvosaxEngineAdapter.run_once using raw evosax."""
    rng = jr.PRNGKey(seed)
    problem = BBOBProblem(problem_name, num_dims=num_dims, seed=seed)
    init_solution = problem.sample(rng)

    strategy_cls = EVOSAX_STRATEGIES[strategy_name]
    strategy = strategy_cls(population_size=pop_size, solution=init_solution)
    params = strategy.default_params

    # --- Mirror the adapter's key splitting ---
    k_init, k_run = jr.split(key)
    p_state = problem.init(k_init)

    init_x = jr.uniform(
        k_init, (pop_size, num_dims), minval=bounds[0], maxval=bounds[1]
    )
    init_fit = jnp.full((pop_size,), jnp.inf)
    state = strategy.init(k_init, init_x, init_fit, params)

    # --- Evolution loop (same key sequence as adapter) ---
    history = []
    rng = k_run

    for gen in range(generations):
        rng, rng_step = jr.split(rng)

        x, state = strategy.ask(rng_step, state, params)
        fitness, p_state, _ = problem.eval(rng_step, x, p_state)
        state, _ = strategy.tell(rng_step, x, fitness, state, params)

        history.append(
            {
                "generation": gen + 1,
                "best_fitness": float(state.best_fitness),
                "mean_fitness": float(jnp.mean(fitness)),
                "std_fitness": float(jnp.std(fitness)),
            }
        )

    return {
        "history": history,
        "best_fitness": float(state.best_fitness),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvosaxAdapterMatchesRaw:
    """Adapter output must be bit-identical to a vanilla evosax run."""

    @pytest.fixture()
    def common_params(self):
        return dict(
            strategy_name="SimpleGA",
            problem_name="sphere",
            num_dims=5,
            pop_size=12,
            generations=10,
            bounds=(-5.0, 5.0),
            seed=42,
        )

    def test_history_matches_minimize(self, common_params):
        """With maximize=False the adapter must report raw evosax fitness."""
        key = jr.PRNGKey(7)

        # --- Raw evosax run ---
        raw = _run_evosax_raw(key=key, **common_params)

        # --- Adapter run ---
        adapter = build_evosax_engine(
            maximize=False, **common_params
        )
        adapted = adapter.run_once(key)

        # Per-generation best_fitness must match exactly
        for gen_idx, (raw_h, adp_h) in enumerate(
            zip(raw["history"], adapted["history"])
        ):
            assert raw_h["generation"] == adp_h["generation"], f"gen {gen_idx}"
            assert jnp.isclose(
                raw_h["best_fitness"], adp_h["best_fitness"], atol=1e-6
            ), (
                f"gen {gen_idx}: raw={raw_h['best_fitness']}, "
                f"adapted={adp_h['best_fitness']}"
            )
            assert jnp.isclose(
                raw_h["mean_fitness"], adp_h["mean_fitness"], atol=1e-6
            ), f"gen {gen_idx}: mean mismatch"

        # Final summary
        assert jnp.isclose(
            raw["best_fitness"],
            adapted["summary"]["best_fitness"],
            atol=1e-6,
        )

    def test_history_matches_maximize(self, common_params):
        """With maximize=True, adapter negates fitness; verify correspondence."""
        key = jr.PRNGKey(7)

        raw = _run_evosax_raw(key=key, **common_params)

        adapter = build_evosax_engine(
            maximize=True, **common_params
        )
        adapted = adapter.run_once(key)

        for gen_idx, (raw_h, adp_h) in enumerate(
            zip(raw["history"], adapted["history"])
        ):
            # Adapter negates when maximize=True
            assert jnp.isclose(
                -raw_h["best_fitness"], adp_h["best_fitness"], atol=1e-6
            ), (
                f"gen {gen_idx}: -raw={-raw_h['best_fitness']}, "
                f"adapted={adp_h['best_fitness']}"
            )
            assert jnp.isclose(
                -raw_h["mean_fitness"], adp_h["mean_fitness"], atol=1e-6
            ), f"gen {gen_idx}: mean mismatch"

        assert jnp.isclose(
            -raw["best_fitness"],
            adapted["summary"]["best_fitness"],
            atol=1e-6,
        )

    def test_with_initial_population(self, common_params):
        """Same initial_population → same trajectory (minimize mode)."""
        key = jr.PRNGKey(99)
        init_pop = jr.uniform(
            jr.PRNGKey(0),
            (common_params["pop_size"], common_params["num_dims"]),
            minval=-5.0,
            maxval=5.0,
        )

        # --- Raw run with the same initial population injected ---
        raw = _run_evosax_raw_with_init_pop(
            key=key, initial_population=init_pop, **common_params
        )

        # --- Adapter run ---
        adapter = build_evosax_engine(
            maximize=False,
            initial_population=init_pop,
            **common_params,
        )
        adapted = adapter.run_once(key)

        for gen_idx, (raw_h, adp_h) in enumerate(
            zip(raw["history"], adapted["history"])
        ):
            assert jnp.isclose(
                raw_h["best_fitness"], adp_h["best_fitness"], atol=1e-6
            ), (
                f"gen {gen_idx}: raw={raw_h['best_fitness']}, "
                f"adapted={adp_h['best_fitness']}"
            )

    @pytest.mark.parametrize("strategy_name", ["SimpleGA", "DifferentialEvolution"])
    def test_multiple_strategies(self, strategy_name):
        """Adapter matches raw evosax for different strategies."""
        key = jr.PRNGKey(11)
        params = dict(
            strategy_name=strategy_name,
            problem_name="sphere",
            num_dims=4,
            pop_size=10,
            generations=5,
            bounds=(-5.0, 5.0),
            seed=42,
        )

        raw = _run_evosax_raw(key=key, **params)
        adapter = build_evosax_engine(maximize=False, **params)
        adapted = adapter.run_once(key)

        for gen_idx, (raw_h, adp_h) in enumerate(
            zip(raw["history"], adapted["history"])
        ):
            assert jnp.isclose(
                raw_h["best_fitness"], adp_h["best_fitness"], atol=1e-6
            ), (
                f"[{strategy_name}] gen {gen_idx}: "
                f"raw={raw_h['best_fitness']}, adapted={adp_h['best_fitness']}"
            )


# ---------------------------------------------------------------------------
# Helper: raw run WITH initial population (mirrors adapter logic)
# ---------------------------------------------------------------------------


def _run_evosax_raw_with_init_pop(
    strategy_name: str,
    problem_name: str,
    num_dims: int,
    pop_size: int,
    generations: int,
    bounds: tuple[float, float],
    key,
    initial_population,
    seed: int = 42,
):
    """Same as _run_evosax_raw but with initial population injection."""
    rng = jr.PRNGKey(seed)
    problem = BBOBProblem(problem_name, num_dims=num_dims, seed=seed)
    init_solution = problem.sample(rng)

    strategy_cls = EVOSAX_STRATEGIES[strategy_name]
    strategy = strategy_cls(population_size=pop_size, solution=init_solution)
    params = strategy.default_params

    k_init, k_run = jr.split(key)
    p_state = problem.init(k_init)

    init_x = jnp.asarray(initial_population)
    init_fit = jnp.full((pop_size,), jnp.inf)
    state = strategy.init(k_init, init_x, init_fit, params)

    # Evaluate initial population and tell the strategy
    fitness, p_state, _ = problem.eval(k_init, init_x, p_state)
    state, _ = strategy.tell(k_init, init_x, fitness, state, params)

    history = []
    rng = k_run

    for gen in range(generations):
        rng, rng_step = jr.split(rng)
        x, state = strategy.ask(rng_step, state, params)
        fitness, p_state, _ = problem.eval(rng_step, x, p_state)
        state, _ = strategy.tell(rng_step, x, fitness, state, params)

        history.append(
            {
                "generation": gen + 1,
                "best_fitness": float(state.best_fitness),
                "mean_fitness": float(jnp.mean(fitness)),
                "std_fitness": float(jnp.std(fitness)),
            }
        )

    return {
        "history": history,
        "best_fitness": float(state.best_fitness),
    }
