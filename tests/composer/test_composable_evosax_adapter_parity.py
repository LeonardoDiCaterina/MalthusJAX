import jax
import numpy as np
import pytest

from malthusjax.composer.backends.evosax import (
    build_composable_evosax_engine,
    build_evosax_engine,
)


@pytest.mark.parametrize("maximize", [True, False])
def test_evosax_adapters_parity(maximize: bool):
    seed = 42

    legacy_engine = build_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec=f"sphere:dim=5,maximize={maximize}",
        pop_size=20,
        generations=5,
        bounds=(-5.0, 5.0),
        seed=seed,
        maximize=maximize,
        num_dims=5,
    )
    legacy_result = legacy_engine.run_once(jax.random.PRNGKey(seed))

    composable_engine = build_composable_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec=f"sphere:dim=5,maximize={maximize}",
        pop_size=20,
        generations=5,
        bounds=(-5.0, 5.0),
        seed=seed,
        maximize=maximize,
        num_dims=5,
    )
    composable_result = composable_engine.run_once(jax.random.PRNGKey(seed))

    assert len(legacy_result["history"]) == 5
    assert len(composable_result["history"]) == 5

    # Verify identical generational trajectories across metrics
    for metric_key in ("best_fitness", "mean_fitness", "std_fitness"):
        leg_vals = [h[metric_key] for h in legacy_result["history"]]
        comp_vals = [h[metric_key] for h in composable_result["history"]]
        np.testing.assert_allclose(comp_vals, leg_vals, rtol=1e-5, atol=1e-5)
