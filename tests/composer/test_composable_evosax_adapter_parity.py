import jax
import jax.numpy as jnp
import pytest
from malthusjax.composer.factory import build_evosax_engine, build_composable_evosax_engine

def test_evosax_adapters_parity():
    seed = 42
    
    legacy_engine = build_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec="sphere:dim=5,maximize=True",
        pop_size=20,
        generations=5,
        bounds=(-5.0, 5.0),
        seed=seed,
        maximize=True,
    )
    legacy_result = legacy_engine.run_once(jax.random.PRNGKey(seed))
    
    composable_engine = build_composable_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec="sphere:dim=5,maximize=True",
        pop_size=20,
        generations=5,
        bounds=(-5.0, 5.0),
        seed=seed,
        maximize=True,
        num_dims=5,
    )
    composable_result = composable_engine.run_once(jax.random.PRNGKey(seed))
    
    assert len(legacy_result.runs[0].metrics.get("best_fitness")) == 5
    assert len(composable_result.runs[0].metrics.get("best_fitness")) == 5
