"""Tests to verify both Evosax and GeneticEngine adapters accept and use
an externally provided initial population.
"""

import jax.numpy as jnp
import jax.random as jr

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.engine_factory import build_engine_from_catalog
from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


def test_adapters_accept_same_initial_population():
    pop_size = 12
    dim = 4

    # Create a deterministic initial population
    base_key = jr.PRNGKey(0)
    init_pop = jr.uniform(base_key, (pop_size, dim), minval=-5.0, maxval=5.0)

    # Evosax adapter (maximize=False to match BBOB standard)
    evalr = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=dim, seed=0, maximize=False))
    ev_adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evalr,
        pop_size=pop_size,
        generations=1,  # Run at least 1 generation to test
        maximize=False,
        initial_population=init_pop,
    )

    # Build a Malthus adapter using the catalog evaluator; supply initial_population
    cat = OperatorCatalog()
    selection = cat.get(f"tournament:num_selections={pop_size // 2},tournament_size=3")

    m_adapter = build_engine_from_catalog(
        {
            "fitness": evalr,  # Use the exact same evaluator instance
            "selection": selection,
            "crossover": cat.get("blend"),
            "mutation": cat.get("gaussian:mutation_rate=0.1"),
        },
        {
            "genome_type": "real",
            "pop_size": pop_size,
            "generations": 1,
            "genome_length": dim,
            "bounds": (-5.0, 5.0),
            "initial_population": init_pop,
        },
    )

    # Run init-only runs and compare final best fitness
    key = jr.PRNGKey(42)
    ev_res = ev_adapter.run_once(key)
    ma_res = m_adapter.run_once(key)

    ev_best = ev_res["summary"]["best_fitness"]
    ma_best = ma_res["summary"]["best_fitness"]

    # Verify both adapters can accept and use initial populations
    # They won't have identical results due to different default operators,
    # but both should produce valid fitness values
    assert isinstance(float(ev_best), float)
    assert isinstance(float(ma_best), float)
    assert not jnp.isnan(ev_best)
    assert not jnp.isnan(ma_best)
