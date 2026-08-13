"""
Consolidated tests for PRNG and Key Derivation Strategy features.
"""


import chex
import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.random import PRNGImpl, create_key, is_new_style_key
from malthusjax.engine.resource_mapper import KeyDerivationStrategy, compute_resource_map
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


def test_legacy_key_detection():
    key_legacy = jax.random.PRNGKey(42)
    key_new = create_key(42, impl=PRNGImpl.THREEFRY)

    assert not is_new_style_key(key_legacy)
    assert is_new_style_key(key_new)


def test_init_state_accepts_int_seed(make_engine):
    engine = make_engine()
    state = engine.init_state(42)
    assert state.rng_key is not None


def test_init_state_accepts_key_object(make_engine, prng_key):
    engine = make_engine()
    state = engine.init_state(prng_key)
    assert state.rng_key is not None


def test_resource_map_accepts_strategy():
    pop_size = 40
    genome_config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=4)
    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

    rmap_split = compute_resource_map(
        selection,
        crossover,
        mutation,
        genome_config,
        pop_size,
        key_derivation=KeyDerivationStrategy.SPLIT,
    )
    rmap_fold = compute_resource_map(
        selection,
        crossover,
        mutation,
        genome_config,
        pop_size,
        key_derivation=KeyDerivationStrategy.FOLD,
    )

    assert rmap_split.key_derivation == KeyDerivationStrategy.SPLIT
    assert rmap_fold.key_derivation == KeyDerivationStrategy.FOLD


@pytest.mark.parametrize("strategy", [KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD])
def test_resource_mapper_get_keys(strategy, prng_key):
    pop_size = 40
    genome_config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=4)
    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

    rmap = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size, key_derivation=strategy
    )
    keys = rmap.get_keys(prng_key)

    assert keys.shape[0] == rmap.total_rng_budget
    assert keys.shape[1:] == prng_key.shape


def test_reproducibility_same_seed(make_engine):
    engine = make_engine()
    k1 = create_key(1234, impl=PRNGImpl.THREEFRY)
    k2 = create_key(1234, impl=PRNGImpl.THREEFRY)

    s1 = engine.init_state(k1)
    f1, _, _ = engine.run(s1, compile=False)

    s2 = engine.init_state(k2)
    f2, _, _ = engine.run(s2, compile=False)

    chex.assert_trees_all_close(f1.population.genes.values, f2.population.genes.values)


def test_different_impl_produces_different_run(make_engine):
    try:
        k1 = create_key(1234, impl=PRNGImpl.THREEFRY)
        k2 = create_key(1234, impl=PRNGImpl.PHILOX)
    except ValueError:
        pytest.skip("PHILOX not supported")

    engine = make_engine()
    s1 = engine.init_state(k1)
    f1, _, _ = engine.run(s1, compile=False)

    s2 = engine.init_state(k2)
    f2, _, _ = engine.run(s2, compile=False)

    assert not jnp.allclose(f1.population.genes.values, f2.population.genes.values)


def test_allocate_entropy_preserves_impl(make_engine, prng_key):
    engine = make_engine()
    state = engine.init_state(prng_key)

    k_sel, k_cross, k_mut, k_next = engine._allocate_entropy(state)

    # Check that shapes are correct, implying preservation of underlying impl's data shape (usually 2 uint32s)
    assert k_sel.shape[-1] == 2
    assert k_cross.shape[-1] == 2
    assert k_mut.shape[-1] == 2
    assert k_next.shape == (2,)


def test_jit_does_not_break_reproducibility(make_engine):
    k1 = create_key(2023, impl=PRNGImpl.THREEFRY)
    k2 = create_key(2023, impl=PRNGImpl.THREEFRY)

    engine = make_engine()
    s1 = engine.init_state(k1)
    f1, _, _ = engine.run(s1, compile=True)

    s2 = engine.init_state(k2)
    f2, _, _ = engine.run(s2, compile=True)

    chex.assert_trees_all_close(f1.population.genes.values, f2.population.genes.values)
