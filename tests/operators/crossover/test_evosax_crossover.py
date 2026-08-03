import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
import chex

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper


@pytest.fixture
def evosax_cx_setup(prng_key):
    config = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
    pop_size = 4

    k1, k2 = jar.split(prng_key)
    parents_1 = RealPopulation.init_random(k1, config, pop_size)
    p2_genes = RealGenome(values=jnp.full((pop_size, config.shape[0]), 1.0))
    parents_2 = parents_1.spawn_offspring(p2_genes)
    
    return config, pop_size, parents_1, parents_2


@pytest.mark.parametrize("num_offspring", [1, 2])
@pytest.mark.parametrize("crossover_rate", [0.5, 0.7])
@pytest.mark.parametrize("injection_mode", [True, False])
def test_evosax_uniform_crossover_wrapper(evosax_cx_setup, prng_key, num_offspring, crossover_rate, injection_mode):
    config, pop_size, parents_1, parents_2 = evosax_cx_setup
    
    wrapper = EvosaxUniformCrossoverWrapper(
        num_offspring=num_offspring,
        crossover_rate=crossover_rate,
        injection_mode=injection_mode,
    ).set_input_length(pop_size)
    
    n_keys = wrapper.num_keys(input_shape=(pop_size,))
    if wrapper.injection_mode:
        expected_n = 1
    else:
        expected_n = int(pop_size * num_offspring * wrapper.num_keys_per_atomic_operation)
    assert n_keys == expected_n

    keys = jar.split(prng_key, n_keys)

    jit_op = jax.jit(wrapper)
    offspring = jit_op(keys, parents_1, parents_2, config)

    if wrapper.injection_mode:
        base_key = keys[0]
        subkeys = jar.split(base_key, pop_size * num_offspring)
    else:
        flat = keys.reshape((-1, keys.shape[-1]))
        subkeys = flat  # already one key per offspring

    def per_row(k):
        return jar.bernoulli(k, p=crossover_rate, shape=config.shape)

    masks = jax.vmap(per_row)(subkeys)
    masks = masks.reshape((pop_size, num_offspring) + masks.shape[1:])

    def per_pair(mask_block, p1, p2):
        return jax.vmap(lambda m: jnp.where(m, p2.values, p1.values), in_axes=0)(mask_block)

    vals = jax.vmap(per_pair)(masks, parents_1.genes, parents_2.genes)
    expected = vals.reshape((-1,) + vals.shape[2:])

    chex.assert_trees_all_close(offspring.genes.values, expected)


@pytest.mark.parametrize("injection_mode", [True, False])
def test_no_keys_raises(evosax_cx_setup, injection_mode):
    config, pop_size, parents_1, parents_2 = evosax_cx_setup
    empty_keys = jnp.empty((0, 2), dtype=jnp.uint32)
    wrapper = EvosaxUniformCrossoverWrapper(
        num_offspring=1, crossover_rate=0.5, injection_mode=injection_mode
    )
    with pytest.raises(ValueError):
        wrapper(empty_keys, parents_1, parents_2, config)
