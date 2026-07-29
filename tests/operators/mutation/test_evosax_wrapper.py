import pytest
import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper


@pytest.fixture
def evosax_mut_setup(prng_key):
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    pop_size = 6
    population = RealPopulation.init_random(prng_key, config, pop_size)
    return config, pop_size, population


@pytest.mark.parametrize("injection_mode", [True, False])
def test_evosex_wrapper_mutation(evosax_mut_setup, prng_key, injection_mode):
    config, pop_size, population = evosax_mut_setup
    
    mutator = EvosaxGaussianWrapper(mutation_strength=0.2, injection_mode=injection_mode)

    # Resource allocation
    n_keys = mutator.num_keys(input_shape=(pop_size,))
    expected_keys = 1 if injection_mode else pop_size * mutator.num_offspring * mutator.num_keys_per_atomic_operation
    assert n_keys == expected_keys

    k_op, _ = jar.split(prng_key)
    keys = jar.split(k_op, n_keys)

    # JIT and execute
    jit_mutator = jax.jit(mutator)
    new_pop = jit_mutator(keys, population, config)

    # Basic assertions
    assert new_pop.genes.values.shape == population.genes.values.shape
    diff = jnp.sum(jnp.abs(new_pop.genes.values - population.genes.values))
    # With non-zero mutation strength, we expect some change in the genes
    assert diff > 0, "Evosax wrapper failed to modify genes"


@pytest.mark.parametrize("injection_mode", [True, False])
def test_no_keys_raises(evosax_mut_setup, injection_mode):
    config, pop_size, population = evosax_mut_setup
    empty_keys = jnp.empty((0, 2), dtype=jnp.uint32)
    mutator = EvosaxGaussianWrapper(mutation_strength=0.2, injection_mode=injection_mode)
    with pytest.raises(ValueError):
        mutator(empty_keys, population, config)
