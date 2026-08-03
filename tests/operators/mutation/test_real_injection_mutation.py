import pytest
import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.real import (
    BallMutation_injection,
    GaussianMutation_injection,
    PolynomialMutation_injection,
)


@pytest.fixture
def real_injection_mut_setup(prng_key):
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    pop_size = 5
    population = RealPopulation.init_random(prng_key, config, pop_size)
    return config, pop_size, population


def _run_injection_mutation(operator_cls, setup_data, prng_key, **kwargs):
    config, pop_size, population = setup_data
    
    mutator = operator_cls(num_offspring=1, **kwargs).set_input_length(pop_size)

    n_keys = mutator.num_keys(input_shape=(pop_size,))
    k_op, _ = jar.split(prng_key)
    keys = jar.split(k_op, n_keys)

    jit_mutator = jax.jit(mutator)
    new_pop = jit_mutator(keys, population, config)

    assert new_pop.genes.values.shape == population.genes.values.shape

    diff = jnp.sum(jnp.abs(new_pop.genes.values - population.genes.values))
    assert diff > 0, "Injection mutation failed to modify genes"


def test_gaussian_injection(real_injection_mut_setup, prng_key):
    _run_injection_mutation(GaussianMutation_injection, real_injection_mut_setup, prng_key, mutation_rate=1.0, mutation_strength=0.5)


def test_ball_injection(real_injection_mut_setup, prng_key):
    _run_injection_mutation(BallMutation_injection, real_injection_mut_setup, prng_key, mutation_rate=1.0, radius=0.5)


def test_polynomial_injection(real_injection_mut_setup, prng_key):
    _run_injection_mutation(PolynomialMutation_injection, real_injection_mut_setup, prng_key, mutation_rate=1.0, eta=20.0)
