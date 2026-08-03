import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
import chex

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import (
    BinomialCrossover_injection,
    BlendCrossover_injection,
    SimulatedBinaryCrossover_injection,
    UniformCrossover_injection,
)


@pytest.fixture
def real_injection_cx_setup(prng_key):
    config = RealGenomeConfig(shape=(10,), bounds=(-10.0, 10.0))
    pop_size = 4
    k1, k2 = jar.split(prng_key)
    parents_1 = RealPopulation.init_random(k1, config, pop_size)
    p2_genes = RealGenome(values=jnp.full((pop_size, 10), 10.0))
    parents_2 = parents_1.spawn_offspring(p2_genes)
    
    return config, pop_size, parents_1, parents_2


def _run_injection_crossover(operator_cls, setup_data, prng_key, num_offspring=1, **kwargs):
    config, pop_size, parents_1, parents_2 = setup_data
    
    crossover = operator_cls(num_offspring=num_offspring, **kwargs)
    crossover = crossover.set_input_length(pop_size)

    n_keys = crossover.num_keys(input_shape=(pop_size,))
    k_op, _ = jar.split(prng_key)
    keys = jar.split(k_op, n_keys)

    jit_op = jax.jit(crossover)
    offspring_pop = jit_op(keys, parents_1, parents_2, config)

    assert len(offspring_pop) == pop_size * num_offspring

    vals = offspring_pop.genes.values
    is_same_p1 = jnp.allclose(vals, parents_1.genes.values.repeat(num_offspring, axis=0))
    is_same_p2 = jnp.allclose(vals, parents_2.genes.values.repeat(num_offspring, axis=0))

    assert not (is_same_p1 and is_same_p2), "Injection offspring are identical to parents"
    
    return offspring_pop


def test_uniform_injection(real_injection_cx_setup, prng_key):
    _run_injection_crossover(UniformCrossover_injection, real_injection_cx_setup, prng_key, num_offspring=1, crossover_rate=0.5)


def test_blend_injection(real_injection_cx_setup, prng_key):
    _run_injection_crossover(BlendCrossover_injection, real_injection_cx_setup, prng_key, num_offspring=1, alpha=0.5)


def test_sbx_injection(real_injection_cx_setup, prng_key):
    _run_injection_crossover(SimulatedBinaryCrossover_injection, real_injection_cx_setup, prng_key, num_offspring=2, eta=10.0)


def test_binomial_injection(real_injection_cx_setup, prng_key):
    _run_injection_crossover(BinomialCrossover_injection, real_injection_cx_setup, prng_key, num_offspring=1, crossover_rate=0.9)


def test_binomial_injection_rate_edges(real_injection_cx_setup, prng_key):
    config, pop_size, parents_1, parents_2 = real_injection_cx_setup
    
    # Rate=0 => preserve p2 (target)
    op = BinomialCrossover_injection(num_offspring=1, crossover_rate=0.0)
    op = op.set_input_length(pop_size)
    k = jar.PRNGKey(123)
    keys = jar.split(k, op.num_keys(input_shape=(pop_size,)))
    out = op(keys, parents_2, parents_1, config)
    chex.assert_trees_all_close(out.genes.values, parents_2.genes.values)

    # Rate=1 => preserve p1 (mutant)
    op = BinomialCrossover_injection(num_offspring=1, crossover_rate=1.0)
    op = op.set_input_length(pop_size)
    k = jar.PRNGKey(456)
    keys = jar.split(k, op.num_keys(input_shape=(pop_size,)))
    out = op(keys, parents_2, parents_1, config)
    chex.assert_trees_all_close(out.genes.values, parents_1.genes.values)
