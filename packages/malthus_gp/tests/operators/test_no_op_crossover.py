import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.no_op_crossover import NoOpCrossover


@pytest.fixture
def config():
    return CartesianGenomeConfig(
        num_rows=1,
        num_cols=10,
        num_inputs=2,
        num_outputs=1,
        num_ops=4,
        max_arity=2,
        levels_back=-1,
    )


@pytest.fixture
def population(config):
    key = jax.random.PRNGKey(0)
    return config.init_population(key, 4)


def test_no_op_crossover_preserves_parent(config, population):
    crossover = NoOpCrossover(input_length=4)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4 * crossover.num_keys_per_atomic_operation)
    offspring = crossover(keys, population, population, config)

    assert jnp.array_equal(offspring.genes.ops, population.genes.ops)
    assert jnp.array_equal(offspring.genes.args, population.genes.args)
    assert jnp.array_equal(offspring.genes.out_nodes, population.genes.out_nodes)
