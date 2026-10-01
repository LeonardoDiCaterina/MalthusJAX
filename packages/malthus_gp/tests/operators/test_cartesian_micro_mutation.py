import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.cartesian_micro_mutation import CartesianMicroMutation


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
    return config.init_population(key, 5)


def test_mutation_zero_rate_is_identity(config, population):
    mutation = CartesianMicroMutation(mutation_rate=0.0, input_length=5)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5 * mutation.num_keys_per_atomic_operation)
    mutated = mutation(keys, population, config)

    assert jnp.array_equal(mutated.genes.ops, population.genes.ops)
    assert jnp.array_equal(mutated.genes.args, population.genes.args)
    assert jnp.array_equal(mutated.genes.out_nodes, population.genes.out_nodes)


def test_mutation_high_rate_perturbs_genes(config, population):
    mutation = CartesianMicroMutation(mutation_rate=1.0, input_length=5)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5 * mutation.num_keys_per_atomic_operation)
    mutated = mutation(keys, population, config)

    # All opcodes should have been resampled
    diff_ops = jnp.sum(mutated.genes.ops != population.genes.ops)
    assert diff_ops > 0


def test_mutation_respects_bounds(config, population):
    # Test with strict levels_back
    strict_config = CartesianGenomeConfig(
        num_rows=1,
        num_cols=10,
        num_inputs=2,
        num_outputs=1,
        num_ops=4,
        max_arity=2,
        levels_back=2,
    )
    pop = strict_config.init_population(jax.random.PRNGKey(1), 5)
    mutation = CartesianMicroMutation(mutation_rate=0.5, input_length=5)
    keys = jax.random.split(jax.random.PRNGKey(2), 5 * mutation.num_keys_per_atomic_operation)
    mutated = mutation(keys, pop, strict_config)

    lo, hi = strict_config._col_connection_bounds()
    for i in range(5):
        args_i = mutated.genes.args[i]
        # Check that each node's connections stay in [lo, hi)
        assert jnp.all(args_i >= lo[:, None])
        assert jnp.all(args_i < hi[:, None])
