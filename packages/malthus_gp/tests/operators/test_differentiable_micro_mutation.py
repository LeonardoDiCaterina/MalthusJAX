import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenomeConfig
from malthus_gp.operators.differentiable_mutation import DifferentiableMicroMutation


@pytest.fixture
def config():
    return DifferentiableCartesianGenomeConfig(
        num_rows=1,
        num_cols=10,
        num_inputs=2,
        num_outputs=1,
        num_ops=8,
        max_arity=2,
        levels_back=-1,
    )


@pytest.fixture
def population(config):
    key = jax.random.PRNGKey(0)
    return config.init_population(key, 5)


def test_differentiable_mutation_preserves_weights(config, population):
    # With weight_noise_std=0.0, weights must be preserved intact
    mutation = DifferentiableMicroMutation(mutation_rate=0.5, weight_noise_std=0.0, input_length=5)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5 * mutation.num_keys_per_atomic_operation)
    mutated = mutation(keys, population, config)

    assert jnp.array_equal(mutated.genes.weights, population.genes.weights)
    assert jnp.array_equal(mutated.genes.biases, population.genes.biases)


def test_differentiable_mutation_perturbs_weights_when_requested(config, population):
    mutation = DifferentiableMicroMutation(mutation_rate=0.0, weight_noise_std=0.1, input_length=5)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5 * mutation.num_keys_per_atomic_operation)
    mutated = mutation(keys, population, config)

    assert not jnp.array_equal(mutated.genes.weights, population.genes.weights)
    assert not jnp.array_equal(mutated.genes.biases, population.genes.biases)
    # Topology should remain identical since mutation_rate=0.0
    assert jnp.array_equal(mutated.genes.ops, population.genes.ops)
