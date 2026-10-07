import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
    DifferentiableCartesianGenomeConfig,
)
from malthusjax.testing.compliance import GenomeComplianceSuite


class TestDifferentiableCartesianGenome(GenomeComplianceSuite):
    @pytest.fixture
    def config(self):
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
    def component(self, config):
        rng = jax.random.PRNGKey(0)
        return DifferentiableCartesianGenome.random_init(rng, config=config)


def test_differentiable_continuous_params():
    config = DifferentiableCartesianGenomeConfig(
        num_rows=1,
        num_cols=10,
        num_inputs=2,
        num_outputs=1,
        max_arity=2,
    )
    rng = jax.random.PRNGKey(42)
    genome = DifferentiableCartesianGenome.random_init(rng, config)

    assert genome.weights.shape == (10, 2)
    assert genome.biases.shape == (10,)
    assert genome.weights.dtype == jnp.float32
    assert genome.biases.dtype == jnp.float32

    # Verify forget step randomizes weights while keeping topology
    k_forget = jax.random.PRNGKey(99)
    forgotten = genome.forget_weights(k_forget, config)
    assert jnp.array_equal(forgotten.ops, genome.ops)
    assert jnp.array_equal(forgotten.args, genome.args)
    assert jnp.array_equal(forgotten.out_nodes, genome.out_nodes)
    assert not jnp.array_equal(forgotten.weights, genome.weights)
