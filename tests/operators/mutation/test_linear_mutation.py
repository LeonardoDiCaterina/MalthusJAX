"""Tests for the LinearGenome mutation operator."""

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
from malthusjax.operators.mutation.linear import (
    DECAY_FUNCTIONS,
    LinearMutation,
    LinearMutationConfig,
)


@pytest.fixture
def dummy_genome() -> LinearGenome:
    """A valid 10-length LinearGenome."""
    ops = jnp.zeros(10, dtype=jnp.int32)
    args = jnp.zeros((10, 2), dtype=jnp.int32)
    return LinearGenome(ops=ops, args=args)


@pytest.fixture
def genome_config() -> LinearGenomeConfig:
    """Config with 5 inputs, 8 ops, arity 2."""
    return LinearGenomeConfig(length=10, num_inputs=5, num_ops=8, max_arity=2)


def test_decay_functions():
    """Verify decay function outputs and shapes."""
    max_len = 10
    
    # 1. Uniform
    w_uni = DECAY_FUNCTIONS["uniform"](4, 0.0, max_len)
    assert jnp.allclose(w_uni[:4], 1.0)
    assert jnp.allclose(w_uni[4:], 0.0)
    
    # 2. Geometric (beta=0.5)
    w_geo = DECAY_FUNCTIONS["geometric"](4, 0.5, max_len)
    assert jnp.isclose(w_geo[3], 1.0)      # Most recent (dist 1) -> 0.5^0
    assert jnp.isclose(w_geo[2], 0.5)      # Dist 2 -> 0.5^1
    assert jnp.isclose(w_geo[1], 0.25)     # Dist 3 -> 0.5^2
    assert jnp.isclose(w_geo[0], 0.125)    # Dist 4 -> 0.5^3
    assert jnp.allclose(w_geo[4:], 0.0)    # Future rows invalid
    
    # 3. Linear
    w_lin = DECAY_FUNCTIONS["linear"](4, 0.0, max_len)
    assert jnp.allclose(w_lin[:4], jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.allclose(w_lin[4:], 0.0)
    
    # 4. Window (size=2)
    w_win = DECAY_FUNCTIONS["window"](4, 2.0, max_len)
    assert jnp.allclose(w_win[:2], 0.0)
    assert jnp.allclose(w_win[2:4], 1.0)
    assert jnp.allclose(w_win[4:], 0.0)


def test_linear_mutation_topological_validity(dummy_genome, genome_config):
    """Ensure mutated genome respects topological order constraints."""
    key = jax.random.PRNGKey(42)
    # High mutation rate to force many changes
    config = LinearMutationConfig(mutation_rate=1.0, p_internal=0.5, decay_name="uniform")
    mutator = LinearMutation()
    
    # _mutate_one manually
    noise = mutator._generate_noise(jax.random.split(key, mutator.num_keys_per_atomic_operation), config)
    mutated = mutator._mutate_one(dummy_genome, noise, config, genome_config=genome_config)
    
    # Ops must be in [0, num_ops-1]
    assert jnp.all(mutated.ops >= 0)
    assert jnp.all(mutated.ops < genome_config.num_ops)
    
    # Args must be in [0, num_inputs + i - 1]
    for i in range(genome_config.length):
        max_valid_idx = genome_config.num_inputs + i - 1
        assert jnp.all(mutated.args[i] <= max_valid_idx)
        assert jnp.all(mutated.args[i] >= 0)


def test_linear_mutation_row_zero_clamping(dummy_genome, genome_config):
    """Row 0 must NEVER reference an internal row, even if p_internal is 1.0."""
    key = jax.random.PRNGKey(1337)
    # Force internal references
    config = LinearMutationConfig(mutation_rate=1.0, p_internal=1.0, decay_name="uniform")
    mutator = LinearMutation()
    
    noise = mutator._generate_noise(jax.random.split(key, 2), config)
    mutated = mutator._mutate_one(dummy_genome, noise, config, genome_config=genome_config)
    
    # Row 0 MUST have references < num_inputs
    assert jnp.all(mutated.args[0] < genome_config.num_inputs)
    # Other rows will have references >= num_inputs (because p_internal=1.0 forces it)
    assert jnp.all(mutated.args[1:] >= genome_config.num_inputs)


def test_linear_mutation_missing_genome_config(dummy_genome):
    """Raises ValueError if genome_config is not provided."""
    key = jax.random.PRNGKey(0)
    config = LinearMutationConfig()
    mutator = LinearMutation()
    noise = mutator._generate_noise(jax.random.split(key, 2), config)
    
    with pytest.raises(ValueError, match="requires 'genome_config'"):
        mutator._mutate_one(dummy_genome, noise, config)


def test_linear_mutation_jit(dummy_genome, genome_config):
    """Verify that the operator is fully JIT-compilable."""
    key = jax.random.PRNGKey(42)
    config = LinearMutationConfig(mutation_rate=0.5, p_internal=0.5)
    mutator = LinearMutation()
    
    @jax.jit
    def jitted_mutate(k, g, c, gc):
        noise = mutator._generate_noise(jax.random.split(k, 2), c)
        return mutator._mutate_one(g, noise, c, genome_config=gc)
        
    mutated = jitted_mutate(key, dummy_genome, config, genome_config)
    
    assert mutated.ops.shape == dummy_genome.ops.shape
    assert mutated.args.shape == dummy_genome.args.shape
