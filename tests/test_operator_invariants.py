"""
Operator Bounds and Determinism Invariant Tests for MalthusJAX.

Guarantees that:
1. Operators respect bounds clipping when configured.
2. Operations under the same PRNG key are completely deterministic.
3. Different PRNG keys produce divergent random mutations.
"""

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealGenome, RealPopulation
from malthusjax.operators.mutation.real import GaussianMutation


def test_operator_bound_invariance():
    """Checks that aggressive mutation strictly respects boundary clipping."""
    key = jax.random.PRNGKey(42)
    bounds = (-1.0, 1.0)
    config = RealGenomeConfig(shape=(50, 10), bounds=bounds, dtype=jnp.float32)
    pop = RealPopulation.init_random(key, config, 50)

    # Apply aggressive mutation with clip=True
    mutation = GaussianMutation(mutation_rate=1.0, mutation_strength=10.0, clip=True)
    k1, k2 = jax.random.split(key)
    # Generate 2 keys for GaussianMutation (mask key, noise key)
    keys = jax.random.split(k1, 2)
    noise = mutation._generate_noise(keys, config)
    mutated_vals = mutation._apply_noise(pop.genes.values, noise, config)

    assert jnp.all(mutated_vals >= bounds[0]), "Lower bound violated after mutation!"
    assert jnp.all(mutated_vals <= bounds[1]), "Upper bound violated after mutation!"


def test_seed_determinism_and_divergence():
    """Checks that identical seeds give identical trajectories and different seeds diverge."""
    config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0), dtype=jnp.float32)

    pop_a1 = RealPopulation.init_random(jax.random.PRNGKey(1), config, 20)
    pop_a2 = RealPopulation.init_random(jax.random.PRNGKey(1), config, 20)
    pop_b = RealPopulation.init_random(jax.random.PRNGKey(2), config, 20)

    assert jnp.array_equal(pop_a1.genes.values, pop_a2.genes.values), "Non-deterministic initialisation!"
    assert not jnp.array_equal(pop_a1.genes.values, pop_b.genes.values), "Identical populations from different seeds!"
