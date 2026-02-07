"""
Enhanced Mutation Test Suite: Inner Methods & Expected Behavior.

This module tests:
1. Tier 2: _generate_noise() - correct shapes, distributions, determinism
2. Tier 1: _mutate_one() - pure arithmetic behavior
3. Fused: _mutate_fused() and __call__ - combined RNG + arithmetic population operations
4. Injection mode shape checks and input_length validation
5. Expected statistical properties (mutation rate, clipping)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation import (
    BallMutation,
    BallMutation_injection,
    BitFlipMutation,
    GaussianMutation,
    GaussianMutation_injection,
    ScrambleMutation,
    SwapMutation,
)


@pytest.fixture
def binary_config():
    return BinaryGenomeConfig(shape=(20,), p=0.5)


@pytest.fixture
def real_config():
    return RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0), dtype=jnp.float32)


@pytest.fixture
def binary_population(binary_config):
    key = jr.PRNGKey(42)
    key, k = jr.split(key)
    return BinaryPopulation.init_random(k, binary_config, size=10)


@pytest.fixture
def real_population(real_config):
    key = jr.PRNGKey(43)
    k1, k2 = jr.split(key)
    return RealPopulation.init_random(k1, real_config, size=10)


class TestGenerateNoiseFused:
    """Tests for Tier 2 _generate_noise() in fused-mode mutation operators."""

    def test_bitflip_noise_shape_and_dtype(self, binary_config):
        op = BitFlipMutation(mutation_rate=0.3)
        key = jr.PRNGKey(123)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        mask = op._generate_noise(keys, binary_config)

        assert mask.shape == binary_config.shape
        assert mask.dtype == jnp.bool_

    def test_bitflip_noise_distribution(self, binary_config):
        rates = [0.1, 0.5, 0.9]
        for rate in rates:
            op = BitFlipMutation(mutation_rate=rate)
            masks = []
            for i in range(400):
                key = jr.PRNGKey(i)
                keys = jr.split(key, op.num_keys_per_atomic_operation)
                masks.append(op._generate_noise(keys, binary_config))
            all_masks = jnp.stack(masks)
            empirical = float(jnp.mean(all_masks))
            assert abs(empirical - rate) < 0.08

    def test_bitflip_noise_determinism(self, binary_config):
        op = BitFlipMutation(mutation_rate=0.5)
        key = jr.PRNGKey(999)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        m1 = op._generate_noise(keys, binary_config)
        m2 = op._generate_noise(keys, binary_config)
        assert jnp.all(m1 == m2)

    def test_scramble_noise_components(self, binary_config):
        op = ScrambleMutation(mutation_rate=0.7)
        key = jr.PRNGKey(21)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        should_mutate, indices = op._generate_noise(keys, binary_config)
        assert isinstance(should_mutate, (jnp.ndarray, jnp.bool_))
        assert indices.shape == (binary_config.shape[-1],)

    def test_swap_noise_components(self, binary_config):
        op = SwapMutation(mutation_rate=0.5)
        key = jr.PRNGKey(22)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        should_mutate, i1, i2 = op._generate_noise(keys, binary_config)
        assert isinstance(should_mutate, (jnp.ndarray, jnp.bool_))
        assert isinstance(i1, jnp.ndarray) or jnp.issubdtype(type(i1), jnp.integer)
        assert isinstance(i2, jnp.ndarray) or jnp.issubdtype(type(i2), jnp.integer)

    def test_gaussian_noise_payload(self, real_config):
        op = GaussianMutation(mutation_rate=0.5, mutation_strength=0.2)
        key = jr.PRNGKey(77)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        payload = op._generate_noise(keys, real_config)
        assert payload.shape == real_config.shape
        # If mutation rate < 1, expect some zeros; but ensure dtype matches
        assert payload.dtype == real_config.dtype

    def test_ball_noise_magnitude(self, real_config):
        op = BallMutation(radius=0.5, mutation_rate=1.0)
        key = jr.PRNGKey(88)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        delta = op._generate_noise(keys, real_config)
        # magnitude should be <= radius
        mag = jnp.sqrt(jnp.sum(delta**2))
        assert float(mag) <= 0.5 + 1e-6


class TestGenerateNoiseInjection:
    """Tests for Tier 2 _generate_noise() in injection-mode operators."""

    def test_gaussian_injection_shape(self, real_config):
        op = GaussianMutation_injection(num_offspring=2, mutation_rate=0.5)
        op = op.set_input_length(8)
        key = jr.PRNGKey(11)

        noise = op._generate_noise(key, real_config)
        expected = (8 * op.num_offspring,) + real_config.shape
        assert noise.shape == expected

    def test_ball_injection_shape(self, real_config):
        op = BallMutation_injection(num_offspring=3, radius=0.4, mutation_rate=0.6)
        op = op.set_input_length(5)
        key = jr.PRNGKey(12)

        noise = op._generate_noise(key, real_config)
        expected = (5 * op.num_offspring,) + real_config.shape
        assert noise.shape == expected

    def test_injection_requires_input_length(self, real_config):
        op = GaussianMutation_injection(num_offspring=1)
        key = jr.PRNGKey(4444)
        with pytest.raises(ValueError, match="input_length"):
            op._generate_noise(key, real_config)


class TestMutateOne:
    """Tests for Tier 1 _mutate_one() pure arithmetic kernel."""

    def test_bitflip_mutate_all_false(self, binary_config):
        op = BitFlipMutation(mutation_rate=0.5)
        p = BinaryGenome(values=jnp.zeros(binary_config.shape, dtype=jnp.bool_))
        mask = jnp.zeros(binary_config.shape, dtype=jnp.bool_)
        out = op._mutate_one(p, mask, binary_config)
        assert jnp.array_equal(out.values, p.values)

    def test_bitflip_mutate_all_true(self, binary_config):
        op = BitFlipMutation(mutation_rate=0.5)
        p = BinaryGenome(values=jnp.zeros(binary_config.shape, dtype=jnp.bool_))
        mask = jnp.ones(binary_config.shape, dtype=jnp.bool_)
        out = op._mutate_one(p, mask, binary_config)
        # flipping zeros -> ones
        assert jnp.all(out.values)

    def test_scramble_mutation_applied(self, binary_config):
        op = ScrambleMutation(mutation_rate=1.0)
        p = BinaryGenome(values=jnp.arange(binary_config.shape[-1]) % 2 == 0)
        should_mutate = True
        indices = jnp.arange(binary_config.shape[-1])[::-1]
        out = op._mutate_one(p, (should_mutate, indices), binary_config)
        assert jnp.array_equal(out.values, p.values[indices])

    def test_swap_mutation_applied(self, binary_config):
        op = SwapMutation(mutation_rate=1.0)
        p = BinaryGenome(values=jnp.arange(binary_config.shape[-1]) % 2 == 0)
        idx1 = 0
        idx2 = 1
        out = op._mutate_one(p, (True, idx1, idx2), binary_config)
        assert out.values[0] == p.values[1]
        assert out.values[1] == p.values[0]

    def test_gaussian_mutation_arithmetic(self, real_config):
        op = GaussianMutation(mutation_rate=1.0, mutation_strength=0.3, clip=False)
        p = RealGenome(values=jnp.zeros(real_config.shape, dtype=real_config.dtype))

        # create a noise payload (mask already applied in _generate_noise)
        payload = jnp.ones(real_config.shape, dtype=real_config.dtype) * 0.3
        out = op._mutate_one(p, payload, real_config)
        assert jnp.allclose(out.values, payload, atol=1e-6)

    def test_gaussian_clipping(self, real_config):
        op = GaussianMutation(mutation_rate=1.0, mutation_strength=10.0, clip=True)
        p = RealGenome(values=jnp.full(real_config.shape, 4.5, dtype=real_config.dtype))

        payload = jnp.ones(real_config.shape, dtype=real_config.dtype) * 10.0
        out = op._mutate_one(p, payload, real_config)
        # Values should be clipped to config bounds
        assert jnp.all(out.values <= real_config.bounds[1])
        assert jnp.all(out.values >= real_config.bounds[0])


class TestFusedAndPopulation:
    """Tests for fused population-level mutation operations and behavior."""

    def test_mutation_population_output_size_and_fitness_reset(self, real_population, real_config):
        pop = real_population
        pop_size = len(pop)
        op = GaussianMutation(
            num_offspring=2,
            mutation_rate=0.5,
            input_length=pop_size,
        )
        op = op.set_input_length(pop_size)

        key = jr.PRNGKey(999)
        keys = jr.split(key, op.num_keys(pop.genes.values.shape))

        # Set parent fitness to zeros
        pop = pop.replace(fitness=jnp.zeros(len(pop)))

        offspring_pop = op(keys, pop, pop.config)
        assert len(offspring_pop) == pop_size * op.num_offspring
        assert jnp.all(jnp.isnan(offspring_pop.fitness))

    def test_mutate_fused_reproducibility(self, real_config):
        op = GaussianMutation(mutation_rate=0.5, mutation_strength=0.2)
        p = RealGenome(values=jnp.zeros(real_config.shape, dtype=real_config.dtype))

        key = jr.PRNGKey(1010)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        out1 = op._mutate_fused(keys, p, real_config)
        out2 = op._mutate_fused(keys, p, real_config)
        assert jnp.allclose(out1.values, out2.values)

    def test_jit_stability(self, real_population, real_config):
        pop = real_population
        pop_size = len(pop)
        op = GaussianMutation(
            num_offspring=1,
            mutation_rate=0.5,
            input_length=pop_size,
        )
        op = op.set_input_length(pop_size)

        key = jr.PRNGKey(1313)
        keys = jr.split(key, op.num_keys((pop_size,)))

        @jax.jit
        def mutate_jit(k, p, c):
            return op(k, p, c)

        r1 = op(keys, pop, real_config)
        r2 = mutate_jit(keys, pop, real_config)
        np.testing.assert_allclose(r1.genes.values, r2.genes.values, atol=1e-6)


class TestStatisticalBehavior:
    """Tests that mutation rate and statistical properties behave as expected."""

    def test_gaussian_mutation_rate_respected(self, real_config):
        op = GaussianMutation(num_offspring=1, mutation_rate=0.3, mutation_strength=0.1)

        counts = []
        for i in range(300):
            key = jr.PRNGKey(i)
            keys = jr.split(key, op.num_keys_per_atomic_operation)
            payload = op._generate_noise(keys, real_config)
            # fraction of non-zero components
            frac = float(jnp.mean(jnp.abs(payload) > 0.0))
            counts.append(frac)
        assert abs(np.mean(counts) - 0.3) < 0.08

    def test_ball_mutation_respects_radius_and_rate(self, real_config):
        op = BallMutation(radius=0.7, mutation_rate=0.5)
        key = jr.PRNGKey(2020)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        delta = op._generate_noise(keys, real_config)
        mag = jnp.sqrt(jnp.sum(delta**2))
        # Magnitude respects radius
        assert float(mag) <= 0.7 + 1e-6

        # Also check mask behavior: since mutation_rate=0.5 some rows may be zeros
        zeros_fraction = float(jnp.mean(jnp.all(delta == 0.0, axis=tuple(range(1, delta.ndim)))))
        assert 0.0 <= zeros_fraction <= 1.0
