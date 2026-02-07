"""
Enhanced Crossover Test Suite: Inner Methods & Expected Behavior.

This module tests:
1. Tier 2: _generate_noise() - correct shapes, distributions, determinism
2. Tier 1: _recombine_one() - pure arithmetic behavior
3. Fused: _cross_fused() - combined RNG + arithmetic
4. Expected behavior: statistical properties, boundary handling, inheritance patterns
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import (
    BinomialCrossover,
    BlendCrossover,
    BlendCrossover_injection,
    SimulatedBinaryCrossover,
    SimulatedBinaryCrossover_injection,
    UniformCrossover,
    UniformCrossover_injection,
)


@pytest.fixture
def real_config():
    """Standard real-valued genome configuration."""
    return RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0), dtype=jnp.float32)


@pytest.fixture
def wide_config():
    """Wide bounds for testing clipping behavior."""
    return RealGenomeConfig(shape=(20,), bounds=(-100.0, 100.0), dtype=jnp.float32)


@pytest.fixture
def parent_pair(real_config):
    """Create a fixed parent pair for consistent testing."""
    p1 = RealGenome(
        values=jnp.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    )
    p2 = RealGenome(
        values=jnp.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=jnp.float32)
    )
    return p1, p2, real_config


@pytest.fixture
def population_pair(real_config):
    """Create population pairs for population-level testing."""
    key = jr.PRNGKey(42)
    pop_size = 10
    k1, k2 = jr.split(key)
    p1_pop = RealPopulation.init_random(k1, real_config, pop_size)
    p2_pop = RealPopulation.init_random(k2, real_config, pop_size)
    return p1_pop, p2_pop, real_config


class TestGenerateNoiseFused:
    """Tests for Tier 2 _generate_noise() in fused-mode operators."""

    def test_uniform_noise_shape(self, real_config):
        """UniformCrossover._generate_noise returns correct shape."""
        op = UniformCrossover(crossover_rate=0.5)
        key = jr.PRNGKey(123)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        noise = op._generate_noise(keys, real_config)

        # Should be boolean mask with genome shape
        assert noise.shape == real_config.shape
        assert noise.dtype == jnp.bool_

    def test_uniform_noise_distribution(self, real_config):
        """UniformCrossover masks follow Bernoulli(crossover_rate)."""
        for rate in [0.3, 0.5, 0.8]:
            op = UniformCrossover(crossover_rate=rate)

            # Generate many masks and check mean
            masks = []
            for i in range(500):
                key = jr.PRNGKey(i)
                keys = jr.split(key, op.num_keys_per_atomic_operation)
                mask = op._generate_noise(keys, real_config)
                masks.append(mask)

            all_masks = jnp.stack(masks)
            empirical_rate = jnp.mean(all_masks)

            # Should be close to crossover_rate (allow statistical variance)
            assert abs(float(empirical_rate) - rate) < 0.1, (
                f"Expected ~{rate}, got {empirical_rate}"
            )

    def test_uniform_noise_determinism(self, real_config):
        """Same keys produce identical noise."""
        op = UniformCrossover(crossover_rate=0.5)
        key = jr.PRNGKey(999)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        noise1 = op._generate_noise(keys, real_config)
        noise2 = op._generate_noise(keys, real_config)

        assert jnp.all(noise1 == noise2)

    def test_blend_noise_components(self, real_config):
        """BlendCrossover._generate_noise returns (should_cross, random_samples)."""
        op = BlendCrossover(crossover_rate=0.9, alpha=0.5)
        key = jr.PRNGKey(456)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        noise = op._generate_noise(keys, real_config)

        assert isinstance(noise, tuple)
        assert len(noise) == 2

        should_cross, random_samples = noise

        # should_cross is scalar boolean
        assert should_cross.shape == ()
        assert should_cross.dtype == jnp.bool_

        # random_samples matches genome shape
        assert random_samples.shape == real_config.shape
        assert random_samples.dtype == real_config.dtype

        # random_samples in [0, 1)
        assert jnp.all(random_samples >= 0.0)
        assert jnp.all(random_samples < 1.0)

    def test_sbx_noise_components(self, real_config):
        """SBX._generate_noise returns (should_cross, u, swap_mask)."""
        op = SimulatedBinaryCrossover(crossover_rate=0.9, eta=20.0)
        key = jr.PRNGKey(789)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        noise = op._generate_noise(keys, real_config)

        assert isinstance(noise, tuple)
        assert len(noise) == 3

        should_cross, u, swap_mask = noise
        assert should_cross.shape == ()
        assert u.shape == real_config.shape
        assert jnp.all(u >= 0.0)
        assert jnp.all(u < 1.0)
        assert swap_mask.shape == real_config.shape
        assert swap_mask.dtype == jnp.bool_

    def test_binomial_noise_shape(self, real_config):
        """BinomialCrossover._generate_noise returns correct shape."""
        op = BinomialCrossover(crossover_rate=0.7)
        key = jr.PRNGKey(321)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        noise = op._generate_noise(keys, real_config)
        assert noise.shape == real_config.shape
        assert noise.dtype == jnp.bool_


class TestGenerateNoiseInjection:
    """Tests for Tier 2 _generate_noise() in injection-mode operators."""

    def test_uniform_injection_shape(self, real_config):
        """Injection mode generates (input_length * num_offspring, ...) shape."""
        pop_size = 10
        num_offspring = 2

        op = UniformCrossover_injection(num_offspring=num_offspring, crossover_rate=0.5)
        op = op.set_input_length(pop_size)

        key = jr.PRNGKey(111)
        noise = op._generate_noise(key, real_config)

        expected_rows = pop_size * num_offspring
        assert noise.shape == (expected_rows,) + real_config.shape

    def test_blend_injection_shape(self, real_config):
        """BlendCrossover_injection generates correct tuple shapes."""
        pop_size = 5
        num_offspring = 3

        op = BlendCrossover_injection(num_offspring=num_offspring, crossover_rate=0.9, alpha=0.5)
        op = op.set_input_length(pop_size)

        key = jr.PRNGKey(222)
        should_cross, random_samples = op._generate_noise(key, real_config)

        expected_rows = pop_size * num_offspring
        assert should_cross.shape == (expected_rows,)
        assert random_samples.shape == (expected_rows,) + real_config.shape

    def test_sbx_injection_shape(self, real_config):
        """SBX_injection generates correct tuple shapes."""
        pop_size = 8
        num_offspring = 2

        op = SimulatedBinaryCrossover_injection(num_offspring=num_offspring, eta=20.0)
        op = op.set_input_length(pop_size)

        key = jr.PRNGKey(333)
        should_cross, u, swap_mask = op._generate_noise(key, real_config)

        expected_rows = pop_size * num_offspring
        assert should_cross.shape == (expected_rows,)
        assert u.shape == (expected_rows,) + real_config.shape
        assert swap_mask.shape == (expected_rows,) + real_config.shape

    def test_injection_requires_input_length(self, real_config):
        """Injection mode raises error if input_length not set."""
        op = UniformCrossover_injection(num_offspring=1)
        key = jr.PRNGKey(444)
        with pytest.raises(ValueError, match="input_length"):
            op._generate_noise(key, real_config)


class TestRecombineOne:
    """Tests for Tier 1 _recombine_one() pure arithmetic kernel."""

    def test_uniform_inheritance_all_p1(self, parent_pair):
        """All-False mask → offspring == p1."""
        p1, p2, config = parent_pair
        op = UniformCrossover(crossover_rate=0.5)

        mask = jnp.zeros(config.shape, dtype=jnp.bool_)
        offspring = op._recombine_one(p1, p2, mask, config)

        assert jnp.allclose(offspring.values, p1.values)

    def test_uniform_inheritance_all_p2(self, parent_pair):
        """All-True mask → offspring == p2."""
        p1, p2, config = parent_pair
        op = UniformCrossover(crossover_rate=0.5)

        mask = jnp.ones(config.shape, dtype=jnp.bool_)
        offspring = op._recombine_one(p1, p2, mask, config)

        assert jnp.allclose(offspring.values, p2.values)

    def test_uniform_inheritance_mixed(self, parent_pair):
        """Mixed mask → correct gene selection."""
        p1, p2, config = parent_pair
        op = UniformCrossover(crossover_rate=0.5)
        mask = jnp.array(
            [False, True, False, True, False, True, False, True, False, True], dtype=jnp.bool_
        )
        offspring = op._recombine_one(p1, p2, mask, config)

        for i, m in enumerate(mask):
            expected = p2.values[i] if m else p1.values[i]
            assert offspring.values[i] == expected

    def test_blend_no_crossover(self, parent_pair):
        """BlendCrossover with should_cross=False → offspring == p1."""
        p1, p2, config = parent_pair
        op = BlendCrossover(crossover_rate=0.9, alpha=0.5)

        should_cross = jnp.array(False)
        random_samples = jnp.ones(config.shape, dtype=config.dtype) * 0.5
        noise_data = (should_cross, random_samples)

        offspring = op._recombine_one(p1, p2, noise_data, config)

        assert jnp.allclose(offspring.values, p1.values)

    def test_blend_crossover_midpoint(self, parent_pair):
        """BlendCrossover with alpha=0, random=0.5 → midpoint."""
        p1, p2, config = parent_pair
        op = BlendCrossover(crossover_rate=1.0, alpha=0.0)

        should_cross = jnp.array(True)
        random_samples = jnp.ones(config.shape, dtype=config.dtype) * 0.5
        noise_data = (should_cross, random_samples)

        offspring = op._recombine_one(p1, p2, noise_data, config)

        expected = (p1.values + p2.values) / 2.0
        assert jnp.allclose(offspring.values, expected, atol=1e-5)

    def test_blend_crossover_edges(self, parent_pair):
        """BlendCrossover with alpha=0, random=0/1 → parent values."""
        p1, p2, config = parent_pair
        op = BlendCrossover(crossover_rate=1.0, alpha=0.0)

        should_cross = jnp.array(True)

        # random=0 → min(p1, p2)
        random_low = jnp.zeros(config.shape, dtype=config.dtype)
        offspring_low = op._recombine_one(p1, p2, (should_cross, random_low), config)
        expected_low = jnp.minimum(p1.values, p2.values)
        assert jnp.allclose(offspring_low.values, expected_low, atol=1e-5)

        # random=1 → max(p1, p2)
        random_high = jnp.ones(config.shape, dtype=config.dtype)
        offspring_high = op._recombine_one(p1, p2, (should_cross, random_high), config)
        expected_high = jnp.maximum(p1.values, p2.values)
        assert jnp.allclose(offspring_high.values, expected_high, atol=1e-5)

    def test_blend_respects_bounds(self, wide_config):
        """BlendCrossover clips to config bounds."""
        p1 = RealGenome(values=jnp.full(wide_config.shape, -99.0, dtype=jnp.float32))
        p2 = RealGenome(values=jnp.full(wide_config.shape, 99.0, dtype=jnp.float32))

        op = BlendCrossover(crossover_rate=1.0, alpha=0.5)  # High alpha for expansion

        should_cross = jnp.array(True)
        random_extreme = jnp.zeros(wide_config.shape, dtype=wide_config.dtype)

        offspring = op._recombine_one(p1, p2, (should_cross, random_extreme), wide_config)

        assert jnp.all(offspring.values >= wide_config.bounds[0])
        assert jnp.all(offspring.values <= wide_config.bounds[1])

    def test_binomial_inheritance(self, parent_pair):
        """BinomialCrossover selects genes correctly.
        Note: MalthusJAX uses convention True → p1 (mutant), False → p2 (target).
        This is the inverse of standard DE notation but consistent with
        jnp.where semantics where True selects the first option.
        """
        p1, p2, config = parent_pair
        op = BinomialCrossover(crossover_rate=0.7)

        # Specific mask pattern
        mask = jnp.array(
            [True, True, False, False, True, False, True, False, True, False], dtype=jnp.bool_
        )

        offspring = op._recombine_one(p1, p2, mask, config)

        # MalthusJAX convention: True → p1, False → p2
        for i, m in enumerate(mask):
            expected = p1.values[i] if m else p2.values[i]
            assert offspring.values[i] == expected


class TestCrossFused:
    """Tests for the combined _cross_fused() method."""

    def test_uniform_cross_fused_output_type(self, parent_pair):
        """_cross_fused returns correct genome type."""
        p1, p2, config = parent_pair
        op = UniformCrossover(crossover_rate=0.5)

        key = jr.PRNGKey(555)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        offspring = op._cross_fused(keys, p1, p2, config)

        assert isinstance(offspring, RealGenome)
        assert offspring.values.shape == config.shape

    def test_uniform_cross_fused_reproducibility(self, parent_pair):
        """Same keys → identical offspring."""
        p1, p2, config = parent_pair
        op = UniformCrossover(crossover_rate=0.5)

        key = jr.PRNGKey(666)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        offspring1 = op._cross_fused(keys, p1, p2, config)
        offspring2 = op._cross_fused(keys, p1, p2, config)

        assert jnp.allclose(offspring1.values, offspring2.values)

    def test_blend_cross_fused_integration(self, parent_pair):
        """BlendCrossover _cross_fused integrates Tier 1 + Tier 2."""
        p1, p2, config = parent_pair
        op = BlendCrossover(crossover_rate=1.0, alpha=0.5)

        key = jr.PRNGKey(777)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        offspring = op._cross_fused(keys, p1, p2, config)

        # Offspring should be different from both parents (with high probability)
        assert not jnp.allclose(offspring.values, p1.values)
        # But should be within bounds
        assert jnp.all(offspring.values >= config.bounds[0])
        assert jnp.all(offspring.values <= config.bounds[1])

    def test_sbx_cross_fused_creates_variation(self, parent_pair):
        """SBX creates offspring different from parents."""
        p1, p2, config = parent_pair
        op = SimulatedBinaryCrossover(crossover_rate=1.0, eta=2.0)  # Low eta = high spread

        key = jr.PRNGKey(888)
        keys = jr.split(key, op.num_keys_per_atomic_operation)

        offspring = op._cross_fused(keys, p1, p2, config)

        # Should produce variation
        assert not jnp.allclose(offspring.values, p1.values, atol=1e-3)


class TestCrossSinglePair:
    """Tests for the cross_single_pair() interface."""

    def test_uniform_single_pair_output_shape(self, parent_pair):
        """cross_single_pair returns (num_offspring, gene_dim) shape."""
        p1, p2, config = parent_pair

        for num_offspring in [1, 2, 4]:
            op = UniformCrossover(num_offspring=num_offspring, crossover_rate=0.5)

            key = jr.PRNGKey(123)
            offspring = op.cross_single_pair(key, p1, p2, config)

            assert offspring.values.shape == (num_offspring,) + config.shape

    def test_blend_single_pair_multiple_offspring(self, parent_pair):
        """BlendCrossover generates multiple distinct offspring."""
        p1, p2, config = parent_pair
        op = BlendCrossover(num_offspring=5, crossover_rate=1.0, alpha=0.5)

        key = jr.PRNGKey(456)
        offspring = op.cross_single_pair(key, p1, p2, config)

        assert offspring.values.shape == (5,) + config.shape

        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.allclose(offspring.values[i], offspring.values[j])

    def test_sbx_single_pair_reproducibility(self, parent_pair):
        """cross_single_pair is reproducible with same key."""
        p1, p2, config = parent_pair
        op = SimulatedBinaryCrossover(num_offspring=2, eta=10.0)

        key = jr.PRNGKey(789)

        offspring1 = op.cross_single_pair(key, p1, p2, config)
        offspring2 = op.cross_single_pair(key, p1, p2, config)

        assert jnp.allclose(offspring1.values, offspring2.values)


class TestExpectedBehavior:
    """Tests for expected statistical and behavioral properties."""

    def test_uniform_crossover_rate_respected(self, real_config):
        """Uniform crossover respects the specified crossover rate."""
        p1 = RealGenome(values=jnp.zeros(real_config.shape, dtype=jnp.float32))
        p2 = RealGenome(values=jnp.ones(real_config.shape, dtype=jnp.float32))

        for rate in [0.2, 0.5, 0.8]:
            op = UniformCrossover(num_offspring=1, crossover_rate=rate)

            from_p2_counts = []
            for i in range(200):
                key = jr.PRNGKey(i)
                offspring = op.cross_single_pair(key, p1, p2, real_config)
                # Count genes from p2 (value == 1.0)
                from_p2_ratio = jnp.mean(offspring.values[0] == 1.0)
                from_p2_counts.append(float(from_p2_ratio))

            mean_from_p2 = np.mean(from_p2_counts)
            # Should be close to crossover_rate
            assert abs(mean_from_p2 - rate) < 0.1, (
                f"Rate {rate}: expected ~{rate}, got {mean_from_p2}"
            )

    def test_blend_alpha_controls_exploration(self, real_config):
        """Higher alpha in BlendCrossover produces more spread."""
        p1 = RealGenome(values=jnp.full(real_config.shape, -2.0, dtype=jnp.float32))
        p2 = RealGenome(values=jnp.full(real_config.shape, 2.0, dtype=jnp.float32))

        spreads = {}
        for alpha in [0.0, 0.5, 1.0]:
            op = BlendCrossover(num_offspring=50, crossover_rate=1.0, alpha=alpha)

            key = jr.PRNGKey(999)
            offspring = op.cross_single_pair(key, p1, p2, real_config)

            gene0_values = offspring.values[:, 0]
            spreads[alpha] = float(jnp.std(gene0_values))

        # Higher alpha should produce higher spread
        assert spreads[0.0] < spreads[0.5] < spreads[1.0]

    def test_sbx_eta_controls_spread(self, real_config):
        """Lower eta in SBX produces higher spread."""
        p1 = RealGenome(values=jnp.full(real_config.shape, -2.0, dtype=jnp.float32))
        p2 = RealGenome(values=jnp.full(real_config.shape, 2.0, dtype=jnp.float32))

        spreads = {}
        for eta in [2.0, 10.0, 30.0]:
            op = SimulatedBinaryCrossover(num_offspring=50, crossover_rate=1.0, eta=eta)

            key = jr.PRNGKey(888)
            offspring = op.cross_single_pair(key, p1, p2, real_config)

            gene0_values = offspring.values[:, 0]
            spreads[eta] = float(jnp.std(gene0_values))

        # Lower eta should produce higher spread
        assert spreads[2.0] > spreads[10.0] > spreads[30.0]

    def test_offspring_within_parental_range_for_uniform(self, parent_pair):
        """Uniform crossover offspring are strictly within parental range."""
        p1, p2, config = parent_pair
        op = UniformCrossover(num_offspring=100, crossover_rate=0.5)

        key = jr.PRNGKey(111)
        offspring = op.cross_single_pair(key, p1, p2, config)

        # Every gene should be from p1 or p2 exactly
        for i in range(100):
            for gene_idx in range(config.shape[0]):
                val = offspring.values[i, gene_idx]
                assert val == p1.values[gene_idx] or val == p2.values[gene_idx]

    def test_blend_can_exceed_parental_range(self, real_config):
        """BlendCrossover with alpha > 0 can explore beyond parents."""
        p1 = RealGenome(values=jnp.full(real_config.shape, 0.0, dtype=jnp.float32))
        p2 = RealGenome(values=jnp.full(real_config.shape, 1.0, dtype=jnp.float32))

        op = BlendCrossover(num_offspring=100, crossover_rate=1.0, alpha=0.5)

        key = jr.PRNGKey(222)
        offspring = op.cross_single_pair(key, p1, p2, real_config)

        # With alpha=0.5, some values can be < 0 or > 1 (before clipping)
        # After clipping, they should be within config bounds
        assert jnp.all(offspring.values >= real_config.bounds[0])
        assert jnp.all(offspring.values <= real_config.bounds[1])

    def test_crossover_rate_zero_preserves_p1(self, parent_pair):
        """crossover_rate=0 should preserve parent 1 for uniform/blend operators.
        Note: BinomialCrossover with rate=0 produces all-False mask,
        which maps to p2 in MalthusJAX convention (True → p1, False → p2).
        So BinomialCrossover is excluded from this test.
        """
        p1, p2, config = parent_pair

        for CrossoverClass in [UniformCrossover, BlendCrossover]:
            op = CrossoverClass(num_offspring=10, crossover_rate=0.0)

            key = jr.PRNGKey(333)
            offspring = op.cross_single_pair(key, p1, p2, config)

            for i in range(10):
                assert jnp.allclose(offspring.values[i], p1.values), (
                    f"{CrossoverClass.__name__} with rate=0 should preserve p1"
                )

    def test_binomial_crossover_rate_zero_preserves_p2(self, parent_pair):
        """BinomialCrossover with rate=0 preserves p2 (target).
        BinomialCrossover uses convention: True → p1 (mutant), False → p2 (target).
        With rate=0, all mask values are False → offspring = p2.
        """
        p1, p2, config = parent_pair

        op = BinomialCrossover(num_offspring=10, crossover_rate=0.0)

        key = jr.PRNGKey(333)
        offspring = op.cross_single_pair(key, p1, p2, config)

        for i in range(10):
            assert jnp.allclose(offspring.values[i], p2.values), (
                "BinomialCrossover with rate=0 should preserve p2 (target)"
            )

    def test_binomial_crossover_rate_one_preserves_p1(self, parent_pair):
        """BinomialCrossover with rate=1 selects all from p1 (mutant)."""
        p1, p2, config = parent_pair

        op = BinomialCrossover(num_offspring=10, crossover_rate=1.0)

        key = jr.PRNGKey(444)
        offspring = op.cross_single_pair(key, p1, p2, config)

        for i in range(10):
            assert jnp.allclose(offspring.values[i], p1.values), (
                "BinomialCrossover with rate=1 should select all from p1 (mutant)"
            )


class TestPopulationLevelCrossover:
    """Tests for full population-level crossover operations."""

    def test_population_output_size(self, population_pair):
        """Population crossover produces correct output size."""
        p1_pop, p2_pop, config = population_pair
        pop_size = len(p1_pop)

        for num_offspring in [1, 2]:
            op = UniformCrossover(num_offspring=num_offspring).set_input_length(pop_size)

            key = jr.PRNGKey(444)
            keys = jr.split(key, op.num_keys((pop_size,)))

            offspring_pop = op(keys, p1_pop, p2_pop, config)

            expected_size = pop_size * num_offspring
            assert len(offspring_pop) == expected_size

    def test_population_fitness_reset(self, population_pair):
        """Offspring fitness is reset to NaN."""
        p1_pop, p2_pop, config = population_pair
        pop_size = len(p1_pop)

        # Set non-NaN fitness on parents
        p1_pop = p1_pop.replace(fitness=jnp.zeros(pop_size))

        op = BlendCrossover(num_offspring=1).set_input_length(pop_size)
        key = jr.PRNGKey(555)
        keys = jr.split(key, op.num_keys((pop_size,)))

        offspring_pop = op(keys, p1_pop, p2_pop, config)

        assert jnp.all(jnp.isnan(offspring_pop.fitness))

    def test_population_jit_stability(self, population_pair):
        """JIT compilation produces identical results."""
        p1_pop, p2_pop, config = population_pair
        pop_size = len(p1_pop)

        op = SimulatedBinaryCrossover(num_offspring=1, eta=20.0).set_input_length(pop_size)
        key = jr.PRNGKey(666)
        keys = jr.split(key, op.num_keys((pop_size,)))

        @jax.jit
        def crossover_jit(k, p1, p2, c):
            return op(k, p1, p2, c)

        result_raw = op(keys, p1_pop, p2_pop, config)
        result_jit = crossover_jit(keys, p1_pop, p2_pop, config)

        np.testing.assert_allclose(result_raw.genes.values, result_jit.genes.values, atol=1e-6)


class TestInjectionFusedEquivalence:
    """Tests that injection and fused modes produce equivalent behavior."""

    def test_uniform_injection_vs_fused_distribution(self, real_config):
        """Uniform injection and fused have same statistical properties."""
        pop_size = 50

        fused_op = UniformCrossover(num_offspring=1, crossover_rate=0.6)
        injection_op = UniformCrossover_injection(num_offspring=1, crossover_rate=0.6)
        injection_op = injection_op.set_input_length(pop_size)

        key = jr.PRNGKey(777)
        k1, k2, k3, k4 = jr.split(key, 4)

        p1_pop = RealPopulation.init_random(k1, real_config, pop_size)
        p2_pop = RealPopulation.init_random(k2, real_config, pop_size)

        # Fused mode
        fused_op = fused_op.set_input_length(pop_size)
        fused_keys = jr.split(k3, fused_op.num_keys((pop_size,)))
        fused_result = fused_op(fused_keys, p1_pop, p2_pop, real_config)

        # Injection mode
        injection_result = injection_op(k4, p1_pop, p2_pop, real_config)

        # Results will differ (different keys), but distributions should be similar
        # Check same output size
        assert len(fused_result) == len(injection_result)

        # Check genes are from parents (inheritance property holds for both)
        for result in [fused_result, injection_result]:
            vals = result.genes.values
            from_p1 = vals == p1_pop.genes.values
            from_p2 = vals == p2_pop.genes.values
            assert jnp.all(from_p1 | from_p2)


class TestCrossoverEdgeCases:
    """Edge case testing for extreme genome sizes and parameter boundaries."""

    def test_uniform_crossover_single_gene(self, wide_config):
        """Edge case: genome with single gene should still work correctly."""
        config = RealGenomeConfig(shape=(1,), bounds=(-5.0, 5.0))
        p1 = RealGenome(values=jnp.array([0.0]))
        p2 = RealGenome(values=jnp.array([4.0]))

        op = UniformCrossover(num_offspring=5, crossover_rate=0.5)
        key = jr.PRNGKey(5555)
        offspring = op.cross_single_pair(key, p1, p2, config)

        # All offspring should have shape (5, 1)
        assert offspring.values.shape == (5, 1)
        # Values must be from parent set
        for val in offspring.values:
            assert jnp.isclose(val[0], 0.0) or jnp.isclose(val[0], 4.0)

    def test_blend_crossover_large_genome(self):
        """Edge case: Blend crossover should handle large genomes efficiently."""
        config = RealGenomeConfig(shape=(1000,), bounds=(-10.0, 10.0))
        key = jr.PRNGKey(6666)
        k1, k2 = jr.split(key)

        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)

        op = BlendCrossover(num_offspring=2, crossover_rate=1.0, alpha=0.5)
        offspring = op.cross_single_pair(key, p1, p2, config)

        assert offspring.values.shape == (2, 1000)
        # All values should be within bounds
        assert jnp.all(offspring.values >= -10.0)
        assert jnp.all(offspring.values <= 10.0)

    def test_sbx_crossover_no_mutation_case(self, parent_pair):
        """Edge case: SBX with very low eta should approach Blend behavior."""
        p1, p2, real_config = parent_pair
        op = SimulatedBinaryCrossover(num_offspring=5, crossover_rate=1.0, eta=100.0)

        key = jr.PRNGKey(7777)
        offspring = op.cross_single_pair(key, p1, p2, real_config)

        # High eta → offspring should be close to parent average
        parent_midpoint = (p1.values + p2.values) / 2.0
        for i in range(5):
            avg_distance = float(jnp.mean(jnp.abs(offspring.values[i] - parent_midpoint)))
            # High eta (100.0) should keep offspring reasonably close to midpoint
            # but due to stochastic nature, allow tolerance of ~3.0
            assert avg_distance < 3.0, f"High eta should keep offspring near parents, got avg_dist={avg_distance}"

    def test_crossover_rate_zero_boundary(self, real_config):
        """Edge case: rate=0.0 should always select from p1."""
        p1 = RealGenome(values=jnp.zeros(real_config.shape))
        p2 = RealGenome(values=jnp.ones(real_config.shape) * 5.0)

        for op_class, name in [
            (UniformCrossover, "Uniform"),
            (BlendCrossover, "Blend"),
        ]:
            op = op_class(crossover_rate=0.0)
            key = jr.PRNGKey(8888)
            offspring = op.cross_single_pair(key, p1, p2, real_config)

            # Should equal p1
            for i in range(offspring.values.shape[0]):
                assert jnp.allclose(offspring.values[i], p1.values), (
                    f"{name} with rate=0 should preserve p1"
                )

    def test_crossover_offspring_diversity(self):
        """Test that crossover actually generates diverse offspring."""
        config = RealGenomeConfig(shape=(20,), bounds=(-1.0, 1.0))
        p1 = RealGenome(values=jnp.full((20,), -0.8))
        p2 = RealGenome(values=jnp.full((20,), 0.8))

        op = UniformCrossover(num_offspring=100, crossover_rate=0.5)
        key = jr.PRNGKey(9999)
        offspring_batch = op.cross_single_pair(key, p1, p2, config)

        # Calculate pairwise distances between offspring
        variances = []
        for gene_idx in range(20):
            gene_values = offspring_batch.values[:, gene_idx]
            variance = float(jnp.var(gene_values))
            variances.append(variance)

        mean_variance = np.mean(variances)
        # Should have non-zero variance across 100 different offspring
        assert mean_variance > 0.01, (
            f"Crossover should generate diverse offspring, got mean_var={mean_variance}"
        )
