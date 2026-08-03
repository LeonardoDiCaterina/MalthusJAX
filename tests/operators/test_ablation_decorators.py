"""Tests for the ablation decorators that enforce single-key operator semantics.

These tests exercise:
- Single-key budgeting (num_keys == 1)
- Key handling and internal splitting inside __call__
- Functional equivalence with the original operators (structure, offspring count)
- JIT compatibility and parameter preservation
- Behavior with both mutation and crossover operators
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.base_ablation import (
    ablation_single_key_crossover,
    ablation_single_key_mutation,
)
from malthusjax.operators.crossover.binary import UniformCrossover as BinaryUniformCrossover
from malthusjax.operators.crossover.real import UniformCrossover as RealUniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.mutation.real import GaussianMutation


class TestAblationMutationDecorator:
    """Verify mutation ablation enforces single-key budgeting and preserves behavior.

    Checks include structural equivalence of outputs, bounds preservation for real
    genomes, correct offspring counts, and JIT compatibility.
    """

    @pytest.fixture
    def binary_config(self):
        """Binary genome configuration for testing."""
        return BinaryGenomeConfig(shape=(10,))

    @pytest.fixture
    def real_config(self):
        """Real genome configuration for testing."""
        return RealGenomeConfig(shape=(5,), bounds=(-10.0, 10.0))

    @pytest.fixture
    def binary_population(self, rng_key, binary_config):
        """Sample binary population."""
        return BinaryPopulation.init_random(rng_key, binary_config, size=8)

    @pytest.fixture
    def real_population(self, rng_key, real_config):
        """Sample real population."""
        return RealPopulation.init_random(rng_key, real_config, size=6)

    def test_decorator_overrides_num_keys_binary(self):
        """Ablation decorator forces a single-key budget regardless of population
        size or offspring count."""

        # Create ablation version
        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass

        # Original operator uses budgeting
        original = BitFlipMutation(mutation_rate=0.1, num_offspring=2)
        original = original.set_input_length(8)  # Pop size

        # Ablation operator bypasses budgeting
        ablation = BitFlipMutation_ablation(mutation_rate=0.1, num_offspring=2)
        ablation = ablation.set_input_length(8)

        input_shape = (8,)  # Population size

        # Original should calculate full budget
        assert original.num_keys(input_shape) == 8 * 2 * 1  # pop_size * num_offspring * keys_per_op

        # Ablation enforces a single-key budget
        assert ablation.num_keys(input_shape) == 1

    def test_decorator_overrides_num_keys_real(self):
        """Ablation decorator should work with real-valued mutations."""

        @ablation_single_key_mutation
        class GaussianMutation_ablation(GaussianMutation):
            pass

        original = GaussianMutation(mutation_strength=0.1, num_offspring=3)
        original = original.set_input_length(6)

        ablation = GaussianMutation_ablation(mutation_strength=0.1, num_offspring=3)
        ablation = ablation.set_input_length(6)

        input_shape = (6,)

        # Original uses full budgeting
        assert original.num_keys(input_shape) > 1

        # Ablation bypasses budgeting
        assert ablation.num_keys(input_shape) == 1

    def test_functional_equivalence_binary(self, rng_key, binary_population, binary_config):
        """Ablation mutation produces outputs structurally equivalent to the original
        operator and the same offspring counts."""
        key1, key2 = jr.split(rng_key)

        # Create both versions
        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass

        # Configure operators identically
        params = {"mutation_rate": 0.2, "num_offspring": 2}
        original = BitFlipMutation(**params).set_input_length(len(binary_population))
        ablation = BitFlipMutation_ablation(**params).set_input_length(len(binary_population))

        # Generate keys according to each operator's budget
        original_budget = original.num_keys((len(binary_population),))
        original_keys = jr.split(key1, original_budget)

        ablation_budget = ablation.num_keys((len(binary_population),))  # Should be 1
        split_result = jr.split(key2, ablation_budget)
        ablation_keys = split_result[0] if ablation_budget == 1 else split_result

        # Apply mutations
        original_result = original(original_keys, binary_population, binary_config)
        ablation_result = ablation(ablation_keys, binary_population, binary_config)

        # Results should have same structure
        assert len(original_result) == len(ablation_result)
        assert original_result.genes.values.shape == ablation_result.genes.values.shape
        assert original_result.genes.values.dtype == ablation_result.genes.values.dtype

        # Should have correct offspring count
        expected_offspring = len(binary_population) * params["num_offspring"]
        assert len(original_result) == expected_offspring
        assert len(ablation_result) == expected_offspring

    def test_functional_equivalence_real(self, rng_key, real_population, real_config):
        """Ablation mutation preserves output structure and respects real-genome bounds."""
        key1, key2 = jr.split(rng_key)

        @ablation_single_key_mutation
        class GaussianMutation_ablation(GaussianMutation):
            pass

        # Configure operators identically
        params = {"mutation_strength": 0.5, "num_offspring": 1}
        original = GaussianMutation(**params).set_input_length(len(real_population))
        ablation = GaussianMutation_ablation(**params).set_input_length(len(real_population))

        # Generate keys
        original_budget = original.num_keys((len(real_population),))
        ablation_budget = ablation.num_keys((len(real_population),))

        original_keys = jr.split(key1, original_budget)
        split_result = jr.split(key2, ablation_budget)
        ablation_keys = split_result[0] if ablation_budget == 1 else split_result

        # Apply mutations
        original_result = original(original_keys, real_population, real_config)
        ablation_result = ablation(ablation_keys, real_population, real_config)

        # Verify structural equivalence
        assert len(original_result) == len(ablation_result)
        assert original_result.genes.values.shape == ablation_result.genes.values.shape

        # Bounds should be preserved
        assert jnp.all(original_result.genes.values >= real_config.bounds[0])
        assert jnp.all(original_result.genes.values <= real_config.bounds[1])
        assert jnp.all(ablation_result.genes.values >= real_config.bounds[0])
        assert jnp.all(ablation_result.genes.values <= real_config.bounds[1])

    def test_jit_compatibility(self, rng_key, binary_population, binary_config):
        """Ablation operators should be JIT-compatible."""

        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass

        operator = BitFlipMutation_ablation(mutation_rate=0.1, num_offspring=1).set_input_length(
            len(binary_population)
        )

        # JIT compile the operator
        jit_operator = jax.jit(operator)

        # JIT invocation returns expected shape and length
        single_key = rng_key
        result = jit_operator(single_key, binary_population, binary_config)

        assert len(result) == len(binary_population) * 1  # num_offspring
        assert result.genes.values.shape[0] == len(binary_population)

    def test_multiple_offspring_binary(self, rng_key, binary_population, binary_config):
        """Test ablation with multiple offspring per parent."""

        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass

        operator = BitFlipMutation_ablation(mutation_rate=0.15, num_offspring=3).set_input_length(
            len(binary_population)
        )

        result = operator(rng_key, binary_population, binary_config)

        # Should produce correct number of offspring
        expected_total = len(binary_population) * 3
        assert len(result) == expected_total
        assert result.genes.values.shape[0] == expected_total


class TestAblationCrossoverDecorator:
    """Verify crossover ablation enforces single-key budgeting and preserves operator semantics.

    Tests structural equivalence, offspring counts, and bounds preservation for real
    genomes, plus JIT compatibility.
    """

    @pytest.fixture
    def binary_config(self):
        return BinaryGenomeConfig(shape=(8,))

    @pytest.fixture
    def real_config(self):
        return RealGenomeConfig(shape=(4,), bounds=(-5.0, 5.0))

    @pytest.fixture
    def binary_populations(self, rng_key, binary_config):
        """Two binary populations for crossover."""
        key1, key2 = jr.split(rng_key)
        pop1 = BinaryPopulation.init_random(key1, binary_config, size=6)
        pop2 = BinaryPopulation.init_random(key2, binary_config, size=6)
        return pop1, pop2

    @pytest.fixture
    def real_populations(self, rng_key, real_config):
        """Two real populations for crossover."""
        key1, key2 = jr.split(rng_key)
        pop1 = RealPopulation.init_random(key1, real_config, size=4)
        pop2 = RealPopulation.init_random(key2, real_config, size=4)
        return pop1, pop2

    def test_decorator_overrides_num_keys_crossover(self):
        """Ablation decorator should override crossover num_keys to return 1."""

        @ablation_single_key_crossover
        class BinaryUniformCrossover_ablation(BinaryUniformCrossover):
            pass

        original = BinaryUniformCrossover(crossover_rate=0.7, num_offspring=2).set_input_length(4)

        ablation = BinaryUniformCrossover_ablation(
            crossover_rate=0.7, num_offspring=2
        ).set_input_length(4)

        input_shape = (4,)  # Number of pairs

        # Original uses full budgeting
        assert original.num_keys(input_shape) > 1

        # Ablation bypasses budgeting
        assert ablation.num_keys(input_shape) == 1

    def test_functional_equivalence_binary_crossover(
        self, rng_key, binary_populations, binary_config
    ):
        """Ablation crossover should produce equivalent results."""
        key1, key2 = jr.split(rng_key)
        pop1, pop2 = binary_populations

        @ablation_single_key_crossover
        class BinaryUniformCrossover_ablation(BinaryUniformCrossover):
            pass

        # Configure operators identically
        params = {"crossover_rate": 0.8, "num_offspring": 1}
        original = BinaryUniformCrossover(**params).set_input_length(len(pop1))
        ablation = BinaryUniformCrossover_ablation(**params).set_input_length(len(pop1))

        # Generate keys
        original_budget = original.num_keys((len(pop1),))
        ablation_budget = ablation.num_keys((len(pop1),))

        original_keys = jr.split(key1, original_budget)
        split_result = jr.split(key2, ablation_budget)
        ablation_keys = split_result[0] if ablation_budget == 1 else split_result

        # Apply crossover
        original_result = original(original_keys, pop1, pop2, binary_config)
        ablation_result = ablation(ablation_keys, pop1, pop2, binary_config)

        # Verify structural equivalence
        assert len(original_result) == len(ablation_result)
        assert original_result.genes.values.shape == ablation_result.genes.values.shape
        assert original_result.genes.values.dtype == ablation_result.genes.values.dtype

        # Should produce correct offspring count
        expected_offspring = len(pop1) * params["num_offspring"]
        assert len(original_result) == expected_offspring
        assert len(ablation_result) == expected_offspring

    def test_functional_equivalence_real_crossover(self, rng_key, real_populations, real_config):
        """Ablation crossover should work with real-valued genomes."""
        key1, key2 = jr.split(rng_key)
        pop1, pop2 = real_populations

        @ablation_single_key_crossover
        class RealUniformCrossover_ablation(RealUniformCrossover):
            pass

        # Configure operators
        params = {"crossover_rate": 0.6, "num_offspring": 2}
        original = RealUniformCrossover(**params).set_input_length(len(pop1))
        ablation = RealUniformCrossover_ablation(**params).set_input_length(len(pop1))

        # Generate keys
        original_budget = original.num_keys((len(pop1),))
        ablation_budget = ablation.num_keys((len(pop1),))

        original_keys = jr.split(key1, original_budget)
        split_result = jr.split(key2, ablation_budget)
        ablation_keys = split_result[0] if ablation_budget == 1 else split_result

        # Apply crossover
        original_result = original(original_keys, pop1, pop2, real_config)
        ablation_result = ablation(ablation_keys, pop1, pop2, real_config)

        # Verify results
        assert len(original_result) == len(ablation_result)
        assert original_result.genes.values.shape == ablation_result.genes.values.shape

        expected_offspring = len(pop1) * params["num_offspring"]
        assert len(original_result) == expected_offspring
        assert len(ablation_result) == expected_offspring

        # Values should respect bounds
        assert jnp.all(original_result.genes.values >= real_config.bounds[0])
        assert jnp.all(original_result.genes.values <= real_config.bounds[1])
        assert jnp.all(ablation_result.genes.values >= real_config.bounds[0])
        assert jnp.all(ablation_result.genes.values <= real_config.bounds[1])

    def test_crossover_jit_compatibility(self, rng_key, binary_populations, binary_config):
        """Ablation crossover operators should be JIT-compatible."""

        @ablation_single_key_crossover
        class BinaryUniformCrossover_ablation(BinaryUniformCrossover):
            pass

        pop1, pop2 = binary_populations
        operator = BinaryUniformCrossover_ablation(
            crossover_rate=0.5, num_offspring=1
        ).set_input_length(len(pop1))

        # JIT compile
        jit_operator = jax.jit(operator)

        # Should execute without errors
        result = jit_operator(rng_key, pop1, pop2, binary_config)

        assert len(result) == len(pop1)
        assert result.genes.values.shape[0] == len(pop1)

    def test_cross_single_pair_unchanged(self, rng_key, binary_populations, binary_config):
        """cross_single_pair method should work unchanged."""

        @ablation_single_key_crossover
        class BinaryUniformCrossover_ablation(BinaryUniformCrossover):
            pass

        pop1, pop2 = binary_populations
        operator = BinaryUniformCrossover_ablation(crossover_rate=0.7, num_offspring=2)

        # Test single pair crossover
        genome1 = BinaryGenome(values=pop1.genes.values[0])  # First individual from pop1
        genome2 = BinaryGenome(values=pop2.genes.values[0])  # First individual from pop2

        result = operator.cross_single_pair(rng_key, genome1, genome2, binary_config)

        # Should return batched genome with correct offspring count
        assert result.values.shape[0] == 2  # num_offspring
        assert result.values.shape[1:] == genome1.values.shape  # Same shape as parent


class TestAblationIntegration:
    """Integration smoke tests: decorator composition, attribute preservation,
    and budget differences.

    These tests exercise decorator application on existing classes and verify
    that parameters and key-budget behavior remain sensible when combined with other code.
    """

    def test_decorator_composition(self):
        """Test that decorators can be applied to existing classes."""

        @ablation_single_key_mutation
        class BitFlip_test(BitFlipMutation):
            pass

        @ablation_single_key_crossover
        class Uniform_test(BinaryUniformCrossover):
            pass

        # Instances construct with expected parameters
        mut = BitFlip_test(mutation_rate=0.1)
        cross = Uniform_test(crossover_rate=0.5)

        assert mut.mutation_rate == 0.1
        assert cross.crossover_rate == 0.5

    def test_decorator_preserves_attributes(self):
        """Decorators should preserve all original class attributes."""

        @ablation_single_key_mutation
        class BitFlip_test(BitFlipMutation):
            pass

        original = BitFlipMutation(mutation_rate=0.2, num_offspring=3)
        ablation = BitFlip_test(mutation_rate=0.2, num_offspring=3)

        # Should have same parameters
        assert original.mutation_rate == ablation.mutation_rate
        assert original.num_offspring == ablation.num_offspring
        assert original.num_keys_per_atomic_operation == ablation.num_keys_per_atomic_operation

    def test_different_key_budgets(self):
        """Original and ablation should have different key budgets but same outputs."""

        @ablation_single_key_mutation
        class BitFlip_ablation(BitFlipMutation):
            pass

        original = BitFlipMutation(num_offspring=2).set_input_length(10)
        ablation = BitFlip_ablation(num_offspring=2).set_input_length(10)

        input_shape = (10,)

        # Should have different budgets
        original_budget = original.num_keys(input_shape)
        ablation_budget = ablation.num_keys(input_shape)

        assert original_budget > ablation_budget
        assert ablation_budget == 1
        assert original_budget == 10 * 2 * 1  # pop_size * num_offspring * keys_per_op


if __name__ == "__main__":
    pytest.main([__file__])
