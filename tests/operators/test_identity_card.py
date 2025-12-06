"""
Unit tests for the Identity Card operator interface (Step 1).

Tests verify:
1. Default implementations delegate to legacy __call__
2. num_keys returns correct counts
3. get_output_shape computes correct shapes
4. apply_kernel maintains backward compatibility
5. Operators can override for kernel optimization
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def rng_key():
    """Provide a test PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def real_config():
    """Provide a real genome config for testing."""
    return RealGenomeConfig(length=10, bounds=(-5.0, 5.0))


@pytest.fixture
def real_genome(rng_key, real_config):
    """Provide a sample real genome."""
    return RealGenome.random_init(rng_key, real_config)


@pytest.fixture
def fitness_array():
    """Provide a sample fitness array."""
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ============================================================================
# CONCRETE TEST OPERATORS (for testing base classes)
# ============================================================================

@struct.dataclass
class SimpleMutation(BaseMutation):
    """Simple mutation for testing (adds Gaussian noise)."""
    mutation_strength: float = 0.1

    def _mutate_one(self, key, genome, config):
        noise = jax.random.normal(key, genome.values.shape) * self.mutation_strength
        return genome.replace(values=genome.values + noise)


@struct.dataclass
class SimpleCrossover(BaseCrossover):
    """Simple uniform crossover for testing."""
    crossover_rate: float = 0.5

    def _cross_one(self, key, p1, p2, config):
        mask = jax.random.uniform(key, p1.values.shape) < self.crossover_rate
        child_data = jnp.where(mask, p1.values, p2.values)
        return p1.replace(values=child_data)


@struct.dataclass
class SimpleSelection(BaseSelection):
    """Simple top-k selection for testing."""

    def __call__(self, key, fitness):
        # Select top num_selections
        return jnp.argsort(fitness)[-self.num_selections:]


# ============================================================================
# TESTS: BaseMutation Identity Card
# ============================================================================

class TestBaseMutationIdentityCard:
    """Test Identity Card methods on BaseMutation."""

    def test_num_keys_default(self, real_config):
        """Test num_keys returns num_offspring by default."""
        mutation = SimpleMutation(num_offspring=3)
        input_shape = (10,)

        num_keys = mutation.num_keys(real_config, input_shape)

        assert num_keys == 3, "num_keys should equal num_offspring"

    def test_num_keys_single_offspring(self, real_config):
        """Test num_keys with single offspring."""
        mutation = SimpleMutation(num_offspring=1)
        input_shape = (10,)

        num_keys = mutation.num_keys(real_config, input_shape)

        assert num_keys == 1

    def test_get_output_shape_default(self, real_config):
        """Test get_output_shape computes correct shape."""
        mutation = SimpleMutation(num_offspring=3)
        input_shape = (10,)

        output_shape = mutation.get_output_shape(real_config, input_shape)

        assert output_shape == (3, 10), "Shape should be (num_offspring, *input_shape)"

    def test_get_output_shape_multidim(self, real_config):
        """Test get_output_shape with multi-dimensional input."""
        mutation = SimpleMutation(num_offspring=2)
        input_shape = (5, 4)

        output_shape = mutation.get_output_shape(real_config, input_shape)

        assert output_shape == (2, 5, 4)

    def test_apply_kernel_delegates_to_call(self, rng_key, real_genome, real_config):
        """Test apply_kernel delegates to __call__ by default."""
        mutation = SimpleMutation(num_offspring=2)
        keys = jax.random.split(rng_key, 2)

        # Call both methods
        result_kernel = mutation.apply_kernel(keys, real_genome, real_config)
        result_call = mutation(keys[0], real_genome, real_config)

        # Both should produce valid output shapes
        assert result_kernel.values.shape[0] == 2, "apply_kernel should produce 2 offspring"
        assert result_call.values.shape[0] == 2, "__call__ should produce 2 offspring"

    def test_apply_kernel_backward_compatible(self, rng_key, real_genome, real_config):
        """Test apply_kernel maintains backward compatibility."""
        mutation = SimpleMutation(num_offspring=1)
        keys = jax.random.split(rng_key, 1)

        # Should not crash and produce valid output
        result = mutation.apply_kernel(keys, real_genome, real_config)

        assert result.values.shape == (1, 10), "Output should be (1, genome_length)"


# ============================================================================
# TESTS: BaseCrossover Identity Card
# ============================================================================

class TestBaseCrossoverIdentityCard:
    """Test Identity Card methods on BaseCrossover."""

    def test_num_keys_default(self, real_config):
        """Test num_keys returns num_offspring by default."""
        crossover = SimpleCrossover(num_offspring=4)
        input_shape = (10,)

        num_keys = crossover.num_keys(real_config, input_shape)

        assert num_keys == 4

    def test_get_output_shape_default(self, real_config):
        """Test get_output_shape computes correct shape."""
        crossover = SimpleCrossover(num_offspring=2)
        input_shape = (10,)

        output_shape = crossover.get_output_shape(real_config, input_shape)

        assert output_shape == (2, 10)

    def test_apply_kernel_delegates_to_call(self, rng_key, real_genome, real_config):
        """Test apply_kernel delegates to __call__ by default."""
        crossover = SimpleCrossover(num_offspring=2)
        keys = jax.random.split(rng_key, 2)

        # Create two parent genomes
        key1, key2 = jax.random.split(rng_key, 2)
        p1 = RealGenome.random_init(key1, real_config)
        p2 = RealGenome.random_init(key2, real_config)

        # Call both methods
        result_kernel = crossover.apply_kernel(keys, p1, p2, real_config)
        result_call = crossover(keys[0], p1, p2, real_config)

        # Both should produce valid output shapes
        assert result_kernel.values.shape[0] == 2
        assert result_call.values.shape[0] == 2

    def test_apply_kernel_backward_compatible(self, rng_key, real_genome, real_config):
        """Test apply_kernel maintains backward compatibility."""
        crossover = SimpleCrossover(num_offspring=1)
        keys = jax.random.split(rng_key, 1)

        key1, key2 = jax.random.split(rng_key, 2)
        p1 = RealGenome.random_init(key1, real_config)
        p2 = RealGenome.random_init(key2, real_config)

        result = crossover.apply_kernel(keys, p1, p2, real_config)

        assert result.values.shape == (1, 10)


# ============================================================================
# TESTS: BaseSelection Identity Card
# ============================================================================

class TestBaseSelectionIdentityCard:
    """Test Identity Card methods on BaseSelection."""

    def test_num_keys_default(self):
        """Test num_keys returns 1 by default for selection."""
        selection = SimpleSelection(num_selections=3)
        input_shape = (10,)  # population size

        num_keys = selection.num_keys(input_shape)

        assert num_keys == 1, "Selection should need 1 key by default"

    def test_get_output_shape_default(self):
        """Test get_output_shape returns (num_selections,)."""
        selection = SimpleSelection(num_selections=5)
        input_shape = (20,)  # population size

        output_shape = selection.get_output_shape(input_shape)

        assert output_shape == (5,), "Output should be (num_selections,)"

    def test_apply_kernel_delegates_to_call(self, rng_key, fitness_array):
        """Test apply_kernel delegates to __call__ by default."""
        selection = SimpleSelection(num_selections=3)
        keys = jax.random.split(rng_key, 1)

        # Call both methods
        result_kernel = selection.apply_kernel(keys, fitness_array)
        result_call = selection(keys[0], fitness_array)

        # Both should produce valid indices
        assert result_kernel.shape == (3,)
        assert result_call.shape == (3,)
        chex.assert_trees_all_equal(result_kernel, result_call)

    def test_apply_kernel_backward_compatible(self, rng_key, fitness_array):
        """Test apply_kernel maintains backward compatibility."""
        selection = SimpleSelection(num_selections=2)
        keys = jax.random.split(rng_key, 1)

        result = selection.apply_kernel(keys, fitness_array)

        assert result.shape == (2,)
        assert jnp.all(result >= 0) and jnp.all(result < len(fitness_array))


# ============================================================================
# TESTS: Custom Kernel Override
# ============================================================================

@struct.dataclass
class FastMutation(SimpleMutation):
    """Custom mutation with overridden kernel for testing."""

    def num_keys(self, config, input_shape):
        """Override: need one key per gene per offspring."""
        genome_length = input_shape[0] if len(input_shape) > 0 else 1
        return self.num_offspring * genome_length

    def apply_kernel(self, keys, genome, config):
        """Override: fused kernel implementation."""
        # keys has shape (num_keys, 2) where num_keys = num_offspring * genome_length
        # We need to reshape to (num_offspring, genome_length, 2) for vectorization
        genome_length = genome.values.shape[0]
        keys_reshaped = keys.reshape(self.num_offspring, genome_length, 2)

        def mutate_with_keys(key_row, data):
            noise = jax.random.normal(key_row[0], data.shape) * self.mutation_strength
            return data + noise

        # Vectorized mutation
        mutated_data = jax.vmap(mutate_with_keys, in_axes=(0, None))(
            keys_reshaped, genome.values
        )

        return genome.replace(values=mutated_data)


class TestCustomKernelOverride:
    """Test that operators can override kernel methods."""

    def test_custom_num_keys(self, real_config):
        """Test custom num_keys implementation."""
        mutation = FastMutation(num_offspring=2)
        input_shape = (10,)

        num_keys = mutation.num_keys(real_config, input_shape)

        assert num_keys == 20, "Custom num_keys should return 2 * 10 = 20"

    def test_custom_apply_kernel(self, rng_key, real_genome, real_config):
        """Test custom apply_kernel implementation."""
        mutation = FastMutation(num_offspring=3)
        num_keys = mutation.num_keys(real_config, (10,))
        keys = jax.random.split(rng_key, num_keys)

        result = mutation.apply_kernel(keys, real_genome, real_config)

        assert result.values.shape == (3, 10), "Custom kernel should produce correct shape"
        # Verify mutation actually happened (data changed)
        assert not jnp.allclose(result.values, real_genome.values), "Mutation should change data"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIdentityCardIntegration:
    """Integration tests for Identity Card interface."""

    def test_all_operators_have_identity_card(self):
        """Test that all base operators expose Identity Card methods."""
        mutation = SimpleMutation(num_offspring=1)
        crossover = SimpleCrossover(num_offspring=1)
        selection = SimpleSelection(num_selections=1)

        # Check mutation
        assert hasattr(mutation, 'num_keys')
        assert hasattr(mutation, 'get_output_shape')
        assert hasattr(mutation, 'apply_kernel')
        assert callable(mutation.num_keys)
        assert callable(mutation.get_output_shape)
        assert callable(mutation.apply_kernel)

        # Check crossover
        assert hasattr(crossover, 'num_keys')
        assert hasattr(crossover, 'get_output_shape')
        assert hasattr(crossover, 'apply_kernel')

        # Check selection
        assert hasattr(selection, 'num_keys')
        assert hasattr(selection, 'get_output_shape')
        assert hasattr(selection, 'apply_kernel')

    def test_jit_compatibility(self, rng_key, real_genome, real_config):
        """Test that Identity Card methods work with JAX JIT."""
        mutation = SimpleMutation(num_offspring=2)

        @jax.jit
        def jitted_kernel(keys, genome, config):
            return mutation.apply_kernel(keys, genome, config)

        keys = jax.random.split(rng_key, 2)
        result = jitted_kernel(keys, real_genome, real_config)

        assert result.values.shape == (2, 10), "JIT-compiled kernel should work"

    def test_vmap_compatibility(self, rng_key, real_genome, real_config):
        """Test that Identity Card methods work with vmap."""
        mutation = SimpleMutation(num_offspring=1)

        # Create batch of genomes
        batch_size = 5
        keys_batch = jax.random.split(rng_key, batch_size)
        genome_batch = jax.vmap(lambda k: RealGenome.random_init(k, real_config))(keys_batch)

        # Vmap over apply_kernel
        keys = jax.random.split(rng_key, batch_size)
        result = jax.vmap(
            lambda k, g: mutation.apply_kernel(jnp.array([k]), g, real_config),
            in_axes=(0, 0)
        )(keys, genome_batch)

        assert result.values.shape == (5, 1, 10), "Vmap should batch correctly"


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class TestIdentityCardProperties:
    """Property-based tests for Identity Card invariants."""

    @pytest.mark.parametrize("num_offspring", [1, 2, 5, 10])
    def test_num_keys_matches_offspring(self, num_offspring, real_config):
        """Test num_keys equals num_offspring for default implementations."""
        mutation = SimpleMutation(num_offspring=num_offspring)
        crossover = SimpleCrossover(num_offspring=num_offspring)

        assert mutation.num_keys(real_config, (10,)) == num_offspring
        assert crossover.num_keys(real_config, (10,)) == num_offspring

    @pytest.mark.parametrize("genome_length", [5, 10, 20, 50])
    def test_output_shape_preserves_dims(self, genome_length, real_config):
        """Test output shape preserves genome dimensions."""
        mutation = SimpleMutation(num_offspring=2)
        input_shape = (genome_length,)

        output_shape = mutation.get_output_shape(real_config, input_shape)

        assert output_shape == (2, genome_length)
        assert len(output_shape) == len(input_shape) + 1  # adds offspring dim

    @pytest.mark.parametrize("num_selections", [1, 3, 5, 10])
    def test_selection_output_shape(self, num_selections):
        """Test selection output shape matches num_selections."""
        selection = SimpleSelection(num_selections=num_selections)
        input_shape = (100,)  # population size

        output_shape = selection.get_output_shape(input_shape)

        assert output_shape == (num_selections,)
