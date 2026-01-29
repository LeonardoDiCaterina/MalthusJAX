import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation


@pytest.fixture
def binary_context():
    """Sets up a standard binary population, config, and RNG key."""
    key = jr.PRNGKey(42)
    # Using a 10-bit shape for easy verification
    config = BinaryGenomeConfig(shape=(10,), p=0.5, dtype=jnp.int32)
    pop_size = 8
    population = BinaryPopulation.init_random(key, config, size=pop_size)
    return population, config, key


class TestBinaryMutationHarness:
    """Validates 3-Tier Binary Operators and ResourceMapper integration."""

    @pytest.mark.parametrize(
        "mut_cls, expected_keys", [(BitFlipMutation, 1), (ScrambleMutation, 2), (SwapMutation, 3)]
    )
    def test_key_budgeting(self, mut_cls, expected_keys):
        """Verifies num_keys_per_atomic_operation matches implementation."""
        mut = mut_cls()
        # Ensure the budget matches Tier 2 requirements
        assert mut.num_keys_per_atomic_operation == expected_keys

    def test_bit_flip_logic(self, binary_context):
        """Tests that BitFlip actually toggles bits via XOR logic."""
        pop, config, key = binary_context
        # Set mutation_rate to 1.0 to ensure all bits flip
        mut = BitFlipMutation(mutation_rate=1.0).set_input_length(len(pop))

        n_keys = mut.num_keys((len(pop),))
        all_keys = jr.split(key, n_keys)

        offspring_pop = mut(all_keys, pop, config)

        # In a BitFlip with rate 1.0, new_bits = 1 - old_bits
        # XOR logic check
        expected_values = 1 - pop.genes.values
        assert jnp.all(offspring_pop.genes.values == expected_values)

    def test_swap_mutation_immutability(self, binary_context):
        """Ensures SwapMutation only changes positions, not values."""
        pop, config, key = binary_context
        mut = SwapMutation(mutation_rate=1.0).set_input_length(len(pop))

        n_keys = mut.num_keys((len(pop),))
        all_keys = jr.split(key, n_keys)

        offspring_pop = mut(all_keys, pop, config)

        # The sum of bits (Hamming Weight) must remain identical after a swap
        original_counts = jnp.sum(pop.genes.values, axis=-1)
        mutated_counts = jnp.sum(offspring_pop.genes.values, axis=-1)
        assert jnp.all(original_counts == mutated_counts)

    def test_scramble_mutation_shuffling(self, binary_context):
        """Verifies that ScrambleMutation reorders bits."""
        pop, config, key = binary_context
        # High rate to ensure the lax.cond branch is taken
        mut = ScrambleMutation(mutation_rate=1.0).set_input_length(len(pop))

        n_keys = mut.num_keys((len(pop),))
        all_keys = jr.split(key, n_keys)

        offspring_pop = mut(all_keys, pop, config)

        # Values should be different due to permutation
        # (Though technically a scramble could result in the same bits,
        # it's highly unlikely for length 10)
        assert not jnp.all(offspring_pop.genes.values == pop.genes.values)

        # Like Swap, the total number of set bits must be preserved
        original_counts = jnp.sum(pop.genes.values, axis=-1)
        mutated_counts = jnp.sum(offspring_pop.genes.values, axis=-1)
        assert jnp.all(original_counts == mutated_counts)

    def test_jit_reproducibility(self, binary_context):
        """Ensures that the 3-Tier call is deterministic under JIT."""
        pop, config, key = binary_context
        mut = BitFlipMutation(mutation_rate=0.5).set_input_length(len(pop))

        n_keys = mut.num_keys((len(pop),))
        all_keys = jr.split(key, n_keys)

        @jax.jit
        def run_mut(k, p, c):
            return mut(k, p, c)

        # Execute JIT vs Raw
        res_jit = run_mut(all_keys, pop, config)
        res_raw = mut(all_keys, pop, config)

        # Binary data is exact (integers), so we check for absolute equality
        assert jnp.all(res_jit.genes.values == res_raw.genes.values)
