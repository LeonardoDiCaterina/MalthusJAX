import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex

from malthusjax.operators.mutation.permutation import ScrambleMutation, SwapMutation

# --- Mock Genome for Testing ---
# The current permutation.py implementation expects an attribute named '.genome'
# This mock satisfies that interface so we can test the logic.
@struct.dataclass
class MockPermutationGenome:
    genome: chex.Array  # Shape (L,)

    @property
    def shape(self):
        # Helper for tests, though the operator accesses .genome.shape
        return self.genome.shape

# --- Fixtures ---

@pytest.fixture
def perm_genome():
    """Creates a genome [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"""
    return MockPermutationGenome(genome=jnp.arange(10))

@pytest.fixture
def key():
    return jar.PRNGKey(42)

# --- Tests for Scramble Mutation ---

def test_scramble_mutation_active(perm_genome, key):
    """Test that scrambling changes the order (rate=1.0)."""
    mutation = ScrambleMutation(mutation_rate=1.0)
    
    # Run mutation
    mutated = mutation._mutate_one(key, perm_genome, None)
    
    # 1. Check it's different
    # (Statistically possible to be same, but 1/10! chance is negligible)
    assert not jnp.array_equal(mutated.genome, perm_genome.genome)
    
    # 2. Check conservation of elements (it's a permutation)
    # Sorting both should result in identical arrays
    assert jnp.array_equal(jnp.sort(mutated.genome), jnp.sort(perm_genome.genome))

def test_scramble_mutation_inactive(perm_genome, key):
    """Test that scrambling does nothing (rate=0.0)."""
    mutation = ScrambleMutation(mutation_rate=0.0)
    
    mutated = mutation._mutate_one(key, perm_genome, None)
    
    # Must be identical
    assert jnp.array_equal(mutated.genome, perm_genome.genome)

# --- Tests for Swap Mutation ---

def test_swap_mutation_active(perm_genome, key):
    """Test that swap exchanges exactly two elements (rate=1.0)."""
    mutation = SwapMutation(mutation_rate=1.0)
    
    # We need a key that ensures pos1 != pos2 for the test to see a change
    # But even if pos1 == pos2, the logic holds (0 changes).
    # We loop a few times to ensure we catch a real swap.
    keys = jar.split(key, 5)
    
    changed = False
    for k in keys:
        mutated = mutation._mutate_one(k, perm_genome, None)
        
        diffs = mutated.genome != perm_genome.genome
        num_diffs = jnp.sum(diffs)
        
        if num_diffs > 0:
            changed = True
            # If swap occurs between different indices, exactly 2 positions change
            assert num_diffs == 2
            # Check conservation (still a permutation)
            assert jnp.array_equal(jnp.sort(mutated.genome), jnp.sort(perm_genome.genome))
            break
            
    assert changed, "Swap failed to change genome in 5 attempts (unlikely)"

def test_swap_mutation_inactive(perm_genome, key):
    """Test that swap does nothing (rate=0.0)."""
    mutation = SwapMutation(mutation_rate=0.0)
    mutated = mutation._mutate_one(key, perm_genome, None)
    assert jnp.array_equal(mutated.genome, perm_genome.genome)

def test_jit_compatibility(perm_genome, key):
    """Ensure operators can be JIT compiled."""
    mutation = SwapMutation(mutation_rate=1.0)
    
    # Wrap in JIT
    @jax.jit
    def run_mut(k, g):
        return mutation._mutate_one(k, g, None)
        
    # Should run without error
    out = run_mut(key, perm_genome)
    assert out.genome.shape == perm_genome.genome.shape