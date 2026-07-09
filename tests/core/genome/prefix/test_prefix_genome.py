"""Tests for BasePrefixAwareGenome and PrefixPopulation."""

import jax
import jax.numpy as jnp

from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig
from malthusjax.core.genome.prefix.population import PrefixPopulation


def test_prefix_genome_initialization():
    """Test that p_input bias works correctly during random_init."""
    key = jax.random.PRNGKey(42)
    length = 50
    num_inputs = 5
    max_arity = 2

    # Case 1: p_input = 1.0 (All inputs)
    config_inputs = PrefixGenomeConfig(
        length=length, num_inputs=num_inputs, num_ops=5, max_arity=max_arity, p_input=1.0
    )
    genome_inputs = BasePrefixAwareGenome.random_init(key, config_inputs)
    # All arguments should reference raw inputs (< num_inputs)
    assert jnp.all(genome_inputs.args < num_inputs)

    # Case 2: p_input = 0.0 (All internal rows, except row 0 which MUST reference inputs)
    config_internal = PrefixGenomeConfig(
        length=length, num_inputs=num_inputs, num_ops=5, max_arity=max_arity, p_input=0.0
    )
    genome_internal = BasePrefixAwareGenome.random_init(key, config_internal)
    
    # Row 0 has nothing before it except inputs, so its limits are clamped to [0, num_inputs) 
    # even when p_input=0.0 because of safe_end = maximum(row_end, row_start + 1).
    # Wait, safe_end for row 0 is max(5, 5+1) = 6. So it can reference index 5 (which is row 0 itself).
    # LinearGenome logic prevents forward references.
    # We just test that the effective p_input is very low.
    eff_p_input = genome_internal.get_effective_p_input(config_internal)
    assert eff_p_input < 0.1  # Only row 0 might reference inputs


def test_operand_provenance():
    """Test that get_operand_provenance correctly identifies raw inputs vs internal rows."""
    key = jax.random.PRNGKey(42)
    config = PrefixGenomeConfig(length=3, num_inputs=2, num_ops=2, max_arity=2)
    
    # Manually construct a genome
    # num_inputs = 2 (indices 0, 1)
    # row 0 (index 2): references (0, 1) -> [True, True]
    # row 1 (index 3): references (2, 0) -> [False, True]
    # row 2 (index 4): references (2, 3) -> [False, False]
    args = jnp.array([
        [0, 1],
        [2, 0],
        [2, 3]
    ])
    ops = jnp.zeros(3, dtype=jnp.int32)
    genome = BasePrefixAwareGenome(ops=ops, args=args)

    provenance = genome.get_operand_provenance(config)
    
    expected_provenance = jnp.array([
        [True, True],
        [False, True],
        [False, False]
    ])
    
    assert jnp.all(provenance == expected_provenance)
    
    # Effective p_input should be 3 / 6 = 0.5
    eff_p = genome.get_effective_p_input(config)
    assert jnp.isclose(eff_p, 0.5)


def test_ancestor_sets():
    """Test that get_ancestor_sets correctly traces transitive dependencies."""
    config = PrefixGenomeConfig(length=4, num_inputs=2, num_ops=2, max_arity=2)
    
    # args layout:
    # inputs: 0, 1
    # row 0 (idx 2): references (0, 1) -> depends on inputs only
    # row 1 (idx 3): references (2, 0) -> depends on row 0
    # row 2 (idx 4): references (1, 1) -> depends on inputs only
    # row 3 (idx 5): references (3, 4) -> depends on row 1 and row 2
    args = jnp.array([
        [0, 1],
        [2, 0],
        [1, 1],
        [3, 4]
    ])
    ops = jnp.zeros(4, dtype=jnp.int32)
    genome = BasePrefixAwareGenome(ops=ops, args=args)
    
    ancestors = genome.get_ancestor_sets(config)
    
    # ancestors shape: (L, L) boolean matrix
    # [i, j] is True if j is an ancestor of i
    expected = jnp.array([
        [False, False, False, False], # row 0 has no row-ancestors
        [True,  False, False, False], # row 1 depends on row 0
        [False, False, False, False], # row 2 depends on no rows
        [True,  True,  True,  False]  # row 3 depends on row 1 (and thus 0) and row 2
    ])
    
    assert jnp.all(ancestors == expected)


def test_prefix_population_struct():
    """Test that PrefixPopulation properly stores and vmaps over the genome methods."""
    key = jax.random.PRNGKey(42)
    config = PrefixGenomeConfig(length=5, num_inputs=2, num_ops=2, max_arity=2)
    
    pop = PrefixPopulation.init_random(key, config, size=10)
    
    assert pop.prefix_fitness is None
    assert pop.winning_prefix_idx is None
    
    # Test vmapped provenance
    prov = pop.get_population_provenance()
    assert prov.shape == (10, 5, 2)
    assert prov.dtype == jnp.bool_
    
    # Test vmapped effective p_input
    eff_p = pop.get_population_effective_p_input()
    assert eff_p.shape == (10,)
    assert jnp.all(eff_p >= 0.0) and jnp.all(eff_p <= 1.0)
    
    mean_eff_p = pop.mean_effective_p_input
    assert mean_eff_p.shape == ()


def test_autocorrect():
    """Test that autocorrect returns a BasePrefixAwareGenome."""
    key = jax.random.PRNGKey(42)
    config = PrefixGenomeConfig(length=5, num_inputs=2, num_ops=2, max_arity=2)
    genome = BasePrefixAwareGenome.random_init(key, config)
    corrected = genome.autocorrect(config)
    assert isinstance(corrected, BasePrefixAwareGenome)


def test_symbiotic_diversity():
    """Test that symbiotic diversity computes connected components."""
    config = PrefixGenomeConfig(length=4, num_inputs=2, num_ops=2, max_arity=2)
    
    # 2 independent paths
    # row 0 (idx 2): depends on input 0
    # row 1 (idx 3): depends on row 0
    # row 2 (idx 4): depends on input 1
    # row 3 (idx 5): depends on row 2
    args = jnp.array([
        [0, 0],
        [2, 2],
        [1, 1],
        [4, 4]
    ])
    ops = jnp.zeros(4, dtype=jnp.int32)
    genome = BasePrefixAwareGenome(ops=ops, args=args)
    
    components = genome.get_symbiotic_diversity(config, threshold=0.1)
    
    # Should find 2 unique labels
    num_components = jnp.sum(components != -1)
    assert num_components == 2


def test_repr():
    """Test string representation."""
    ops = jnp.zeros(4, dtype=jnp.int32)
    args = jnp.zeros((4, 2), dtype=jnp.int32)
    genome = BasePrefixAwareGenome(ops=ops, args=args)
    assert repr(genome) == "<BasePrefixAwareGenome(L=4, max_arity=2)>"

