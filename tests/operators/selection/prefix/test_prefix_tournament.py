"""Tests for PrefixTournamentSelection and rank deflation."""

import jax
import jax.numpy as jnp

from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig
from malthusjax.core.genome.prefix.population import PrefixPopulation
from malthusjax.operators.selection.prefix.tournament import (
    PrefixTournamentConfig,
    PrefixTournamentSelection,
)


def _make_dummy_population() -> PrefixPopulation:
    pop_size = 4
    L = 5
    prefix_fitness = jnp.array([
        [0.1, 0.11, 0.12, 0.13, 0.14],  # Genome 0 (Best overall)
        [0.5, 0.51, 0.52, 0.53, 0.54],  # Genome 1
        [10.0, 10.1, 10.2, 10.3, 10.4], # Genome 2
        [20.0, 20.1, 20.2, 20.3, 20.4], # Genome 3
    ])
    dummy_genome = BasePrefixAwareGenome(ops=jnp.zeros((pop_size, L)), args=jnp.zeros((pop_size, L, 2)))
    return PrefixPopulation(
        genes=dummy_genome, 
        fitness=jnp.zeros(pop_size), 
        prefix_fitness=prefix_fitness, 
        winning_prefix_idx=jnp.zeros(pop_size), 
        config=PrefixGenomeConfig(length=L, num_inputs=2, num_ops=2, max_arity=2)
    )


def test_prefix_tournament_selection_shape():
    """Test that PrefixTournamentSelection returns expected shapes."""
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    
    num_selections = 10
    config = PrefixTournamentConfig(alpha=1.0, maximize=False)
    selector = PrefixTournamentSelection(num_selections=num_selections, tournament_size=3)
    
    parents, elites = selector(key, pop, config=config)
    
    # Check parent shape: (num_selections,) for 1D return
    assert parents.shape == (num_selections,)
    
    # Indices should be valid genome indices
    assert jnp.all(parents >= 0) and jnp.all(parents < 4)


def test_rank_deflation_logic():
    """Test that rank deflation properly penalizes lower-ranked prefixes."""
    pop = _make_dummy_population()
    config = PrefixTournamentConfig(alpha=0.5, maximize=False)
    selector = PrefixTournamentSelection(num_selections=1, tournament_size=3)
    
    # 1. Test Minimization (alpha=0.5 means lower-ranked items get divided by 0.5^rank -> increase)
    deflated_min = selector._apply_rank_deflation(pop.prefix_fitness, alpha=0.5, maximize=False)
    
    # Genome 0 original: [0.1, 0.11, 0.12, 0.13, 0.14]
    # Deflated: [0.1, 0.22, 0.48, 1.04, 2.24] (approximately)
    assert jnp.isclose(deflated_min[0, 0], 0.1)
    assert deflated_min[0, 1] > 0.2  # Rank 1 penalty
    assert deflated_min[0, 2] > 0.4  # Rank 2 penalty
    
    # 2. Test Maximization (alpha=0.5 means lower-ranked items get multiplied by 0.5^rank -> decrease)
    # Re-create fitness matrix where higher is better
    max_fitness = jnp.array([
        [10.0, 8.0, 6.0],
        [20.0, 18.0, 16.0]
    ])
    deflated_max = selector._apply_rank_deflation(max_fitness, alpha=0.5, maximize=True)
    
    # Genome 1: Rank 0 = 20.0, Rank 1 = 18.0, Rank 2 = 16.0
    assert jnp.isclose(deflated_max[1, 0], 20.0)
    assert jnp.isclose(deflated_max[1, 1], 9.0)  # 18.0 * 0.5^1
    assert jnp.isclose(deflated_max[1, 2], 4.0)  # 16.0 * 0.5^2


def test_strict_deflation_alpha_zero():
    """Test that alpha=0.0 strongly penalizes sub-optimal prefixes but keeps rank order."""
    pop = _make_dummy_population()
    config = PrefixTournamentConfig(alpha=0.0, maximize=False)
    selector = PrefixTournamentSelection(num_selections=10, tournament_size=3)
    
    # Deflate using safe bound (1e-6)
    deflated = selector._apply_rank_deflation(pop.prefix_fitness, alpha=0.0, maximize=False)
    
    # Genome 0 rank 0 should be original
    assert jnp.isclose(deflated[0, 0], 0.1)
    
    # Rank 1 should be enormously penalized (e.g. divided by 1e-6)
    assert deflated[0, 1] > 1000.0


def test_elitism():
    """Test that elitism extracts the absolute best (parent, prefix) pairs."""
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    config = PrefixTournamentConfig(alpha=1.0, maximize=False)
    
    selector = PrefixTournamentSelection(num_selections=10, tournament_size=3, n_elites=2)
    _, elites = selector(key, pop, config=config)
    
    assert elites.shape == (2,)
    
    # The two absolute best in the whole matrix are Genome 0, Prefix 0 (0.1) and Prefix 1 (0.11)
    # Order doesn't matter, but they should both be Genome 0
    assert jnp.all(elites == 0)


def test_base_selection_methods_and_properties():
    """Test standard methods inherited from BasePrefixSelection."""
    sel = PrefixTournamentSelection(num_selections=10)
    
    # Test setters
    sel2 = sel.set_input_length(100)
    assert sel2.input_length == 100
    
    sel3 = sel.set_typed_keys(True)
    assert sel3.typed_keys is True
    
    sel4 = sel.set_n_elites(5)
    assert sel4.n_elites == 5
    
    # Test properties
    assert sel.num_keys_per_atomic_operation == 1
    assert sel.num_keys((100,)) == 1


def test_base_selection_exceptions():
    """Test that __call__ raises error if prefix_fitness is missing."""
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    # Remove prefix_fitness to trigger exception
    bad_pop = PrefixPopulation(
        genes=pop.genes,
        fitness=pop.fitness,
        prefix_fitness=None,  # Missing
        winning_prefix_idx=None,
        config=pop.config
    )
    
    sel = PrefixTournamentSelection(num_selections=10)
    import pytest
    with pytest.raises(ValueError, match="missing prefix_fitness"):
        sel(key, bad_pop)


def test_get_elite_indices_edge_cases():
    """Test get_elite_indices when n_elites >= total_candidates and when maximize=True."""
    pop = _make_dummy_population()
    sel = PrefixTournamentSelection(num_selections=10, n_elites=100) # >= 20 total candidates
    
    elites_all = sel.get_elite_indices(pop.prefix_fitness)
    assert elites_all.shape == (20, 2)
    
    # Test maximize=True path
    sel_max = PrefixTournamentSelection(num_selections=10, n_elites=2)
    elites_max = sel_max.get_elite_indices(pop.prefix_fitness, maximize=True)
    assert elites_max.shape == (2, 2)
    # The highest values are in Genome 3 (20.0 to 20.4). So they should be selected.
    assert jnp.all(elites_max[:, 0] == 3)


def test_tournament_maximize():
    """Test PrefixTournamentSelection with maximize=True."""
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    
    config = PrefixTournamentConfig(alpha=1.0, maximize=True)
    sel = PrefixTournamentSelection(num_selections=10)
    
    parents, elites = sel(key, pop, config=config)
    
    # Just verify shapes to ensure the maximize=True branch doesn't crash
    assert parents.shape == (10,)


def test_typed_keys_branch():
    """Test that typed_keys=True extracts the key correctly without crashing."""
    import unittest.mock as mock
    
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    
    config = PrefixTournamentConfig(alpha=1.0, maximize=False)
    sel = PrefixTournamentSelection(num_selections=10, typed_keys=True)
    
    # We mock randint so JAX doesn't complain about the legacy key ndim in typed mode
    with mock.patch("jax.random.randint", return_value=jnp.zeros((10, 3), dtype=jnp.int32)):
        parents, _ = sel(key, pop, config=config)
        
    assert parents.shape == (10,)

def test_select_with_provenance():
    """Test that select_with_provenance returns full 2D arrays."""
    key = jax.random.PRNGKey(42)
    pop = _make_dummy_population()
    
    config = PrefixTournamentConfig(alpha=1.0, maximize=False)
    selector = PrefixTournamentSelection(num_selections=10, tournament_size=3)
    
    parent_pairs, elite_pairs = selector.select_with_provenance(key, pop, config=config)
    
    # Should be 2D
    assert parent_pairs.shape == (10, 2)
    assert elite_pairs.shape == (0, 2)
    
    # Ensure they map correctly to the genome indices returned by __call__
    parent_idx, elite_idx = selector(key, pop, config=config)
    assert jnp.array_equal(parent_pairs[:, 0], parent_idx)

