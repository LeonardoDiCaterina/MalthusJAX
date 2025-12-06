"""
Tests for the Resource Mapper (Step 3 of Optimization Roadmap).

Tests RNG budget calculation, key map generation, and integration with engine.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.engine.resource_mapper import (
    compute_operator_budget,
    compute_resource_map,
    get_resource_summary,
    OperatorRNGBudget,
    ResourceMap,
)
from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.genome.categorical_genome import CategoricalGenomeConfig


# ==========================================
# Test Fixtures: Mock Operators
# ==========================================

@struct.dataclass
class MockMutationCustomKeys(BaseMutation):
    """Mock mutation with custom num_keys."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _mutate_one(self, key, genome, config):
        return genome
    
    def num_keys(self, config, input_shape):
        # Custom: needs 3 keys per offspring
        return self.num_offspring * 3


@struct.dataclass
class MockCrossoverCustomKeys(BaseCrossover):
    """Mock crossover with custom num_keys."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _crossover_pair(self, key, parent1, parent2, config):
        return parent1
    
    def num_keys(self, config, input_shape):
        # Custom: needs 5 keys total
        return 5


@struct.dataclass
class MockSelectionCustomKeys(BaseSelection):
    """Mock selection with custom num_keys."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def _select_batch(self, key, fitness_values):
        return jnp.arange(self.num_selections)
    
    def num_keys(self, input_shape):
        # Custom: needs 2 keys for selection
        return 2


@struct.dataclass
class MockMutationDefault(BaseMutation):
    """Mock mutation using default num_keys."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    def _mutate_one(self, key, genome, config):
        return genome


@struct.dataclass
class MockCrossoverDefault(BaseCrossover):
    """Mock crossover using default num_keys."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _crossover_pair(self, key, parent1, parent2, config):
        return parent1


@struct.dataclass
class MockSelectionDefault(BaseSelection):
    """Mock selection using default num_keys."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    def _select_batch(self, key, fitness_values):
        return jnp.arange(self.num_selections)


# ==========================================
# Test: compute_operator_budget()
# ==========================================

def test_compute_operator_budget_mutation():
    """Test budget computation for mutation operator."""
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=10)
    
    budget = compute_operator_budget(
        mutation, 'mutation', genome_config, (10,), start_idx=0
    )
    
    assert isinstance(budget, OperatorRNGBudget)
    assert budget.num_keys == 6  # 2 offspring * 3 keys each
    assert budget.start_idx == 0
    assert budget.end_idx == 6
    assert budget.operator_type == 'mutation'


def test_compute_operator_budget_crossover():
    """Test budget computation for crossover operator."""
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    genome_config = RealGenomeConfig(length=5, bounds=(-1.0, 1.0))
    
    budget = compute_operator_budget(
        crossover, 'crossover', genome_config, (5,), start_idx=10
    )
    
    assert budget.num_keys == 5
    assert budget.start_idx == 10
    assert budget.end_idx == 15
    assert budget.operator_type == 'crossover'


def test_compute_operator_budget_selection():
    """Test budget computation for selection operator."""
    selection = MockSelectionCustomKeys(num_selections=20)
    genome_config = BinaryGenomeConfig(length=100)
    
    budget = compute_operator_budget(
        selection, 'selection', genome_config, (100,), start_idx=5
    )
    
    assert budget.num_keys == 2
    assert budget.start_idx == 5
    assert budget.end_idx == 7
    assert budget.operator_type == 'selection'


def test_compute_operator_budget_default_implementations():
    """Test budget computation with default num_keys implementations."""
    mutation = MockMutationDefault(num_offspring=3)
    genome_config = BinaryGenomeConfig(length=50)
    
    budget = compute_operator_budget(
        mutation, 'mutation', genome_config, (50,), start_idx=0
    )
    
    # Default implementation returns num_offspring
    assert budget.num_keys == 3


# ==========================================
# Test: compute_resource_map()
# ==========================================

def test_compute_resource_map_custom_operators():
    """Test complete resource map with custom operators."""
    selection = MockSelectionCustomKeys(num_selections=100)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=20)
    pop_size = 100
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    assert isinstance(resource_map, ResourceMap)
    assert resource_map.pop_size == 100
    assert resource_map.genome_shape == (20,)
    
    # Check individual budgets
    assert resource_map.selection_budget.num_keys == 2
    assert resource_map.selection_budget.start_idx == 0
    assert resource_map.selection_budget.end_idx == 2
    
    assert resource_map.crossover_budget.num_keys == 5
    assert resource_map.crossover_budget.start_idx == 2
    assert resource_map.crossover_budget.end_idx == 7
    
    assert resource_map.mutation_budget.num_keys == 6
    assert resource_map.mutation_budget.start_idx == 7
    assert resource_map.mutation_budget.end_idx == 13
    
    # Check total
    assert resource_map.total_rng_budget == 13


def test_compute_resource_map_default_operators():
    """Test resource map with default operator implementations."""
    selection = MockSelectionDefault(num_selections=50)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(num_offspring=1)
    genome_config = RealGenomeConfig(length=10, bounds=(-5.0, 5.0))
    pop_size = 50
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    # Default implementations: num_keys = num_offspring (or 1 for selection)
    assert resource_map.selection_budget.num_keys == 1
    assert resource_map.crossover_budget.num_keys == 2
    assert resource_map.mutation_budget.num_keys == 1
    assert resource_map.total_rng_budget == 4


def test_compute_resource_map_different_genome_types():
    """Test resource map with different genome configurations."""
    selection = MockSelectionDefault(num_selections=10)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(num_offspring=1)
    
    # Binary genome
    binary_config = BinaryGenomeConfig(length=50)
    rmap_binary = compute_resource_map(
        selection, crossover, mutation, binary_config, 100
    )
    assert rmap_binary.genome_shape == (50,)
    
    # Real genome
    real_config = RealGenomeConfig(length=20, bounds=(-10.0, 10.0))
    rmap_real = compute_resource_map(
        selection, crossover, mutation, real_config, 100
    )
    assert rmap_real.genome_shape == (20,)
    
    # Categorical genome
    cat_config = CategoricalGenomeConfig(length=5, num_categories=15)
    rmap_cat = compute_resource_map(
        selection, crossover, mutation, cat_config, 100
    )
    assert rmap_cat.genome_shape == (5,)  # length is the shape, num_categories is choices per position


def test_compute_resource_map_large_population():
    """Test resource map with large population size."""
    selection = MockSelectionCustomKeys(num_selections=1000)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.05, num_offspring=5)
    genome_config = RealGenomeConfig(length=100, bounds=(-1.0, 1.0))
    pop_size = 1000
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    assert resource_map.pop_size == 1000
    assert resource_map.genome_shape == (100,)
    assert resource_map.mutation_budget.num_keys == 15  # 5 offspring * 3 keys


# ==========================================
# Test: ResourceMap.get_key_slice()
# ==========================================

def test_resource_map_get_key_slice():
    """Test key slice retrieval from resource map."""
    selection = MockSelectionCustomKeys(num_selections=10)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    # Get slices
    sel_slice = resource_map.get_key_slice('selection')
    cross_slice = resource_map.get_key_slice('crossover')
    mut_slice = resource_map.get_key_slice('mutation')
    
    # Verify slices
    assert sel_slice == slice(0, 2)
    assert cross_slice == slice(2, 7)
    assert mut_slice == slice(7, 13)


def test_resource_map_get_key_slice_invalid():
    """Test that invalid operator type raises error."""
    selection = MockSelectionDefault(num_selections=10)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(num_offspring=1)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    with pytest.raises(ValueError, match="Unknown operator type"):
        resource_map.get_key_slice('invalid_type')


def test_resource_map_key_slicing_workflow():
    """Test complete workflow: compute map, split keys, slice for operators."""
    selection = MockSelectionCustomKeys(num_selections=10)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    # Simulate key allocation
    main_key = jar.PRNGKey(42)
    all_keys = jar.split(main_key, resource_map.total_rng_budget)
    
    # Verify we can slice correctly
    assert all_keys.shape == (resource_map.total_rng_budget, 2)
    
    selection_keys = all_keys[resource_map.get_key_slice('selection')]
    crossover_keys = all_keys[resource_map.get_key_slice('crossover')]
    mutation_keys = all_keys[resource_map.get_key_slice('mutation')]
    
    assert selection_keys.shape == (2, 2)
    assert crossover_keys.shape == (5, 2)
    assert mutation_keys.shape == (6, 2)


# ==========================================
# Test: get_resource_summary()
# ==========================================

def test_get_resource_summary():
    """Test summary string generation."""
    selection = MockSelectionCustomKeys(num_selections=50)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=3)
    genome_config = RealGenomeConfig(length=25, bounds=(-1.0, 1.0))
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 50
    )
    
    summary = get_resource_summary(resource_map)
    
    # Check that summary contains expected information
    assert "Resource Allocation Summary" in summary
    assert "Total RNG Budget:" in summary
    assert "Population Size: 50" in summary
    assert "Genome Shape: (25,)" in summary
    assert "Selection:" in summary
    assert "Crossover:" in summary
    assert "Mutation:" in summary
    assert "[0:2]" in summary  # Selection slice
    assert "[2:7]" in summary  # Crossover slice
    assert "[7:16]" in summary  # Mutation slice (3 offspring * 3 keys = 9)


# ==========================================
# Test: Integration with GeneticEngine
# ==========================================

def test_genetic_engine_resource_map_property():
    """Test that GeneticEngine exposes resource_map property."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=20)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionCustomKeys(num_selections=100),
        crossover=MockCrossoverCustomKeys(num_offspring=2),
        mutation=MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    )
    
    # Check property exists
    assert hasattr(engine, "resource_map")
    
    # Get resource map
    rmap = engine.resource_map
    
    assert isinstance(rmap, ResourceMap)
    assert rmap.genome_shape == (20,)
    assert rmap.total_rng_budget > 0


def test_genetic_engine_resource_map_consistency():
    """Test that resource map is consistent across multiple accesses."""
    from malthusjax.engine.genetic_engine import GeneticEngine
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = RealGenomeConfig(length=15, bounds=(-5.0, 5.0))
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    
    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionDefault(num_selections=50),
        crossover=MockCrossoverDefault(num_offspring=2),
        mutation=MockMutationDefault(num_offspring=1)
    )
    
    # Multiple accesses should give consistent results
    rmap1 = engine.resource_map
    rmap2 = engine.resource_map
    
    assert rmap1.total_rng_budget == rmap2.total_rng_budget
    assert rmap1.selection_budget.num_keys == rmap2.selection_budget.num_keys
    assert rmap1.crossover_budget.num_keys == rmap2.crossover_budget.num_keys
    assert rmap1.mutation_budget.num_keys == rmap2.mutation_budget.num_keys


# ==========================================
# Test: Acceptance Criteria from Roadmap
# ==========================================

def test_acceptance_total_budget_matches_manual_count():
    """Acceptance: Generated TOTAL_RNG_BUDGET matches manual counting."""
    # Manual setup
    sel_keys = 2
    cross_keys = 5
    mut_keys = 6
    expected_total = sel_keys + cross_keys + mut_keys
    
    # Compute via resource mapper
    selection = MockSelectionCustomKeys(num_selections=10)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    # Verify total matches manual count
    assert resource_map.total_rng_budget == expected_total


def test_acceptance_key_slices_non_overlapping():
    """Acceptance: Key slices don't overlap and cover full range."""
    selection = MockSelectionCustomKeys(num_selections=10)
    crossover = MockCrossoverCustomKeys(num_offspring=2)
    mutation = MockMutationCustomKeys(mutation_rate=0.1, num_offspring=2)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    sel_budget = resource_map.selection_budget
    cross_budget = resource_map.crossover_budget
    mut_budget = resource_map.mutation_budget
    
    # No overlaps
    assert sel_budget.end_idx == cross_budget.start_idx
    assert cross_budget.end_idx == mut_budget.start_idx
    
    # Covers full range
    assert sel_budget.start_idx == 0
    assert mut_budget.end_idx == resource_map.total_rng_budget


def test_acceptance_resource_map_enables_static_allocation():
    """Acceptance: Resource map enables pre-allocation without runtime splits."""
    selection = MockSelectionDefault(num_selections=20)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(num_offspring=1)
    genome_config = BinaryGenomeConfig(length=30)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 50
    )
    
    # Pre-allocate all keys at once (no runtime splitting)
    main_key = jar.PRNGKey(123)
    all_keys = jar.split(main_key, resource_map.total_rng_budget)
    
    # Verify we can access operator-specific keys via slicing
    sel_keys = all_keys[resource_map.get_key_slice('selection')]
    cross_keys = all_keys[resource_map.get_key_slice('crossover')]
    mut_keys = all_keys[resource_map.get_key_slice('mutation')]
    
    # All keys allocated statically
    assert sel_keys.shape[0] == resource_map.selection_budget.num_keys
    assert cross_keys.shape[0] == resource_map.crossover_budget.num_keys
    assert mut_keys.shape[0] == resource_map.mutation_budget.num_keys
