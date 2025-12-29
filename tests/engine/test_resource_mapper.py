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
    compute_resource_map,
    get_resource_summary,
    OperatorAllocation,
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
    """Mock mutation with custom num_keys_per_atomic_operation."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        # Custom: needs 3 keys per offspring
        return 3
    
    def _mutate_one(self, key, genome, config):
        return genome


@struct.dataclass
class MockCrossoverCustomKeys(BaseCrossover):
    """Mock crossover with custom num_keys_per_atomic_operation."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        # Custom: needs 2 keys per crossover operation
        return 2
    
    def _cross_one(self, key_block, p1, p2, config):
        return p1


@struct.dataclass
class MockSelectionCustomKeys(BaseSelection):
    """Mock selection with custom keys_per_selection."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    @property
    def keys_per_selection(self) -> int:
        # Custom: needs 2 keys per selection
        return 2
    
    def _select(self, keys, fitness, config):
        return jnp.arange(self.num_selections)


@struct.dataclass
class MockMutationDefault(BaseMutation):
    """Mock mutation using default num_keys (1 key per atomic op)."""
    mutation_rate: float = 0.1
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
    
    def _mutate_one(self, key, genome, config):
        return genome


@struct.dataclass
class MockCrossoverDefault(BaseCrossover):
    """Mock crossover using default num_keys."""
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    # Uses default num_keys_per_atomic_operation = 1 from BaseCrossover
    
    def _cross_one(self, key_block, p1, p2, config):
        return p1


@struct.dataclass
class MockSelectionDefault(BaseSelection):
    """Mock selection using default num_keys."""
    num_selections: int = struct.field(pytree_node=False, default=10)
    
    # Uses default keys_per_selection = 1 from BaseSelection
    
    def _select(self, keys, fitness, config):
        return jnp.arange(self.num_selections)


# ==========================================
# Test: compute_resource_map()
# ==========================================

def test_compute_resource_map_basic():
    """Test basic resource map computation."""
    selection = MockSelectionDefault(num_selections=100)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    genome_config = BinaryGenomeConfig(length=20)
    pop_size = 100
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    assert isinstance(resource_map, ResourceMap)
    assert resource_map.pop_size == 100
    assert resource_map.genome_shape == (20,)
    
    # Check that allocations exist
    assert isinstance(resource_map.selection, OperatorAllocation)
    assert isinstance(resource_map.crossover, OperatorAllocation)
    assert isinstance(resource_map.mutation, OperatorAllocation)
    assert isinstance(resource_map.next_key, OperatorAllocation)
    
    # Total budget includes all operators + next_key
    assert resource_map.total_rng_budget > 0


def test_compute_resource_map_cascade_flow():
    """Test that the cascade data flow is computed correctly."""
    pop_size = 17  # Odd number to test ceiling division
    offspring_per_pair = 2
    
    selection = MockSelectionDefault(num_selections=100)
    crossover = MockCrossoverDefault(num_offspring=offspring_per_pair)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    # Verify cascade calculations:
    # pairs_needed = ceil(17 / 2) = 9
    # parents_needed = 9 * 2 = 18
    pairs_needed = (pop_size + offspring_per_pair - 1) // offspring_per_pair
    parents_needed = pairs_needed * 2
    
    assert resource_map.selection.output_count == parents_needed
    assert resource_map.crossover.input_count == parents_needed
    assert resource_map.crossover.output_count == pairs_needed * offspring_per_pair


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
    assert rmap_cat.genome_shape == (5,)


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
    # Mutation keys: input_count * num_offspring * keys_per_atomic_op
    assert resource_map.mutation.num_keys > 0


# ==========================================
# Test: ResourceMap.get_key_slice()
# ==========================================

def test_resource_map_get_key_slice():
    """Test key slice retrieval from resource map."""
    selection = MockSelectionDefault(num_selections=10)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    # Get slices
    sel_slice = resource_map.get_key_slice('selection')
    cross_slice = resource_map.get_key_slice('crossover')
    mut_slice = resource_map.get_key_slice('mutation')
    next_slice = resource_map.get_key_slice('next_key')
    
    # Verify slices are consistent with allocations
    assert sel_slice == slice(resource_map.selection.start_idx, resource_map.selection.end_idx)
    assert cross_slice == slice(resource_map.crossover.start_idx, resource_map.crossover.end_idx)
    assert mut_slice == slice(resource_map.mutation.start_idx, resource_map.mutation.end_idx)
    assert next_slice == slice(resource_map.next_key.start_idx, resource_map.next_key.end_idx)


def test_resource_map_key_slicing_workflow():
    """Test complete workflow: compute map, split keys, slice for operators."""
    selection = MockSelectionDefault(num_selections=10)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
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
    next_key = all_keys[resource_map.get_key_slice('next_key')]
    
    assert selection_keys.shape == (resource_map.selection.num_keys, 2)
    assert crossover_keys.shape == (resource_map.crossover.num_keys, 2)
    assert mutation_keys.shape == (resource_map.mutation.num_keys, 2)
    assert next_key.shape == (1, 2)


# ==========================================
# Test: get_resource_summary()
# ==========================================

def test_get_resource_summary():
    """Test summary string generation."""
    selection = MockSelectionDefault(num_selections=50)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    genome_config = RealGenomeConfig(length=25, bounds=(-1.0, 1.0))
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 50
    )
    
    summary = get_resource_summary(resource_map)
    
    # Check that summary contains expected information
    assert "Total RNG Budget:" in summary
    assert "SELECTION" in summary
    assert "CROSSOVER" in summary
    assert "MUTATION" in summary
    assert "NEXT GENERATION KEY" in summary


# ==========================================
# Test: Integration with GeneticEngine
# ==========================================

def test_genetic_engine_resource_map_in_state():
    """Test that GeneticEngine init_state includes resource_map."""
    from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticEvolutionState
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=20)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    engine_params = GeneticEngineParams(pop_size=100)
    
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionDefault(num_selections=100),
        crossover=MockCrossoverDefault(num_offspring=2),
        mutation=MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    )
    
    # Initialize state
    state = engine.init_state(jar.PRNGKey(42))
    
    # Check state has resource_map
    assert isinstance(state, GeneticEvolutionState)
    assert hasattr(state, "resource_map")
    
    # Get resource map from state
    rmap = state.resource_map
    
    assert isinstance(rmap, ResourceMap)
    assert rmap.genome_shape == (20,)
    assert rmap.total_rng_budget > 0


def test_genetic_engine_resource_map_consistency():
    """Test that resource map is consistent across multiple state initializations with same engine."""
    from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
    from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
    
    genome_config = BinaryGenomeConfig(length=15)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    engine_params = GeneticEngineParams(pop_size=50)
    
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=MockSelectionDefault(num_selections=50),
        crossover=MockCrossoverDefault(num_offspring=2),
        mutation=MockMutationDefault(num_offspring=1)
    )
    
    # Initialize state twice with different keys
    state1 = engine.init_state(jar.PRNGKey(1))
    state2 = engine.init_state(jar.PRNGKey(2))
    
    # Resource maps should have same structure (different random populations though)
    rmap1 = state1.resource_map
    rmap2 = state2.resource_map
    
    assert rmap1.total_rng_budget == rmap2.total_rng_budget
    assert rmap1.selection.num_keys == rmap2.selection.num_keys
    assert rmap1.crossover.num_keys == rmap2.crossover.num_keys
    assert rmap1.mutation.num_keys == rmap2.mutation.num_keys


# ==========================================
# Test: Acceptance Criteria from Roadmap
# ==========================================

def test_acceptance_key_slices_non_overlapping():
    """Acceptance: Key slices don't overlap and cover full range."""
    selection = MockSelectionDefault(num_selections=10)
    crossover = MockCrossoverDefault(num_offspring=2)
    mutation = MockMutationDefault(mutation_rate=0.1, num_offspring=1)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, 100
    )
    
    sel = resource_map.selection
    cross = resource_map.crossover
    mut = resource_map.mutation
    next_k = resource_map.next_key
    
    # No overlaps - sequential allocation
    assert sel.end_idx == cross.start_idx
    assert cross.end_idx == mut.start_idx
    assert mut.end_idx == next_k.start_idx
    
    # Covers full range
    assert sel.start_idx == 0
    assert next_k.end_idx == resource_map.total_rng_budget


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
    assert sel_keys.shape[0] == resource_map.selection.num_keys
    assert cross_keys.shape[0] == resource_map.crossover.num_keys
    assert mut_keys.shape[0] == resource_map.mutation.num_keys


def test_acceptance_output_count_matches_cascade():
    """Acceptance: Output counts cascade correctly through the pipeline."""
    pop_size = 100
    crossover_offspring = 2
    mutation_offspring = 1
    
    selection = MockSelectionDefault(num_selections=100)
    crossover = MockCrossoverDefault(num_offspring=crossover_offspring)
    mutation = MockMutationDefault(num_offspring=mutation_offspring)
    genome_config = BinaryGenomeConfig(length=10)
    
    resource_map = compute_resource_map(
        selection, crossover, mutation, genome_config, pop_size
    )
    
    # Selection input is pop_size
    assert resource_map.selection.input_count == pop_size
    
    # Crossover input matches selection output
    assert resource_map.crossover.input_count == resource_map.selection.output_count
    
    # Mutation input matches crossover output
    assert resource_map.mutation.input_count == resource_map.crossover.output_count
