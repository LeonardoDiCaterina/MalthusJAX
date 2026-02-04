import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from malthusjax.engine.resource_mapper import ShardingManager, compute_resource_map, get_resource_summary
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.base import BaseSelection # Assume a mock/simple selection exists
from malthusjax.core.genome.real_genome import RealGenomeConfig

# --- Mocks for Testing ---
class MockSelection(BaseSelection):
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
    def _select(self, keys, fitness, config=None, **kwargs):
        return jnp.zeros(self.num_selections, dtype=jnp.int32)

@pytest.fixture
def engine_context():
    """Sets up standard operators and config for mapping tests."""
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0), dtype=jnp.float32)
    selection = MockSelection(num_selections=10)
    crossover = UniformCrossover(num_offspring=2)
    mutation = GaussianMutation(num_offspring=1)
    return selection, crossover, mutation, config

class TestResourceMapper:
    """Validates RNG budgeting and Cascade Data Flow logic."""

    def test_sharding_manager_allocation(self):
        """Verifies that the ShardingManager correctly places data on devices."""
        manager = ShardingManager(axis_name='batch')
        shape = (100, 10)
        pop_tensor = manager.alloc_population(shape)
        
        # Check sharding via addressable_shards or similar JAX inspect
        assert pop_tensor.shape == shape
        assert pop_tensor.sharding == manager.pop_sharding

    def test_compute_resource_map_cascade(self, engine_context):
        """Verifies the sequential indexing (no overlaps) in the RNG buffer."""
        sel, cross, mut, config = engine_context
        pop_size = 10
        
        rmap = compute_resource_map(sel, cross, mut, config, pop_size)
        
        # 1. Check for Index Overlaps
        assert rmap.selection.end_idx == rmap.crossover.start_idx
        assert rmap.crossover.end_idx == rmap.mutation.start_idx
        assert rmap.mutation.end_idx == rmap.next_key.start_idx
        
        # 2. Verify Total Budget matches sum of parts
        total_calculated = (rmap.selection.num_keys + 
                            rmap.crossover.num_keys + 
                            rmap.mutation.num_keys + 1)
        assert rmap.total_rng_budget == total_calculated

    def test_odd_population_size_handling(self, engine_context):
        """Ensures the mapper handles ceiling logic for odd population sizes."""
        sel, cross, mut, config = engine_context
        pop_size = 17 # Requires 9 pairs to get 18 offspring
        
        rmap = compute_resource_map(sel, cross, mut, config, pop_size)
        
        # Crossover should aim for 18 to cover 17 (assuming num_offspring=2)
        assert rmap.crossover.output_count == 18
        # Mutation inherits the 18 from crossover
        assert rmap.mutation.input_count == 18

    def test_get_key_slice(self, engine_context):
        """Verifies that slices return the correct start/end ranges."""
        sel, cross, mut, config = engine_context
        rmap = compute_resource_map(sel, cross, mut, config, pop_size=10)
        
        mut_slice = rmap.get_key_slice("mutation")
        assert mut_slice.start == rmap.mutation.start_idx
        assert mut_slice.stop == rmap.mutation.end_idx

    def test_resource_summary_output(self, engine_context):
        """Smoke test for the string summary generator."""
        sel, cross, mut, config = engine_context
        rmap = compute_resource_map(sel, cross, mut, config, pop_size=10)
        summary = get_resource_summary(rmap)
        
        assert "Total RNG Budget" in summary
        assert "[1. SELECTION]" in summary
        assert "[3. MUTATION]" in summary