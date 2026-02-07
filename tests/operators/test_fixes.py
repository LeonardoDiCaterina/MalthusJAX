"""
Comprehensive test suite for all performance and correctness fixes.

Tests cover:
- Fix 1: BaseSelection abstract methods enforcement
- Fix 2: Crossover return type consistency  
- Fix 2b: Crossover implementation updates
- Fix 3: getattr replacement for hasattr
- Fix 4: Single-pair crossover separation
- Fix 6: base_injection.py bug fixes
- Fix 7: Duplicate __index__ removal
"""

import pytest
import jax
import jax.numpy as jnp
from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection
from malthusjax.operators.crossover.real import UniformCrossover, UniformCrossover_injection
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


# ============================================================================
# FIX 1: BaseSelection Abstract Methods
# ============================================================================

class TestBaseSelectionAbstractMethods:
    """Test that BaseSelection enforces abstract method implementation."""

    def test_tournament_selection_implements_abstracts(self):
        """TournamentSelection must implement both abstract methods."""
        selector = TournamentSelection(num_selections=5, tournament_size=3)
        assert hasattr(selector, 'num_keys_per_atomic_operation')
        assert hasattr(selector, '_select')
        
    def test_selection_num_keys_per_atomic_is_property(self):
        """num_keys_per_atomic_operation must be a property."""
        selector = TournamentSelection(num_selections=5, tournament_size=3)
        # Should not raise AttributeError or be callable
        keys = selector.num_keys_per_atomic_operation
        assert isinstance(keys, int)

    def test_selection_select_is_abstract_method(self):
        """_select should be defined as abstract method."""
        # Just verify that BaseSelection has the abstract decorator
        assert hasattr(BaseSelection._select, '__isabstractmethod__')


# ============================================================================
# FIX 2: Crossover Return Type (G instead of Tuple[G, ...])
# ============================================================================

class TestCrossoverReturnTypes:
    """Test that crossover operators return single genomes, not tuples."""

    def test_uniform_crossover_recombine_returns_genome_not_tuple(self):
        """_recombine_one should return a single genome."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        op = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        # Generate noise (single key for Uniform crossover)
        noise_key = jax.random.split(k3, 1)[0:1]  # Keep shape (1, 2)
        noise = op._generate_noise(noise_key, config)
        
        # Call _recombine_one
        result = op._recombine_one(p1, p2, noise, config)
        
        # Result should be RealGenome, NOT a tuple
        assert isinstance(result, RealGenome), f"Expected RealGenome, got {type(result)}"
        assert not isinstance(result, tuple), "Result should not be a tuple"
        
    def test_uniform_crossover_injection_recombine_returns_genome(self):
        """Injection crossover should also return single genomes."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        op = UniformCrossover_injection(num_offspring=1, crossover_rate=0.5)
        op = op.set_input_length(1)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        noise_key = jax.random.split(k3, 1)[0]
        noise = op._generate_noise(noise_key, config)
        
        result = op._recombine_one(p1, p2, noise[0], config)
        
        assert isinstance(result, RealGenome)
        assert not isinstance(result, tuple)


# ============================================================================
# FIX 3: getattr replacement for hasattr
# ============================================================================

class TestSelectionAcceptsRawFitness:
    """Test that BaseSelection.__call__ uses getattr properly."""

    def test_selection_accepts_raw_fitness_array(self):
        """Selection should accept raw fitness array without Population wrapper."""
        key = jax.random.PRNGKey(42)
        fitness = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        indices = selector(key, fitness)
        
        assert indices.shape == (3,)
        assert jnp.all(indices < 5)
        assert jnp.all(indices >= 0)

    def test_selection_accepts_population_object(self):
        """Selection should accept Population object with .fitness attribute."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        pop_size = 5
        
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        
        pop = RealPopulation(
            genes=RealGenome.create_population(k1, config, pop_size),
            fitness=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            config=config
        )
        
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        indices = selector(k2, pop)
        
        assert indices.shape == (3,)
        assert jnp.all(indices < pop_size)


# ============================================================================
# FIX 4: Single-Pair Crossover Method
# ============================================================================

class TestCrossoverSinglePair:
    """Test the dedicated single-pair crossover method."""

    def test_cross_single_pair_produces_correct_shape(self):
        """cross_single_pair should produce offspring with shape (num_offspring, ...)."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        op = UniformCrossover(num_offspring=2, crossover_rate=0.5)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        offspring = op.cross_single_pair(k3, p1, p2, config)
        
        assert offspring.values.shape == (2, 10), f"Expected (2, 10), got {offspring.values.shape}"

    def test_cross_single_pair_is_jitable(self):
        """Single-pair crossover should be JIT-compilable."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        op = UniformCrossover(num_offspring=2, crossover_rate=0.5)
        
        @jax.jit
        def do_cross(key, p1, p2):
            return op.cross_single_pair(key, p1, p2, config)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        offspring = do_cross(k3, p1, p2)
        assert offspring.values.shape == (2, 10)

    def test_cross_single_pair_deterministic(self):
        """Same key should produce same offspring."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        op = UniformCrossover(num_offspring=2, crossover_rate=0.5)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        offspring1 = op.cross_single_pair(k3, p1, p2, config)
        offspring2 = op.cross_single_pair(k3, p1, p2, config)
        
        assert jnp.allclose(offspring1.values, offspring2.values)


# ============================================================================
# FIX 6: base_injection.py Bug Fixes
# ============================================================================

class TestInjectionModeBugs:
    """Test that injection mode bugs are fixed."""

    def test_injection_mutation_num_keys_per_atomic_returns_int(self):
        """num_keys_per_atomic_operation should return an int, not raise."""
        from malthusjax.operators.mutation.real import GaussianMutation_injection
        
        op = GaussianMutation_injection(num_offspring=1, mutation_rate=0.1)
        keys_per_op = op.num_keys_per_atomic_operation
        assert isinstance(keys_per_op, int)
        # Injection mutation requires 2 keys (mask and noise)
        assert keys_per_op == 2

    def test_injection_crossover_num_keys_per_atomic_returns_int(self):
        """Crossover injection should also return int."""
        op = UniformCrossover_injection(num_offspring=1, crossover_rate=0.5)
        keys_per_op = op.num_keys_per_atomic_operation
        assert isinstance(keys_per_op, int)

    def test_injection_crossover_executes_without_error(self):
        """Injection crossover vmap should be properly invoked (test _generate_noise)."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        
        op = UniformCrossover_injection(num_offspring=2)
        op = op.set_input_length(4)
        
        key = jax.random.PRNGKey(42)
        
        # Test that _generate_noise doesn't fail
        # (The main bug fix was that vmap was never invoked, which is hard to test in isolation)
        noise = op._generate_noise(key, config)
        
        # Verify noise has the right shape
        assert noise.shape == (4 * 2, 10)  # input_length * num_offspring, shape


# ============================================================================
# FIX 7: Duplicate __index__ Removal
# ============================================================================

class TestPopulationNoIndex:
    """Test that duplicate __index__ is removed."""

    def test_population_getitem_works(self):
        """__getitem__ should still work for indexing."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        pop_size = 5
        
        key = jax.random.PRNGKey(0)
        pop = RealPopulation(
            genes=RealGenome.create_population(key, config, pop_size),
            fitness=jnp.arange(pop_size, dtype=jnp.float32),
            config=config
        )
        
        # Integer indexing
        single = pop[0]
        assert isinstance(single, RealGenome)
        
        # Slice indexing
        sub_pop = pop[1:3]
        assert isinstance(sub_pop, RealPopulation)
        assert len(sub_pop) == 2

    def test_population_iteration_works(self):
        """Iteration should work via __getitem__."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        pop_size = 3
        
        key = jax.random.PRNGKey(0)
        pop = RealPopulation(
            genes=RealGenome.create_population(key, config, pop_size),
            fitness=jnp.arange(pop_size, dtype=jnp.float32),
            config=config
        )
        
        count = 0
        for individual in pop:
            assert isinstance(individual, RealGenome)
            count += 1
        
        assert count == pop_size

    def test_index_not_implemented_specially(self):
        """__index__ should not be specially defined (only __getitem__)."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        
        key = jax.random.PRNGKey(0)
        pop = RealPopulation(
            genes=RealGenome.create_population(key, config, 5),
            fitness=jnp.zeros(5),
            config=config
        )
        
        # __index__ should not be defined (or should be the default object.__index__)
        # which returns an error when called on a population
        # This tests that we can't do int(pop) accidentally
        with pytest.raises(TypeError):
            int(pop)


# ============================================================================
# Integration Tests
# ============================================================================

class TestFullEvolutionPipeline:
    """Test that all fixes work together in a full evolution cycle."""

    def test_selection_and_crossover_together(self):
        """Test selection feeding into crossover."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        pop_size = 10
        
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 5)
        
        # Create population
        pop = RealPopulation(
            genes=RealGenome.create_population(keys[0], config, pop_size),
            fitness=jnp.arange(pop_size, dtype=jnp.float32),
            config=config
        )
        
        # Select parents
        selector = TournamentSelection(num_selections=4, tournament_size=3)
        indices = selector(keys[1], pop)
        
        parents = pop[indices]
        assert len(parents) == 4
        
        # Crossover
        crossover_op = UniformCrossover(num_offspring=2, crossover_rate=0.5)
        crossover_op = crossover_op.set_input_length(2)  # 2 pairs
        
        p1_pop = parents[0:2]
        p2_pop = parents[2:4]
        
        crossover_keys = jax.random.split(keys[2], crossover_op.num_keys((2,)))
        offspring = crossover_op(crossover_keys, p1_pop, p2_pop, config)
        
        assert len(offspring) == 4  # 2 pairs * 2 offspring
        assert offspring.genes.values.shape == (4, 10)

    def test_single_pair_crossover_in_context(self):
        """Test single-pair crossover as a utility function."""
        config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
        
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        
        p1 = RealGenome.random_init(k1, config)
        p2 = RealGenome.random_init(k2, config)
        
        crossover_op = UniformCrossover(num_offspring=3)
        offspring = crossover_op.cross_single_pair(k3, p1, p2, config)
        
        assert offspring.values.shape == (3, 10)
        assert jnp.all(jnp.isfinite(offspring.values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
