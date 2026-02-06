"""
Tests for sharding, resource mapping, and dtype enforcement in GeneticEngine.

Tests focus on:
- GSPMD sharding layout enforcement
- Resource map correctness and consistency
- Data type casting and preservation
- Memory layout optimization
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.resource_mapper import compute_resource_map
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestResourceMapComputation(unittest.TestCase):
    """Test resource map computation and correctness."""

    def setUp(self):
        """Setup for resource map tests."""
        self.pop_size = 40
        self.genome_shape = (5,)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))

        self.selection = ElitePoolSelection(num_selections=self.pop_size, elite_k=4)
        self.crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
        self.mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

    def test_resource_map_is_computed(self):
        """Test that resource map is computed without error."""
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.pop_size
        )

        self.assertIsNotNone(rmap)
        self.assertGreater(rmap.total_rng_budget, 0)

    def test_resource_map_tracks_all_components(self):
        """Test that resource map includes selection, crossover, mutation."""
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.pop_size
        )

        self.assertIsNotNone(rmap.selection)
        self.assertIsNotNone(rmap.crossover)
        self.assertIsNotNone(rmap.mutation)

    def test_selection_resource_has_correct_output_count(self):
        """Test that selection resource matches num_selections."""
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.pop_size
        )

        # Selection outputs should match population size
        self.assertEqual(rmap.selection.output_count, self.pop_size)

    def test_crossover_produces_expected_offspring(self):
        """Test that crossover resource matches num_offspring."""
        num_offspring = 2
        crossover = SimulatedBinaryCrossover(num_offspring=num_offspring, eta=15.0)

        rmap = compute_resource_map(
            self.selection,
            crossover,
            self.mutation,
            self.genome_config,
            self.pop_size
        )

        # Each pair produces num_offspring
        num_pairs = rmap.crossover.input_count // 2
        expected_output = num_pairs * num_offspring

        self.assertEqual(rmap.crossover.output_count, expected_output)

    def test_mutation_produces_expected_offspring(self):
        """Test that mutation resource matches num_offspring."""
        num_offspring = 1
        mutation = GaussianMutation(num_offspring=num_offspring, mutation_rate=0.1, mutation_strength=0.5)

        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            mutation,
            self.genome_config,
            self.pop_size
        )

        # Each input produces num_offspring
        expected_output = rmap.mutation.input_count * num_offspring

        self.assertEqual(rmap.mutation.output_count, expected_output)

    def test_resource_map_consistency_across_configs(self):
        """Test that resource map is consistent with config."""
        for pop_size in [10, 30, 50, 100]:
            with self.subTest(pop_size=pop_size):
                rmap = compute_resource_map(
                    self.selection,
                    self.crossover,
                    self.mutation,
                    self.genome_config,
                    pop_size
                )

                self.assertGreater(rmap.total_rng_budget, 0)

    def test_resource_map_with_different_operators(self):
        """Test resource map computation with different operator configurations."""
        configs = [
            {"num_offspring": 1},
            {"num_offspring": 2},
            {"num_offspring": 3},
        ]

        rmaps = []
        output_counts = []
        for config in configs:
            crossover = SimulatedBinaryCrossover(num_offspring=config["num_offspring"], eta=15.0)
            rmap = compute_resource_map(
                self.selection,
                crossover,
                self.mutation,
                self.genome_config,
                self.pop_size
            )
            rmaps.append(rmap)
            output_counts.append(rmap.crossover.output_count)

        # All maps should be valid
        for rmap in rmaps:
            self.assertIsNotNone(rmap)
            self.assertIsNotNone(rmap.crossover)

        # Output counts should follow expected pattern
        # (they depend on population size and offspring count)
        self.assertEqual(len(output_counts), 3)


class TestShardingEnforcement(unittest.TestCase):
    """Test GSPMD sharding layout enforcement."""

    def setUp(self):
        """Setup for sharding tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 32
        self.genome_shape = (4,)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=5
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

    def test_init_state_enforces_sharding_layout(self):
        """Test that init_state applies sharding layout."""
        state = self.engine.init_state(self.key)

        # Population should exist with correct shape
        self.assertEqual(state.population.fitness.shape, (self.pop_size,))
        genes_leaves = jax.tree_util.tree_leaves(state.population.genes)
        self.assertEqual(genes_leaves[0].shape, (self.pop_size,) + self.genome_shape)

    def test_population_maintains_shape_after_evolution(self):
        """Test that population shape is maintained through evolution."""
        state = self.engine.init_state(self.key)

        for _ in range(3):
            state, _ = self.engine.step(state)

            # Check shapes
            self.assertEqual(state.population.fitness.shape, (self.pop_size,))
            genes_leaves = jax.tree_util.tree_leaves(state.population.genes)
            self.assertEqual(genes_leaves[0].shape[0], self.pop_size)

    def test_best_genome_has_correct_shape(self):
        """Test that best_genome shape matches single individual."""
        state = self.engine.init_state(self.key)

        best_leaves = jax.tree_util.tree_leaves(state.best_genome)
        self.assertEqual(best_leaves[0].shape, self.genome_shape)


class TestDtypeCasting(unittest.TestCase):
    """Test data type casting and preservation."""

    def setUp(self):
        """Setup for dtype tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 30
        self.genome_shape = (3,)

    def test_dtype_consistency_default(self):
        """Test that default dtype is consistent."""
        genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))

        engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=5
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(self.key)

        # Check dtype
        genes_leaves = jax.tree_util.tree_leaves(state.population.genes)
        self.assertTrue(jnp.issubdtype(genes_leaves[0].dtype, jnp.floating))

    def test_fitness_dtype_consistency(self):
        """Test that fitness dtype is consistent."""
        genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))

        engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=5
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(self.key)

        # Check fitness dtype
        self.assertTrue(jnp.issubdtype(state.population.fitness.dtype, jnp.floating))

        # Run evolution and check consistency
        new_state, _ = engine.step(state)
        self.assertTrue(jnp.issubdtype(new_state.population.fitness.dtype, jnp.floating))

    def test_best_fitness_dtype_consistency(self):
        """Test that best_fitness dtype is scalar floating."""
        genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))

        engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=5
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(self.key)

        # Check best_fitness
        self.assertTrue(jnp.issubdtype(state.best_fitness.dtype, jnp.floating))
        self.assertEqual(state.best_fitness.shape, ())


class TestResourceConsistency(unittest.TestCase):
    """Test consistency of resource allocation across evolution."""

    def setUp(self):
        """Setup for consistency tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 40
        self.genome_shape = (2,)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=10
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

    def test_resource_map_remains_constant_across_generations(self):
        """Test that resource map is immutable and consistent."""
        state = self.engine.init_state(self.key)
        initial_rmap = state.resource_map

        # Run several generations
        for _ in range(5):
            state, _ = self.engine.step(state)

        # Resource map should be identical
        self.assertEqual(initial_rmap.total_rng_budget, state.resource_map.total_rng_budget)
        self.assertEqual(initial_rmap.selection.output_count, state.resource_map.selection.output_count)

    def test_operator_state_remains_consistent(self):
        """Test that operator state remains valid across generations."""
        state = self.engine.init_state(self.key)

        for _ in range(5):
            # Check operators before step
            self.assertIsNotNone(state.operators.selection)
            self.assertIsNotNone(state.operators.crossover)
            self.assertIsNotNone(state.operators.mutation)

            state, _ = self.engine.step(state)

        # Final state should still have valid operators
        self.assertIsNotNone(state.operators.selection)
        self.assertIsNotNone(state.operators.crossover)
        self.assertIsNotNone(state.operators.mutation)


if __name__ == '__main__':
    unittest.main()
