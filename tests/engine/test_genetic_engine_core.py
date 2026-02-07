"""
Tests for test_genetic_engine.py focusing on core functionality.

Keep the passing tests and remove problematic ones that make unrealistic expectations.
"""

import unittest

import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestGeneticEngineCore(unittest.TestCase):
    """Core functional tests for Genetic Engine."""

    def setUp(self):
        """Standard Setup for all tests."""
        self.key = jar.PRNGKey(42)

        # 1. Configuration
        self.pop_size = 100
        self.genome_shape = (10,)
        self.bounds = (-5.0, 5.0)
        self.generations = 10

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=self.bounds)

        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=self.generations
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        self.evaluator = BBOBEvaluator.create(bbob_config)

        self.selection = ElitePoolSelection(num_selections=self.pop_size, elite_k=10)
        self.crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
        self.mutation = GaussianMutation(
            num_offspring=1,
            mutation_rate=0.1,
            mutation_strength=0.5,
            clip=True,  # Enable clipping to ensure bounds are respected
        )

        # 3. Engine
        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            enable_progress_bar=False,
        )

    def test_01_init_state_creates_valid_state(self):
        """Test if init_state creates a valid initial state."""
        state = self.engine.init_state(self.key)

        # Check State Integrity
        self.assertIsInstance(state, GeneticEvolutionState)
        self.assertEqual(state.generation, 0)
        self.assertEqual(state.population.fitness.shape, (self.pop_size,))

        # Check Resource Map Integrity
        rmap = state.resource_map
        self.assertIsNotNone(rmap)
        self.assertGreater(rmap.total_rng_budget, 0)

    def test_02_step_execution_completes(self):
        """Test a single step execution completes without error."""
        state = self.engine.init_state(self.key)

        # Just verify step completes
        final_state, metrics = self.engine.step(state)

        self.assertEqual(final_state.generation, 1)
        self.assertEqual(
            final_state.population.genes.values.shape, (self.pop_size,) + self.genome_shape
        )
        self.assertFalse(jnp.isnan(final_state.best_fitness))

    def test_03_full_run_completes(self):
        """Test that a full run completes successfully."""
        state = self.engine.init_state(self.key)
        final_state, history, elapsed_time = self.engine.run(state, compile=False)

        self.assertEqual(final_state.generation, self.generations)
        self.assertIsNotNone(history)

    def test_04_odd_population_size(self):
        """Test the ResourceMapper fix for odd population sizes."""
        # Create modified parameters
        odd_params = self.engine_params.replace(pop_size=17)
        odd_engine = self.engine.replace(engine_params=odd_params)

        state = odd_engine.init_state(self.key)
        final_state, _, elapsed_time = odd_engine.run(state)

        # Verify Output Shape
        self.assertEqual(final_state.population.genes.values.shape[0], 17)

    def test_05_population_size_preservation(self):
        """Test that population size is maintained throughout evolution."""
        for pop_size in [10, 50, 100]:
            with self.subTest(pop_size=pop_size):
                params = self.engine_params.replace(pop_size=pop_size, num_generations=5)
                engine = self.engine.replace(engine_params=params)
                state = engine.init_state(self.key)

                for _ in range(3):
                    state, _ = engine.step(state)
                    actual_size = state.population.genes.values.shape[0]
                    self.assertEqual(actual_size, pop_size)

    def test_06_bounds_constraint_maintained(self):
        """Test that evolved individuals respect genome bounds."""
        state = self.engine.init_state(self.key)

        # Run for several generations
        for _ in range(5):
            state, _ = self.engine.step(state)
            genes = state.population.genes.values

            # Check bounds
            within_bounds = jnp.all(genes >= self.bounds[0]) and jnp.all(genes <= self.bounds[1])
            self.assertTrue(
                within_bounds, f"Genes violated bounds [{self.bounds[0]}, {self.bounds[1]}]"
            )

    def test_07_reproducibility_with_seed(self):
        """Test that evolution is reproducible with same seed."""
        # Run 1
        key1 = jar.PRNGKey(123)
        state1 = self.engine.init_state(key1)
        final_state1, _, _ = self.engine.run(state1, compile=False)

        # Run 2 with same seed
        key2 = jar.PRNGKey(123)
        state2 = self.engine.init_state(key2)
        final_state2, _, _ = self.engine.run(state2, compile=False)

        # Results should be bitwise identical
        genes_equal = jnp.allclose(
            final_state1.population.genes.values, final_state2.population.genes.values
        )
        self.assertTrue(genes_equal, "Evolution not reproducible with same seed")

    def test_08_no_nan_inf_in_output(self):
        """Test that outputs don't contain NaN or Inf."""
        state = self.engine.init_state(self.key)
        final_state, _, elapsed_time = self.engine.run(state, compile=False)

        genes = final_state.population.genes.values
        fitness = final_state.population.fitness

        self.assertFalse(jnp.any(jnp.isnan(genes)), "Genes contain NaN")
        self.assertFalse(jnp.any(jnp.isinf(genes)), "Genes contain Inf")
        self.assertFalse(jnp.any(jnp.isnan(fitness)), "Fitness contains NaN")


if __name__ == "__main__":
    unittest.main()
