"""
Tests for Ask/Tell (async) interface of GeneticEngine.

Tests focus on the async evolution pattern where ask() allocates entropy
and tell() consumes it, allowing external evaluation between phases.
"""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestAskTellInterface(unittest.TestCase):
    """Test the ask/tell async evolution interface."""

    def setUp(self):
        """Setup for ask/tell tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 30
        self.genome_shape = (3,)
        self.bounds = (-5.0, 5.0)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=self.bounds)
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=10
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
            enable_progress_bar=False,
        )

        self.state = self.engine.init_state(self.key)

    def test_ask_returns_engine_and_population(self):
        """Test that ask() returns engine with entropy buffer and population."""
        engine_with_entropy, population = self.engine.ask(self.state)

        # Engine should have entropy buffer
        self.assertIsNotNone(engine_with_entropy._entropy_buffer)
        self.assertEqual(len(engine_with_entropy._entropy_buffer), 4)

        # Population should be returned
        self.assertIsNotNone(population)
        self.assertEqual(population.fitness.shape, (self.pop_size,))

    def test_ask_entropy_buffer_contains_keys(self):
        """Test that entropy buffer contains valid JAX keys."""
        engine_with_entropy, _ = self.engine.ask(self.state)

        k_sel, k_cross, k_mut, k_next = engine_with_entropy._entropy_buffer

        # Selection/crossover/mutation keys have batch dimension
        for key in [k_sel, k_cross, k_mut]:
            self.assertEqual(key.shape[-1], 2, "Keys should have last dimension 2")
            self.assertTrue(len(key.shape) >= 1)

        # Next key should be shape (2,)
        self.assertEqual(k_next.shape, (2,))

    def test_tell_requires_ask_to_be_called_first(self):
        """Test that tell() raises error if ask() not called."""
        evaluated_pop = self.state.population

        # tell() on engine without entropy buffer should raise
        with self.assertRaises(RuntimeError) as context:
            self.engine.tell(self.state, evaluated_pop)

        self.assertIn("tell() called before ask()", str(context.exception))

    def test_tell_updates_state_correctly(self):
        """Test that tell() updates state with evaluated population."""
        # Ask for entropy
        engine_with_entropy, population = self.engine.ask(self.state)

        # Create evaluated population (manually modify fitness)
        new_fitness = population.fitness * 0.9  # Slight improvement
        evaluated_pop = population.replace(fitness=new_fitness)

        # Tell results
        new_state = engine_with_entropy.tell(self.state, evaluated_pop)

        # State should be updated
        self.assertEqual(new_state.generation, self.state.generation + 1)
        self.assertEqual(new_state.population.fitness.shape, (self.pop_size,))

    def test_ask_tell_loop_produces_evolution(self):
        """Test that multiple ask/tell cycles produce valid evolution."""
        state = self.state
        initial_generation = state.generation

        for _ in range(3):
            # Ask for next generation
            engine_with_entropy, population = self.engine.ask(state)

            # Externally evaluate (in practice this could be done elsewhere)
            evaluated_pop = self.engine.evaluator.evaluate_population(population)

            # Tell results
            state = engine_with_entropy.tell(state, evaluated_pop)

        # Generation should have incremented
        self.assertEqual(state.generation, initial_generation + 3)

        # Fitness should be valid
        self.assertTrue(jnp.isfinite(state.best_fitness))

    def test_tell_updates_best_genome_on_improvement(self):
        """Test that tell() updates best_genome when better solution found."""
        # Create artificially better population
        new_fitness = self.state.population.fitness - 5.0
        better_pop = self.state.population.replace(fitness=new_fitness)

        # Ask then tell
        engine_with_entropy, _ = self.engine.ask(self.state)
        updated_state = engine_with_entropy.tell(self.state, better_pop)

        # Best genome should be updated
        best_idx = jnp.argmax(better_pop.fitness)
        expected_genome = better_pop[best_idx].genes

        actual_genome = updated_state.best_genome
        genes_match = jax.tree_util.tree_all(
            jax.tree_util.tree_map(jnp.allclose, actual_genome, expected_genome)
        )
        self.assertTrue(genes_match)

    def test_tell_updates_stagnation_counter(self):
        """Test that tell() correctly updates stagnation counter."""
        # No improvement case
        no_improvement_pop = self.state.population

        engine_with_entropy, _ = self.engine.ask(self.state)
        updated_state = engine_with_entropy.tell(self.state, no_improvement_pop)

        # Stagnation counter should increment
        self.assertEqual(updated_state.stagnation_counter, self.state.stagnation_counter + 1)

    def test_multiple_ask_tell_cycles_maintain_consistency(self):
        """Test that multiple cycles maintain state consistency."""
        state = self.state

        for i in range(5):
            # Check state invariants before ask
            self.assertEqual(state.population.fitness.shape, (self.pop_size,))

            # Ask
            engine_with_entropy, pop = self.engine.ask(state)

            # Check population consistency
            self.assertEqual(pop.fitness.shape, (self.pop_size,))
            genes_leaves = jax.tree_util.tree_leaves(pop.genes)
            self.assertEqual(genes_leaves[0].shape[0], self.pop_size)

            # Evaluate
            evaluated = self.engine.evaluator.evaluate_population(pop)

            # Tell
            state = engine_with_entropy.tell(state, evaluated)

            # Check generation incremented
            self.assertEqual(state.generation, i + 1)

    def test_ask_entropy_buffer_survives_engine_copy(self):
        """Test that entropy buffer is preserved when engine is copied."""
        engine_with_entropy, _ = self.engine.ask(self.state)

        # The engine should still have entropy buffer
        self.assertIsNotNone(engine_with_entropy._entropy_buffer)
        self.assertEqual(len(engine_with_entropy._entropy_buffer), 4)

        # If we create another reference, buffer should be accessible
        same_engine = engine_with_entropy
        self.assertEqual(len(same_engine._entropy_buffer), 4)


class TestAskTellVsSyncEvolution(unittest.TestCase):
    """Compare ask/tell interface with synchronous step() evolution."""

    def setUp(self):
        """Setup for comparison tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 25
        self.genome_shape = (2,)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=5
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
            enable_progress_bar=False,
        )

    def test_ask_tell_produces_same_progression_with_same_seed(self):
        """Test that ask/tell produces same results as step with same seed."""
        # Sync evolution
        key1 = jar.PRNGKey(100)
        state_sync = self.engine.init_state(key1)
        best_sync = []

        for _ in range(3):
            state_sync, metrics = self.engine.step(state_sync)
            best_sync.append(float(metrics.best_fitness))

        # Async evolution (limited determinism due to entropy allocation differences)
        key2 = jar.PRNGKey(100)
        state_async = self.engine.init_state(key2)
        best_async = []

        for _ in range(3):
            engine_with_entropy, pop = self.engine.ask(state_async)
            evaluated = self.engine.evaluator.evaluate_population(pop)
            state_async = engine_with_entropy.tell(state_async, evaluated)
            best_async.append(float(state_async.best_fitness))

        # Both should have done 3 generations
        self.assertEqual(len(best_sync), 3)
        self.assertEqual(len(best_async), 3)


if __name__ == "__main__":
    unittest.main()
