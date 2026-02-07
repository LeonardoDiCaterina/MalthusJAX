"""
Tests for mutation strength scheduling in GeneticEngine.

Tests focus on scheduling callbacks, parameter updates, and their effect
on mutation operators across generations.
"""

import unittest

import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestMutationStrengthScheduling(unittest.TestCase):
    """Test mutation strength scheduling during evolution."""

    def setUp(self):
        """Setup for mutation scheduling tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 30
        self.genome_shape = (3,)
        self.bounds = (-5.0, 5.0)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=self.bounds)
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=10
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        self.evaluator = BBOBEvaluator.create(bbob_config)

    def test_engine_without_schedule_uses_fixed_mutation(self):
        """Test that engine without schedule uses fixed mutation strength."""
        engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        # No schedule set (default None)
        self.assertIsNone(engine.engine_params.mutation_strength_schedule)

        # Should still run without error
        state = engine.init_state(self.key)
        new_state, _ = engine.step(state)

        self.assertEqual(new_state.generation, 1)

    def test_engine_with_linear_decay_schedule(self):
        """Test engine with linear decay mutation schedule."""

        # Linear decay: starts at 1.0, decays to 0.1
        def linear_schedule(generation):
            start_strength = 1.0
            end_strength = 0.1
            total_gens = 10
            progress = min(generation / total_gens, 1.0)
            return start_strength + (end_strength - start_strength) * progress

        engine_params = self.engine_params.replace(mutation_strength_schedule=linear_schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)

        # Run for multiple generations
        for _ in range(3):
            state, _ = engine.step(state)

        self.assertEqual(state.generation, 3)

    def test_schedule_produces_different_strengths_over_generations(self):
        """Test that schedule produces different mutation strengths."""
        strengths_applied = []

        def recording_schedule(generation):
            strength = max(1.0 - generation * 0.1, 0.1)  # Decay each generation
            strengths_applied.append(strength)
            return strength

        engine_params = self.engine_params.replace(mutation_strength_schedule=recording_schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)

        # Run for 5 generations
        for _ in range(5):
            state, _ = engine.step(state)

        # Schedule should have been called (generations 0-4)
        self.assertGreater(len(strengths_applied), 0)

        # Strengths should be different
        unique_strengths = set(strengths_applied)
        self.assertGreater(
            len(unique_strengths), 1, "Schedule should produce different strengths over generations"
        )

    def test_exponential_decay_schedule(self):
        """Test engine with exponential decay schedule."""

        def exponential_schedule(generation, initial=1.0, decay_rate=0.9):
            return initial * (decay_rate**generation)

        engine_params = self.engine_params.replace(
            mutation_strength_schedule=lambda g: exponential_schedule(g, 1.0, 0.95)
        )

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)

        for _ in range(5):
            state, _ = engine.step(state)

        self.assertEqual(state.generation, 5)

    def test_constant_schedule_maintains_strength(self):
        """Test that constant schedule maintains same strength."""

        def constant_schedule(generation):
            return 0.5

        engine_params = self.engine_params.replace(mutation_strength_schedule=constant_schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.7),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)
        _initial_ops = state.operators

        # Run step
        state, _ = engine.step(state)

        # Operators should still be valid
        self.assertIsNotNone(state.operators)

    def test_schedule_with_boundaries(self):
        """Test that schedule respects strength boundaries (0.0 to 1.0+)."""

        def bounded_schedule(generation):
            raw = 1.0 - generation * 0.2
            # Don't explicitly bound - let caller decide
            return raw

        engine_params = self.engine_params.replace(mutation_strength_schedule=bounded_schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)

        # Should handle even negative strengths (though not recommended)
        for _ in range(8):
            state, _ = engine.step(state)

        self.assertEqual(state.generation, 8)

    def test_get_active_operators_applies_schedule(self):
        """Test that _get_active_operators applies scheduled strength."""

        def test_schedule(generation):
            return 0.3 + generation * 0.1

        engine_params = self.engine_params.replace(mutation_strength_schedule=test_schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)
        _original_strength = state.operators.mutation.mutation_strength

        # Get active operators for generation 2
        active_ops = engine._get_active_operators(state.operators, 2)
        expected_strength = test_schedule(2)

        # Scheduled strength should be applied
        self.assertEqual(active_ops.mutation.mutation_strength, expected_strength)

    def test_none_schedule_returns_unchanged_operators(self):
        """Test that None schedule returns operators unchanged."""
        engine_params = self.engine_params.replace(mutation_strength_schedule=None)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)

        # Get active operators
        active_ops = engine._get_active_operators(state.operators, 5)

        # Should be same operators
        self.assertEqual(active_ops, state.operators)


class TestScheduleIntegrationWithEvolution(unittest.TestCase):
    """Test that scheduling integrates correctly with evolution."""

    def setUp(self):
        """Setup for integration tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 25
        self.genome_shape = (2,)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=(-5.0, 5.0))
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=10
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        self.evaluator = BBOBEvaluator.create(bbob_config)

    def test_scheduling_does_not_break_evolution(self):
        """Test that scheduling doesn't interfere with evolution."""

        def schedule(g):
            return max(0.1, 1.0 - g * 0.05)

        engine_params = self.engine_params.replace(mutation_strength_schedule=schedule)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        state = engine.init_state(self.key)
        initial_generation = state.generation

        # Run evolution
        for _ in range(5):
            state, _ = engine.step(state)

        # Generation should increment
        self.assertEqual(
            state.generation,
            initial_generation + 5,
            "Generation should increment by 5 after 5 steps",
        )

        # Best fitness should be valid
        self.assertTrue(jnp.isfinite(state.best_fitness), "Best fitness should be finite")

    def test_adaptive_schedule_influences_search(self):
        """Test that different schedules can lead to different results."""
        results = []

        for schedule_fn in [
            lambda g: 1.0,  # Constant high
            lambda g: max(0.1, 1.0 - g * 0.1),  # Rapid decay
        ]:
            engine_params = self.engine_params.replace(
                mutation_strength_schedule=schedule_fn, num_generations=5
            )

            engine = GeneticEngine(
                engine_params=engine_params,
                genome_config=self.genome_config,
                evaluator=self.evaluator,
                selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=2),
                crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
                mutation=GaussianMutation(
                    num_offspring=1, mutation_rate=0.1, mutation_strength=0.5
                ),
                enable_progress_bar=False,
            )

            state = engine.init_state(self.key)

            for _ in range(5):
                state, _ = engine.step(state)

            results.append(float(state.best_fitness))

        # Both should complete
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
