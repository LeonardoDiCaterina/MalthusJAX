"""
Edge Case Tests for MalthusJAX Engines - Simplified and Focused.

Tests focus on:
- Extreme but valid parameters
- Engine stability with edge conditions
"""
import unittest

import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestEdgeCasePopulationSizes(unittest.TestCase):
    """Test extreme population sizes."""

    def _test_pop_size(self, pop_size):
        """Helper to test a population size."""
        genome_config = RealGenomeConfig(shape=(3,), bounds=(-5.0, 5.0))
        engine_params = GeneticEngineParams(
            pop_size=pop_size,
            elitism=max(1, pop_size // 10),
            num_generations=3
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=3, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=pop_size, elite_k=max(1, pop_size // 10)),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(jar.PRNGKey(42))
        final_state, _, elapsed_time = engine.run(state, compile=False)

        self.assertEqual(final_state.population.genes.values.shape[0], pop_size)

    def test_minimum_population(self):
        """Test with minimum population (2)."""
        self._test_pop_size(2)

    def test_odd_population(self):
        """Test with odd population (7)."""
        self._test_pop_size(7)

    def test_large_population(self):
        """Test with large population (200)."""
        self._test_pop_size(200)


class TestEdgeCaseGenerations(unittest.TestCase):
    """Test extreme generation counts."""

    def _test_generations(self, num_gens):
        """Helper to test generation count."""
        genome_config = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0))
        engine_params = GeneticEngineParams(
            pop_size=20,
            elitism=1,
            num_generations=num_gens
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=2, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=20, elite_k=2),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(jar.PRNGKey(42))
        final_state, _, elapsed_time = engine.run(state, compile=False)

        self.assertEqual(final_state.generation, num_gens)

    def test_single_generation(self):
        """Test with 1 generation."""
        self._test_generations(1)

    def test_many_generations(self):
        """Test with many generations."""
        self._test_generations(50)


class TestEdgeCaseGenomeDimensions(unittest.TestCase):
    """Test extreme genome dimensions."""

    def _test_dimension(self, dim):
        """Helper to test dimension."""
        genome_config = RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0))
        engine_params = GeneticEngineParams(
            pop_size=15,
            elitism=1,
            num_generations=3
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=dim, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=15, elite_k=1),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(jar.PRNGKey(42))
        final_state, _, elapsed_time = engine.run(state, compile=False)

        self.assertEqual(final_state.population.genes.values.shape, (15, dim))

    def test_1d_genome(self):
        """Test with 1D genome."""
        self._test_dimension(1)

    def test_high_dim_genome(self):
        """Test with high-dimensional genome (20D)."""
        self._test_dimension(20)


class TestEdgeCaseMutationRates(unittest.TestCase):
    """Test extreme mutation rates."""

    def _test_mutation_rate(self, mutation_rate):
        """Helper to test mutation rate."""
        genome_config = RealGenomeConfig(shape=(3,), bounds=(-5.0, 5.0))
        engine_params = GeneticEngineParams(
            pop_size=15,
            elitism=1,
            num_generations=5
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=3, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        engine = GeneticEngine(
            engine_params=engine_params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=15, elite_k=1),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=mutation_rate, mutation_strength=0.5),
            enable_progress_bar=False
        )

        state = engine.init_state(jar.PRNGKey(42))
        final_state, _, elapsed_time = engine.run(state, compile=False)

        # Just ensure it completes
        self.assertEqual(final_state.generation, 5)

    def test_zero_mutation_rate(self):
        """Test with zero mutation rate."""
        self._test_mutation_rate(0.0)

    def test_high_mutation_rate(self):
        """Test with high mutation rate."""
        self._test_mutation_rate(0.9)


if __name__ == '__main__':
    unittest.main()
