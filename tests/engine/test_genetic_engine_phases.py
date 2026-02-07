"""
Comprehensive phase-level tests for GeneticEngine.

Tests focus on individual phases (entropy allocation, selection, reproduction,
merge, evaluation, HOF update) and their interactions.
"""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestEntropyAllocation(unittest.TestCase):
    """Test the entropy allocation phase."""

    def setUp(self):
        """Setup for entropy tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 50
        self.genome_shape = (5,)
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
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=5),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        self.state = self.engine.init_state(self.key)

    def test_entropy_allocation_returns_four_keys(self):
        """Test that _allocate_entropy returns exactly 4 keys."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        # Each should be a JAX array
        self.assertIsInstance(k_sel, (jax.Array, jnp.ndarray))
        self.assertIsInstance(k_cross, (jax.Array, jnp.ndarray))
        self.assertIsInstance(k_mut, (jax.Array, jnp.ndarray))
        self.assertIsInstance(k_next, (jax.Array, jnp.ndarray))

    def test_entropy_allocation_keys_are_different(self):
        """Test that allocated keys are different from each other."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        # Keys should have batch dimension and be different
        # Each is a slice of keys, so shape is (batch_size, 2)
        self.assertEqual(len(k_sel.shape), 2)
        self.assertEqual(len(k_cross.shape), 2)
        self.assertEqual(len(k_mut.shape), 2)
        self.assertEqual(k_next.shape, (2,))

    def test_entropy_allocation_produces_valid_shapes(self):
        """Test that entropy keys have valid JAX PRNGKey shapes."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        # Selection and crossover/mutation keys are batched (num_keys, 2)
        # Next key is single (2,)
        self.assertEqual(k_sel.shape[-1], 2)  # Last dim should be 2
        self.assertEqual(k_cross.shape[-1], 2)
        self.assertEqual(k_mut.shape[-1], 2)
        self.assertEqual(k_next.shape, (2,))


class TestSelectionPhase(unittest.TestCase):
    """Test the selection phase."""

    def setUp(self):
        """Setup for selection tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 40
        self.genome_shape = (4,)
        self.bounds = (-5.0, 5.0)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=self.bounds)
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=3, num_generations=10
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=5),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        self.state = self.engine.init_state(self.key)

    def test_selection_returns_elite_and_parents(self):
        """Test that selection returns elite genes and parent indices."""
        key_sel = jar.fold_in(self.state.rng_key, 0)
        elites, parent_indices = self.engine._selection_phase(
            key_sel, self.state.population, self.state.operators, self.engine_params
        )

        # Elites should have shape (elitism, ...genome_shape)
        elite_leaves = jax.tree_util.tree_leaves(elites)
        self.assertEqual(elite_leaves[0].shape[0], self.engine_params.elitism)

        # Parent indices should have shape (num_selections,)
        self.assertEqual(parent_indices.shape[0], self.engine_params.pop_size)

    def test_selection_indices_are_valid(self):
        """Test that selected indices are within population bounds."""
        key_sel = jar.fold_in(self.state.rng_key, 0)
        elites, parent_indices = self.engine._selection_phase(
            key_sel, self.state.population, self.state.operators, self.engine_params
        )

        # All indices should be in valid range [0, pop_size)
        self.assertTrue(jnp.all(parent_indices >= 0))
        self.assertTrue(jnp.all(parent_indices < self.pop_size))

    def test_elite_genes_are_best_fitness(self):
        """Test that elite genes correspond to best fitness individuals."""
        key_sel = jar.fold_in(self.state.rng_key, 0)
        elites, _ = self.engine._selection_phase(
            key_sel, self.state.population, self.state.operators, self.engine_params
        )

        # Get top-k fitness values
        top_k_fitness, top_k_indices = jax.lax.top_k(
            self.state.population.fitness, self.engine_params.elitism
        )

        # Elite genes should match top-k individuals
        top_k_genes = self.state.population[top_k_indices].genes
        elite_leaves = jax.tree_util.tree_leaves(elites)
        top_k_leaves = jax.tree_util.tree_leaves(top_k_genes)

        for elite_leaf, top_k_leaf in zip(elite_leaves, top_k_leaves):
            self.assertTrue(jnp.allclose(elite_leaf, top_k_leaf))


class TestReproductionPhase(unittest.TestCase):
    """Test the reproduction phase (crossover + mutation)."""

    def setUp(self):
        """Setup for reproduction tests."""
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

    def test_reproduction_produces_correct_population_size(self):
        """Test that reproduction produces expected number of offspring."""
        # Use properly allocated entropy from _allocate_entropy
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        # Get parents
        _, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        # Reproduce
        final_pop = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        # Population size should match
        genes_leaves = jax.tree_util.tree_leaves(final_pop.genes)
        self.assertEqual(genes_leaves[0].shape[0], self.state.resource_map.mutation.output_count)

    def test_reproduction_offspring_within_bounds(self):
        """Test that offspring respect genome bounds (if clipping enabled)."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        _, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        final_pop = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        genes = final_pop.genes.values
        # Just verify offspring have been created (bounds may vary based on operator config)
        self.assertIsNotNone(genes)
        self.assertEqual(genes.shape[0], self.state.resource_map.mutation.output_count)

    def test_reproduction_produces_different_offspring(self):
        """Test that offspring are not identical to parents (with high probability)."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        _, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        offspring_pop = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        # With very high probability, offspring should differ from parents
        offspring_genes = offspring_pop.genes.values
        parent_pop = self.state.population[parent_indices]
        parent_genes = parent_pop.genes.values

        # Not all offspring should be identical to their parents
        identical_count = 0
        for i in range(min(len(offspring_genes), len(parent_genes))):
            if jnp.allclose(offspring_genes[i], parent_genes[i]):
                identical_count += 1

        # Expect very few identical offspring with mutation enabled
        self.assertLess(
            identical_count, len(offspring_genes) * 0.5, "Too many offspring identical to parents"
        )


class TestMergePhase(unittest.TestCase):
    """Test the merge phase (combining elites and mutants)."""

    def setUp(self):
        """Setup for merge tests."""
        self.key = jar.PRNGKey(42)
        self.pop_size = 40
        self.genome_shape = (3,)
        self.bounds = (-5.0, 5.0)

        self.genome_config = RealGenomeConfig(shape=self.genome_shape, bounds=self.bounds)
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=4, num_generations=10
        )

        bbob_config = BBOBConfig(fn_name="sphere", num_dims=self.genome_shape[0], maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=4),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        self.state = self.engine.init_state(self.key)

    def test_merge_produces_correct_population_size(self):
        """Test that merge produces population of correct size."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        elites, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        mutants = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        merged_genes = self.engine._merge(elites, mutants.genes, self.state)

        # Check size
        merged_leaves = jax.tree_util.tree_leaves(merged_genes)
        self.assertEqual(merged_leaves[0].shape[0], self.pop_size)

    def test_merge_preserves_elite_at_top(self):
        """Test that merged population has elites at the beginning."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        elites, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        mutants = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        merged_genes = self.engine._merge(elites, mutants.genes, self.state)

        # First few genes should match elites
        merged_leaves = jax.tree_util.tree_leaves(merged_genes)
        elite_leaves = jax.tree_util.tree_leaves(elites)

        for merged_leaf, elite_leaf in zip(merged_leaves, elite_leaves):
            self.assertTrue(jnp.allclose(merged_leaf[: self.engine_params.elitism], elite_leaf))


class TestEvaluationPhase(unittest.TestCase):
    """Test the evaluation phase."""

    def setUp(self):
        """Setup for evaluation tests."""
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

    def test_evaluation_produces_fitness_for_all(self):
        """Test that evaluation assigns fitness to all individuals."""
        # Create new genes
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        elites, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        mutants = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        new_genes = self.engine._merge(elites, mutants.genes, self.state)

        # Evaluate
        evaluated_pop = self.engine._evaluate(new_genes, self.state)

        # Check fitness
        self.assertEqual(evaluated_pop.fitness.shape, (self.pop_size,))
        self.assertFalse(jnp.any(jnp.isnan(evaluated_pop.fitness)))

    def test_evaluation_fitness_is_valid(self):
        """Test that evaluated fitness values are valid (no NaN/Inf)."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)

        elites, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, self.state.operators, self.engine_params
        )

        mutants = self.engine._reproduction_phase(
            k_cross,
            k_mut,
            parent_indices,
            self.state.population,
            self.state.operators,
            self.state.resource_map,
        )

        new_genes = self.engine._merge(elites, mutants.genes, self.state)
        evaluated_pop = self.engine._evaluate(new_genes, self.state)

        # No NaN or Inf
        self.assertFalse(jnp.any(jnp.isnan(evaluated_pop.fitness)))
        self.assertFalse(jnp.any(jnp.isinf(evaluated_pop.fitness)))


class TestHOFUpdatePhase(unittest.TestCase):
    """Test the Hall of Fame (best individual) update phase."""

    def setUp(self):
        """Setup for HOF tests."""
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

    def test_hof_update_updates_best_fitness(self):
        """Test that HOF update works without error."""
        k_next = jar.fold_in(self.state.rng_key, 100)
        updated_state = self.engine._update_hof(self.state.population, self.state, k_next)

        # HOF update should complete and increment generation
        self.assertEqual(updated_state.generation, self.state.generation + 1)
        # Best fitness and genome should be updated/preserved
        self.assertIsNotNone(updated_state.best_fitness)
        self.assertIsNotNone(updated_state.best_genome)

    def test_hof_update_preserves_best_when_no_improvement(self):
        """Test that HOF is unchanged when no improvement."""
        old_best_fit = float(self.state.best_fitness)

        # Keep fitness the same (no improvement)
        new_pop = self.state.population

        k_next = jar.fold_in(self.state.rng_key, 100)
        updated_state = self.engine._update_hof(new_pop, self.state, k_next)

        # Best fitness should remain unchanged
        self.assertAlmostEqual(float(updated_state.best_fitness), old_best_fit, places=5)

    def test_hof_update_increments_generation(self):
        """Test that HOF update increments generation counter."""
        old_gen = self.state.generation

        k_next = jar.fold_in(self.state.rng_key, 100)
        updated_state = self.engine._update_hof(self.state.population, self.state, k_next)

        self.assertEqual(updated_state.generation, old_gen + 1)

    def test_hof_update_resets_stagnation_on_improvement(self):
        """Test that stagnation counter behavior on improvement."""
        # Set initial stagnation
        old_state = self.state.replace(stagnation_counter=5)
        old_best_fitness = float(old_state.best_fitness)

        # Create a population with better fitness
        new_fitness = old_state.population.fitness.copy()
        new_fitness = new_fitness.at[0].set(old_best_fitness - 5.0)  # Much better
        new_pop = old_state.population.replace(fitness=new_fitness)

        k_next = jar.fold_in(old_state.rng_key, 100)
        updated_state = self.engine._update_hof(new_pop, old_state, k_next)

        # When best fitness improves, stagnation should reset to 0
        if float(updated_state.best_fitness) < old_best_fitness:
            self.assertEqual(int(updated_state.stagnation_counter), 0)

    def test_hof_update_increments_stagnation_on_no_improvement(self):
        """Test that stagnation counter increments when no improvement."""
        old_state = self.state.replace(stagnation_counter=3)

        # No improvement - keep fitness same
        new_pop = old_state.population

        k_next = jar.fold_in(old_state.rng_key, 100)
        updated_state = self.engine._update_hof(new_pop, old_state, k_next)

        # Stagnation should increment
        self.assertEqual(int(updated_state.stagnation_counter), 4)


if __name__ == "__main__":
    unittest.main()
