"""
Tests for CV-2 fix: assert → if/raise ValueError in _reproduction_phase.

Verifies that the three validation checks in _reproduction_phase raise
ValueError (not AssertionError) when shape mismatches occur.  This
ensures validation survives ``python -O`` (optimized mode).
"""

import unittest

import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    OperatorState,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


class TestReproductionPhaseValidation(unittest.TestCase):
    """Verify _reproduction_phase raises ValueError on shape mismatches."""

    def setUp(self):
        self.key = jar.PRNGKey(42)
        self.pop_size = 30
        genome_config = RealGenomeConfig(shape=(3,), bounds=(-5.0, 5.0))
        bbob_config = BBOBConfig(fn_name="sphere", num_dims=3, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)

        params = GeneticEngineParams(
            pop_size=self.pop_size, elitism=2, num_generations=10,
        )

        self.engine = GeneticEngine(
            engine_params=params,
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=self.pop_size, elite_k=3),
            crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
            mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
            enable_progress_bar=False,
        )

        self.state = self.engine.init_state(self.key)

    def test_raises_value_error_on_wrong_parent_count(self):
        """Mismatched parent_indices length → ValueError, not AssertionError."""
        rmap = self.state.resource_map
        operators = self.state.operators

        # Correct parent_indices has shape (rmap.crossover.input_count,)
        # We deliberately provide the wrong shape
        wrong_parent_indices = jnp.zeros(
            rmap.crossover.input_count + 5, dtype=jnp.int32
        )

        # Allocate valid keys for the correct size
        num_pairs = rmap.crossover.input_count // 2
        k_cross = jar.split(self.key, operators.crossover.num_keys(
            input_shape=(num_pairs,)
        ))
        k_mut = jar.split(self.key, rmap.mutation.num_keys)

        with self.assertRaises(ValueError):
            self.engine._reproduction_phase(
                k_cross, k_mut, wrong_parent_indices,
                self.state.population, operators, rmap,
            )

    def test_raises_value_error_on_wrong_crossover_keys(self):
        """Mismatched crossover keys count → ValueError, not AssertionError."""
        rmap = self.state.resource_map
        operators = self.state.operators

        # Correct parent indices
        parent_indices = jnp.zeros(rmap.crossover.input_count, dtype=jnp.int32)

        # Wrong number of crossover keys
        wrong_k_cross = jar.split(self.key, 1)  # Deliberately too few
        k_mut = jar.split(self.key, rmap.mutation.num_keys)

        with self.assertRaises(ValueError):
            self.engine._reproduction_phase(
                wrong_k_cross, k_mut, parent_indices,
                self.state.population, operators, rmap,
            )

    def test_no_assertion_error_type(self):
        """Ensure no AssertionError is raised (would fail under -O)."""
        rmap = self.state.resource_map
        operators = self.state.operators

        wrong_parent_indices = jnp.zeros(
            rmap.crossover.input_count + 5, dtype=jnp.int32
        )
        num_pairs = rmap.crossover.input_count // 2
        k_cross = jar.split(self.key, operators.crossover.num_keys(
            input_shape=(num_pairs,)
        ))
        k_mut = jar.split(self.key, rmap.mutation.num_keys)

        try:
            self.engine._reproduction_phase(
                k_cross, k_mut, wrong_parent_indices,
                self.state.population, operators, rmap,
            )
            self.fail("Expected ValueError to be raised")
        except ValueError:
            pass  # Expected
        except AssertionError:
            self.fail("Got AssertionError — CV-2 fix not applied (assert still present)")

    def test_valid_inputs_do_not_raise(self):
        """Normal execution should not raise any error."""
        k_sel, k_cross, k_mut, k_next = self.engine._allocate_entropy(self.state)
        active_ops = self.engine._get_active_operators(
            self.state.operators, self.state.generation
        )
        elites, parent_indices = self.engine._selection_phase(
            k_sel, self.state.population, active_ops, self.engine.engine_params,
        )
        # Should succeed without error
        mutants = self.engine._reproduction_phase(
            k_cross, k_mut, parent_indices,
            self.state.population, active_ops, self.state.resource_map,
        )
        self.assertIsNotNone(mutants)


if __name__ == "__main__":
    unittest.main()
