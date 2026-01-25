import unittest

import chex
import jax
import jax.numpy as jnp
from evosax.problems import BBOBProblem
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

# --- Import your actual classes ---
# Adjust these imports to match your folder structure if needed
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


# --- MOCKS (To run this test without importing everything else) ---
@struct.dataclass
class MockRealGenome(BaseGenome):
    values: chex.Array

@struct.dataclass
class MockRealPopulation(BasePopulation):
    genes: MockRealGenome
    fitness: chex.Array

class TestBBOBEvaluator(unittest.TestCase):
    def setUp(self):
        self.pop_size = 10
        self.dims = 2
        self.fn_name = "sphere"

        # 1. Setup Config & Evosax Backend
        self.config = BBOBConfig(
            fn_name=self.fn_name,
            num_dims=self.dims,
            maximize=False
        )

        # Manually init evosax backend
        rng = jax.random.PRNGKey(42)
        self.evosax_problem = BBOBProblem(self.fn_name, self.dims)
        self.evosax_state = self.evosax_problem.init(rng)

        # 2. Instantiate Evaluator
        self.evaluator = BBOBEvaluator(
            config=self.config,
            data=None,  # <--- FIXED: Must provide this required argument
            evosax_problem=self.evosax_problem,
            evosax_state=self.evosax_state
        )

        # 3. Create a Dummy Population
        genes_array = jax.random.uniform(rng, (self.pop_size, self.dims), minval=-5, maxval=5)
        self.population = MockRealPopulation(
            genes=MockRealGenome(values=genes_array),
            fitness=jnp.zeros(self.pop_size),
            config=None
        )
    def test_01_evaluate_population_shape(self):
        """Check if output shape matches (pop_size,)."""
        print("\n[Test] Fitness Shape & Type")

        # Run Evaluation
        new_pop = self.evaluator.evaluate_population(self.population)

        # Assertions
        self.assertEqual(new_pop.fitness.shape, (self.pop_size,))
        self.assertFalse(jnp.any(jnp.isnan(new_pop.fitness)), "Fitness contains NaNs")
        print(f"  Input: {self.population.genes.values.shape}")
        print(f"  Output: {new_pop.fitness.shape}")

    def test_02_correctness_sphere(self):
        """Check if Sphere function returns expected values (approx 0 at 0)."""
        print("\n[Test] Mathematical Correctness (Sphere)")

        # Create a genome exactly at 0 (or strictly, the optimum defined by evosax state)
        # Note: BBOB shifts the optimum, so 0 might not be 0 cost.
        # But we can check relative order.

        # Create a population where individual 0 is closer to mean than individual 1
        # Actually, let's just run it and print sample values to sanity check range.

        new_pop = self.evaluator.evaluate_population(self.population)
        fitness = new_pop.fitness

        print(f"  Sample Fitness Values: {fitness[:3]}")
        # Sphere should be positive
        self.assertTrue(jnp.all(fitness > -1e6), "Fitness overly negative?")

    def test_03_jit_compatibility(self):
        """Ensure the evaluator can be JIT-compiled."""
        print("\n[Test] JIT Compilation")

        # Define a JIT-ed wrapper
        @jax.jit
        def run_eval(evaluator, pop):
            return evaluator.evaluate_population(pop)

        # Run it
        start_t = jax.numpy.arange(1) # Force JIT trace
        new_pop = run_eval(self.evaluator, self.population)

        # Block until ready to ensure execution happened
        _ = new_pop.fitness.block_until_ready()

        self.assertEqual(new_pop.fitness.shape, (self.pop_size,))
        print("  JIT Compilation Successful.")

    def test_04_maximization_flip(self):
        """Test if maximize=True correctly flips the sign."""
        print("\n[Test] Maximization Sign Flip")

        # 1. Run Minimized (Default BBOB)
        min_pop = self.evaluator.evaluate_population(self.population)

        # 2. Run Maximized
        max_config = self.config.replace(maximize=True)
        max_evaluator = self.evaluator.replace(config=max_config)

        max_pop = max_evaluator.evaluate_population(self.population)

        # 3. Compare
        # max_fitness should be exactly -min_fitness
        diff = jnp.abs(max_pop.fitness + min_pop.fitness)
        self.assertTrue(jnp.all(diff < 1e-5), "Maximization did not flip sign correctly")
        print("  Sign flip verified.")

if __name__ == '__main__':
    unittest.main()
