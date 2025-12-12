import unittest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

# --- Mock Imports (Replace with actual package imports) ---
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation import GaussianMutation, BallMutation, PolynomialMutation

class TestRealMutation(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(0)
        self.config = RealGenomeConfig(length=10, bounds=(-5.0, 5.0))
        # Create a population of size 5
        self.pop_size = 5
        self.population = RealPopulation.init_random(self.key, self.config, self.pop_size)

    def _test_operator(self, operator_cls, **kwargs):
        """Generic helper to test any mutation operator."""
        print(f"\nTesting {operator_cls.__name__}...")
        
        # 1. Instantiate
        mutator = operator_cls(num_offspring=1, **kwargs)
        
        # 2. Resource Allocation
        n_keys = mutator.num_keys(input_shape=(self.pop_size,))
        print(f"  Keys required: {n_keys}")
        
        k1, k2 = jar.split(self.key)
        keys = jar.split(k1, n_keys)
        
        # 3. Execution (JIT Compiled)
        jit_mutator = jax.jit(mutator)
        new_pop = jit_mutator(keys, self.population, self.config)
        
        # 4. Assertions
        # Shape must match
        self.assertEqual(new_pop.genes.values.shape, self.population.genes.values.shape)
        
        # Values should change (unless rate=0)
        diff = jnp.sum(jnp.abs(new_pop.genes.values - self.population.genes.values))
        print(f"  Total Change: {diff:.4f}")
        self.assertTrue(diff > 0, "Mutation failed to modify genes")

    def test_gaussian_mutation(self):
        self._test_operator(GaussianMutation, mutation_rate=1.0, mutation_strength=0.5)

    def test_ball_mutation(self):
        self._test_operator(BallMutation, mutation_rate=1.0, mutation_strength=0.5)

    def test_polynomial_mutation(self):
        self._test_operator(PolynomialMutation, mutation_rate=1.0, eta=20.0)

if __name__ == '__main__':
    unittest.main()