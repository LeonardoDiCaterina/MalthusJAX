import unittest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover, BlendCrossover, UniformCrossover

class TestRealCrossover(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(42)
        self.config = RealGenomeConfig(length=10, bounds=(-10.0, 10.0))
        
        # Create 2 Parent Populations (Size 4)
        k1, k2 = jar.split(self.key)
        self.pop_size = 4
        self.parents_1 = RealPopulation.init_random(k1, self.config, self.pop_size)
        
        # Make parents_2 distinctly different (all 10.0)
        # using a hack or init logic to ensure crossover is visible
        p2_genes = RealGenome(values=jnp.full((self.pop_size, 10), 10.0))
        self.parents_2 = self.parents_1.spawn_offspring(p2_genes)

    def _test_crossover(self, operator_cls, num_offspring=1, **kwargs):
        print(f"\nTesting {operator_cls.__name__}...")
        
        # 1. Instantiate
        crossover = operator_cls(num_offspring=num_offspring, **kwargs)
        
        # 2. Resource Allocation
        # Input shape is number of PAIRS
        n_keys = crossover.num_keys(input_shape=(self.pop_size,))
        print(f"  Keys required: {n_keys}")
        
        k_op, _ = jar.split(self.key)
        keys = jar.split(k_op, n_keys)
        
        # 3. Execution (JIT)
        jit_op = jax.jit(crossover)
        offspring_pop = jit_op(keys, self.parents_1, self.parents_2, self.config)
        
        # 4. Assertions
        expected_size = self.pop_size * num_offspring
        self.assertEqual(len(offspring_pop), expected_size)
        print(f"  Offspring Size: {len(offspring_pop)}")
        
        # Check values are mixed (between parents roughly)
        # Not a strict check for all operators, but good sanity check
        vals = offspring_pop.genes.values
        # Should not be exactly equal to parents (statistically)
        is_same_p1 = jnp.allclose(vals, self.parents_1.genes.values.repeat(num_offspring, axis=0))
        is_same_p2 = jnp.allclose(vals, self.parents_2.genes.values.repeat(num_offspring, axis=0))
        
        self.assertFalse(is_same_p1 and is_same_p2, "Offspring are identical to parents")

    def test_sbx(self):
        # SBX with 2 offspring (standard)
        self._test_crossover(SimulatedBinaryCrossover, num_offspring=2, eta=10.0)

    def test_blend(self):
        self._test_crossover(BlendCrossover, num_offspring=1, alpha=0.5)

    def test_uniform(self):
        self._test_crossover(UniformCrossover, num_offspring=1, crossover_rate=0.5)

if __name__ == '__main__':
    unittest.main()