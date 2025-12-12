import unittest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

# --- Mock Imports ---
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.crossover.binary import UniformCrossover, SinglePointCrossover

class TestBinaryCrossover(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(777)
        self.config = BinaryGenomeConfig(length=10)
        self.pop_size = 6
        
        # P1 = All Zeros
        zeros = jnp.zeros((self.pop_size, 10), dtype=jnp.int32)
        self.p1_genes = BinaryGenome(bits=zeros)
        self.parents_1 = BinaryPopulation(genes=self.p1_genes, fitness=jnp.zeros(6), config=None)
        
        # P2 = All Ones
        ones = jnp.ones((self.pop_size, 10), dtype=jnp.int32)
        self.p2_genes = BinaryGenome(bits=ones)
        self.parents_2 = self.parents_1.spawn_offspring(self.p2_genes)

    def _test_op(self, cls, **kwargs):
        print(f"\nTesting {cls.__name__}...")
        op = cls(num_offspring=1, **kwargs)
        
        n_keys = op.num_keys(input_shape=(self.pop_size,))
        keys = jar.split(self.key, n_keys)
        
        # Run
        offspring = jax.jit(op)(keys, self.parents_1, self.parents_2, self.config)
        
        # Results should be mix of 0s and 1s
        bits = offspring.genes.bits
        
        # Verify not all zeros and not all ones
        all_zeros = jnp.all(bits == 0)
        all_ones = jnp.all(bits == 1)
        
        print(f"  Result sample: {bits[0]}")
        self.assertFalse(all_zeros, "Crossover failed: Result is all Parent 1")
        self.assertFalse(all_ones, "Crossover failed: Result is all Parent 2")
        
        # Verify sum is roughly 50% (since P1=0, P2=1)
        avg_val = jnp.mean(bits)
        print(f"  Average Bit Value: {avg_val:.2f}")
        self.assertTrue(0.2 < avg_val < 0.8, "Mixing ratio seems off")

    def test_uniform(self):
        self._test_op(UniformCrossover, crossover_rate=0.5)

    def test_single_point(self):
        self._test_op(SinglePointCrossover)

if __name__ == '__main__':
    unittest.main()