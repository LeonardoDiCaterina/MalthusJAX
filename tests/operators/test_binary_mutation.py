import unittest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation

class TestBinaryMutation(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(101)
        self.config = BinaryGenomeConfig(length=20)
        self.pop_size = 5
        # Assuming BinaryPopulation.init_random creates random bools/ints
        self.population = BinaryPopulation.init_random(self.key, self.config, self.pop_size)

    def _test_op(self, cls, **kwargs):
        print(f"\nTesting {cls.__name__}...")
        mutator = cls(num_offspring=1, **kwargs)
        
        n_keys = mutator.num_keys(input_shape=(self.pop_size,))
        keys = jar.split(self.key, n_keys)
        
        # Run
        new_pop = jax.jit(mutator)(keys, self.population, self.config)
        
        # Verify shape
        self.assertEqual(new_pop.genes.bits.shape, self.population.genes.bits.shape)
        
        # Hamming Distance Check
        # Since these are binary, simple != check works
        diffs = jnp.sum(new_pop.genes.bits != self.population.genes.bits)
        print(f"  Bits Flipped Total: {diffs}")
        if type(mutator) is SwapMutation:
            self.assertTrue(diffs == 0, "No bits should be changed in Swap Mutation")
        else:
            self.assertTrue(diffs > 0, "No bits were mutated")
            

    def test_bitflip(self):
        self._test_op(BitFlipMutation, mutation_rate=0.5)

    def test_scramble(self):
        # Scramble needs high rate to be visible on small genomes
        self._test_op(ScrambleMutation, mutation_rate=1.0)

    def test_swap(self):
        self._test_op(SwapMutation, mutation_rate=1.0)

if __name__ == '__main__':
    unittest.main()