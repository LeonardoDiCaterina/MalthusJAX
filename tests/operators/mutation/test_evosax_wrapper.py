import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper


class TestEvosaxGaussianWrapper(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(0)
        self.config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
        self.pop_size = 6
        self.population = RealPopulation.init_random(self.key, self.config, self.pop_size)

    def test_evosex_wrapper_mutation(self):
        mutator = EvosaxGaussianWrapper(mutation_strength=0.2)

        # Resource allocation
        n_keys = mutator.num_keys(input_shape=(self.pop_size,))
        self.assertEqual(n_keys, 1)

        k_op, _ = jar.split(self.key)
        keys = jar.split(k_op, n_keys)

        # JIT and execute
        jit_mutator = jax.jit(mutator)
        new_pop = jit_mutator(keys, self.population, self.config)

        # Basic assertions
        self.assertEqual(new_pop.genes.values.shape, self.population.genes.values.shape)
        diff = jnp.sum(jnp.abs(new_pop.genes.values - self.population.genes.values))
        # With non-zero mutation strength, we expect some change in the genes
        self.assertTrue(diff > 0, "Evosax wrapper failed to modify genes")

    def test_no_keys_raises(self):
        mutator = EvosaxGaussianWrapper(mutation_strength=0.2)
        # Pass an empty key array
        empty_keys = jnp.empty((0, 2), dtype=jnp.uint32)
        with self.assertRaises(ValueError):
            mutator(empty_keys, self.population, self.config)


if __name__ == "__main__":
    unittest.main()
