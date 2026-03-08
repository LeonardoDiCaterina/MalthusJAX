import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper


class TestEvosaxUniformCrossoverWrapper(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(777)
        self.config = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
        self.pop_size = 4

        k1, k2 = jar.split(self.key)
        self.parents_1 = RealPopulation.init_random(k1, self.config, self.pop_size)
        # create a distinct second parent
        p2_genes = RealGenome(values=jnp.full((self.pop_size, self.config.shape[0]), 1.0))
        self.parents_2 = self.parents_1.spawn_offspring(p2_genes)

    def _test_wrapper(self, num_offspring=1, crossover_rate=0.5):
        wrapper = EvosaxUniformCrossoverWrapper(
            num_offspring=num_offspring, crossover_rate=crossover_rate
        ).set_input_length(self.pop_size)
        n_keys = wrapper.num_keys(input_shape=(self.pop_size,))
        if wrapper.injection_mode:
            expected_n = 1
        else:
            expected_n = int(self.pop_size * num_offspring * wrapper.num_keys_per_atomic_operation)
        self.assertEqual(n_keys, expected_n)

        k_op, _ = jar.split(self.key)
        keys = jar.split(k_op, n_keys)

        # Run wrapper
        jit_op = jax.jit(wrapper)
        offspring = jit_op(keys, self.parents_1, self.parents_2, self.config)

        # Compute expected masks using the same splitting logic as the wrapper.
        if wrapper.injection_mode:
            # keys array contains a single key
            base_key = keys[0]
            subkeys = jar.split(base_key, self.pop_size * num_offspring)
        else:
            flat = keys.reshape((-1, keys.shape[-1]))
            subkeys = flat  # already one key per offspring

        def per_row(k):
            return jar.bernoulli(k, p=crossover_rate, shape=self.config.shape)

        masks = jax.vmap(per_row)(subkeys)
        masks = masks.reshape((self.pop_size, num_offspring) + masks.shape[1:])

        # Build expected offspring values
        def per_pair(mask_block, p1, p2):
            # inner: iterate offspring
            return jax.vmap(lambda m: jnp.where(m, p2.values, p1.values), in_axes=0)(mask_block)

        vals = jax.vmap(per_pair)(masks, self.parents_1.genes, self.parents_2.genes)
        expected = vals.reshape((-1,) + vals.shape[2:])

        self.assertTrue(jnp.allclose(offspring.genes.values, expected))

    def test_uniform_single_offspring(self):
        self._test_wrapper(num_offspring=1, crossover_rate=0.5)

    def test_uniform_two_offspring(self):
        self._test_wrapper(num_offspring=2, crossover_rate=0.7)


if __name__ == "__main__":
    unittest.main()
