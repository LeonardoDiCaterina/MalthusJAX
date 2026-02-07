import unittest

import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import (
    BinomialCrossover_injection,
    BlendCrossover_injection,
    SimulatedBinaryCrossover_injection,
    UniformCrossover_injection,
)


class TestRealInjectionCrossover(unittest.TestCase):
    def setUp(self):
        self.key = jar.PRNGKey(42)
        self.config = RealGenomeConfig(shape=(10,), bounds=(-10.0, 10.0))

        # Create 2 Parent Populations (Size 4)
        k1, k2 = jar.split(self.key)
        self.pop_size = 4
        self.parents_1 = RealPopulation.init_random(k1, self.config, self.pop_size)

        # Make parents_2 distinctly different (all 10.0)
        p2_genes = RealGenome(values=jnp.full((self.pop_size, 10), 10.0))
        self.parents_2 = self.parents_1.spawn_offspring(p2_genes)

    def _test_crossover(self, operator_cls, num_offspring=1, **kwargs):
        print(f"\nTesting {operator_cls.__name__} (injection)...")

        # 1. Instantiate and set input_length
        crossover = operator_cls(num_offspring=num_offspring, **kwargs)
        crossover = crossover.set_input_length(self.pop_size)

        # 2. Resource Allocation
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
        vals = offspring_pop.genes.values
        is_same_p1 = jnp.allclose(vals, self.parents_1.genes.values.repeat(num_offspring, axis=0))
        is_same_p2 = jnp.allclose(vals, self.parents_2.genes.values.repeat(num_offspring, axis=0))

        self.assertFalse(is_same_p1 and is_same_p2, "Injection offspring are identical to parents")

    def test_uniform_injection(self):
        self._test_crossover(UniformCrossover_injection, num_offspring=1, crossover_rate=0.5)

    def test_blend_injection(self):
        self._test_crossover(BlendCrossover_injection, num_offspring=1, alpha=0.5)

    def test_sbx_injection(self):
        self._test_crossover(SimulatedBinaryCrossover_injection, num_offspring=2, eta=10.0)

    def test_binomial_injection(self):
        self._test_crossover(BinomialCrossover_injection, num_offspring=1, crossover_rate=0.9)

    def test_binomial_injection_rate_edges(self):
        # Rate=0 => preserve p2 (target)
        op = BinomialCrossover_injection(num_offspring=1, crossover_rate=0.0)
        op = op.set_input_length(self.pop_size)
        k = jar.PRNGKey(123)
        keys = jar.split(k, op.num_keys(input_shape=(self.pop_size,)))
        # swap parents to match operator's (target, mutant) parameter order
        out = op(keys, self.parents_2, self.parents_1, self.config)
        self.assertTrue(jnp.allclose(out.genes.values, self.parents_2.genes.values))

        # Rate=1 => preserve p1 (mutant)
        op = BinomialCrossover_injection(num_offspring=1, crossover_rate=1.0)
        op = op.set_input_length(self.pop_size)
        k = jar.PRNGKey(456)
        keys = jar.split(k, op.num_keys(input_shape=(self.pop_size,)))
        out = op(keys, self.parents_2, self.parents_1, self.config)
        self.assertTrue(jnp.allclose(out.genes.values, self.parents_1.genes.values))


if __name__ == "__main__":
    unittest.main()
