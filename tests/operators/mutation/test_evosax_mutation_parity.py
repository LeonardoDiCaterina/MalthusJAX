import jax
import jax.numpy as jnp
import jax.random as jar
from evosax.algorithms.population_based.simple_ga import mutation as evosax_mutation

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper


def test_mutation_matches_evosax_direct():
    key = jar.PRNGKey(777)
    cfg = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
    pop_size = 4

    k1, k2 = jar.split(key)
    parents = RealPopulation.init_random(k1, cfg, pop_size)

    wrapper = EvosaxGaussianWrapper(mutation_strength=0.2).set_input_length(pop_size)
    expected_n = pop_size * wrapper.num_offspring * wrapper.num_keys_per_atomic_operation
    n_keys = wrapper.num_keys(input_shape=(pop_size,))
    assert n_keys == expected_n

    k_op, _ = jar.split(key)
    keys = jar.split(k_op, n_keys)

    offspring_wrapper = jax.jit(wrapper)(keys, parents, cfg)

    # Directly apply evosax mutation per genome using matching per-sample keys
    flat = keys.reshape((-1, keys.shape[-1]))

    # Each genome uses its corresponding subkey
    def _evosax_call(k, g):
        return evosax_mutation(k, g, jnp.array(wrapper.mutation_strength, dtype=cfg.dtype))

    expected = jax.vmap(_evosax_call)(flat, parents.genes.values)

    assert jnp.allclose(offspring_wrapper.genes.values, expected)
