import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.compat.evosax_mimic import mutation as evosax_mutation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper


def test_mutation_matches_evosax_direct():
    key = jar.PRNGKey(777)
    cfg = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
    pop_size = 4

    k1, k2 = jar.split(key)
    parents = RealPopulation.init_random(k1, cfg, pop_size)

    for inj in (True, False):
        wrapper = (
            EvosaxGaussianWrapper(mutation_strength=0.2, injection_mode=inj)
            .set_input_length(pop_size)
        )
        if wrapper.injection_mode:
            expected_n = 1
        else:
            expected_n = pop_size * wrapper.num_offspring * wrapper.num_keys_per_atomic_operation
        n_keys = wrapper.num_keys(input_shape=(pop_size,))
        assert n_keys == expected_n

        k_op, _ = jar.split(key)
        keys = jar.split(k_op, n_keys)

        offspring_wrapper = jax.jit(wrapper)(keys, parents, cfg)

        # build subkeys for comparison, handling injection vs standard
        if wrapper.injection_mode:
            base_key = keys[0]
            subkeys = jar.split(base_key, pop_size * wrapper.num_offspring)
        else:
            # flatten  last dimension if legacy uint32; shape== (pop_size * K, 2)
            subkeys = keys.reshape((-1, keys.shape[-1]))

        # Each genome uses its corresponding subkey
        def _evosax_call(k, g):
            return evosax_mutation(k, g, jnp.array(wrapper.mutation_strength, dtype=cfg.dtype))

        # repeat genomes if num_offspring>1 (though parity test uses default K=1)
        genes_to_use = (
            jnp.repeat(parents.genes.values, wrapper.num_offspring, axis=0)
            if wrapper.num_offspring > 1
            else parents.genes.values
        )

        expected = jax.vmap(_evosax_call)(subkeys, genes_to_use)

        assert jnp.allclose(offspring_wrapper.genes.values, expected)
