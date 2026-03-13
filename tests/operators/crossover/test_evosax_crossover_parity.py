import jax
import jax.numpy as jnp
import jax.random as jar

#from malthusjax.compat.evosax_mimic import crossover as evosax_crossover
from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper


def test_matches_evosax_direct():
    key = jar.PRNGKey(777)
    config = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
    pop_size = 4

    k1, k2 = jar.split(key)
    parents_1 = RealPopulation.init_random(k1, config, pop_size)
    p2_genes = RealGenome(values=jnp.full((pop_size, config.shape[0]), 1.0))
    parents_2 = parents_1.spawn_offspring(p2_genes)

    # run the same verification in both injection modes
    for inj in (True, False):
        wrapper = EvosaxUniformCrossoverWrapper(
            num_offspring=2, crossover_rate=0.7, injection_mode=inj
        ).set_input_length(pop_size)
        n_keys = wrapper.num_keys(input_shape=(pop_size,))
        if wrapper.injection_mode:
            expected_n = 1
        else:
            expected_n = pop_size * wrapper.num_offspring * wrapper.num_keys_per_atomic_operation
        assert n_keys == expected_n

        k_op, _ = jar.split(key)
        keys = jar.split(k_op, n_keys)

        offspring_wrapper = jax.jit(wrapper)(keys, parents_1, parents_2, config)

        # Build expected offspring by calling evosax.crossover per pair & offspring
        if wrapper.injection_mode:
            base_key = keys[0]
            subkeys = jar.split(base_key, pop_size * wrapper.num_offspring)
        else:
            flat = keys.reshape((-1, keys.shape[-1]))
            subkeys = flat  # keys are already per-offspring

    children = []
    for i in range(pop_size):
        pair_children = []
        for off in range(wrapper.num_offspring):
            idx = i * wrapper.num_offspring + off
            child_vals = evosax_crossover(
                subkeys[idx],
                parents_1.genes.values[i],
                parents_2.genes.values[i],
                wrapper.crossover_rate,
            )
            pair_children.append(child_vals)
        children.append(jnp.stack(pair_children))
    arr = jnp.stack(children)

    expected = arr.reshape((-1,) + arr.shape[2:])

    assert jnp.allclose(offspring_wrapper.genes.values, expected)
