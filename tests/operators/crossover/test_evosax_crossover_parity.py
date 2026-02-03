import jax
import jax.numpy as jnp
import jax.random as jar
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

    wrapper = EvosaxUniformCrossoverWrapper(num_offspring=2, crossover_rate=0.7).set_input_length(
        pop_size
    )
    n_keys = wrapper.num_keys(input_shape=(pop_size,))
    expected_n = pop_size * wrapper.num_offspring * wrapper.num_keys_per_atomic_operation
    assert n_keys == expected_n

    k_op, _ = jar.split(key)
    keys = jar.split(k_op, n_keys)

    offspring_wrapper = jax.jit(wrapper)(keys, parents_1, parents_2, config)

    # Build expected offspring by calling evosax.crossover per pair & offspring
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

    transposed = jnp.transpose(arr, (1, 0) + tuple(range(2, arr.ndim)))
    expected = transposed.reshape((-1,) + transposed.shape[2:])

    assert jnp.allclose(offspring_wrapper.genes.values, expected)
