import jax
import jax.numpy as jnp

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation

# No GaussianMutationConfig import needed
from malthusjax.operators.mutation.real import GaussianMutation


def test_gaussian_mutation_flow(rng_key):
    """Verifies the full 3-tier execution flow with RealGenomeConfig."""
    pop_size = 5
    genome_len = 10
    # The config passed to the mutation is the genome config
    config = RealGenomeConfig(shape=(genome_len,), bounds=(-5.0, 5.0))

    population = RealPopulation.init_random(rng_key, config, size=pop_size)

    # Parameters are set on the operator itself
    mutation = GaussianMutation(
        mutation_rate=1.0, mutation_strength=0.5, num_offspring=1
    ).set_input_length(pop_size)

    k_mutation = jax.random.split(rng_key, mutation.num_keys((pop_size,)))

    # Tier 3 Call
    offspring_pop = mutation(k_mutation, population, config)

    assert len(offspring_pop) == pop_size
    assert not jnp.allclose(population.values, offspring_pop.values)


def test_gaussian_mutation_masked_arithmetic():
    """Verifies that mutation_rate=0.0 results in zero delta."""
    vals = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    genes = RealGenome(values=vals)
    population = RealPopulation(genes=genes, fitness=jnp.zeros(2), config=None)

    # Set rate to 0.0 on the operator
    mutation = GaussianMutation(mutation_rate=0.0).set_input_length(2)
    keys = jax.random.split(jax.random.PRNGKey(0), mutation.num_keys((2,)))

    config = RealGenomeConfig(shape=(2,))
    offspring = mutation(keys, population, config)

    # Delta should be zero because mask_val is 0.0
    assert jnp.all(offspring.values == population.values)
