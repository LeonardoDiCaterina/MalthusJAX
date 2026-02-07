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


def test_gaussian_mutation_noise_distribution():
    """Verifies that Gaussian noise has correct mean≈0 and std≈mutation_strength."""
    import numpy as np

    config = RealGenomeConfig(shape=(50,), bounds=(-10.0, 10.0))
    sigma = 0.3
    mutation = GaussianMutation(mutation_rate=1.0, mutation_strength=sigma)

    # Generate many deltas to validate distribution
    deltas = []
    for i in range(100):
        key = jax.random.PRNGKey(i)
        keys = jax.random.split(key, mutation.num_keys_per_atomic_operation)
        noise = mutation._generate_noise(keys, config)
        deltas.append(noise)

    all_deltas = jnp.stack(deltas)
    empirical_mean = float(jnp.mean(all_deltas))
    empirical_std = float(jnp.std(all_deltas))

    # Mean should be very close to 0
    assert abs(empirical_mean) < 0.05, f"Mean={empirical_mean}, expected ≈0"
    # Std should be close to sigma (allow 15% tolerance for sampling variance)
    assert 0.85 * sigma < empirical_std < 1.15 * sigma, (
        f"Std={empirical_std}, expected ≈{sigma}"
    )


def test_gaussian_mutation_strength_calibration():
    """Higher mutation_strength should produce larger deltas."""
    config = RealGenomeConfig(shape=(20,), bounds=(-5.0, 5.0))
    parent_vals = jnp.zeros((1, 20))
    genes = RealGenome(values=parent_vals)
    population = RealPopulation(genes=genes, fitness=jnp.zeros(1), config=config)

    sigmas = [0.1, 0.5, 1.0]
    deltas_by_sigma = []

    for sigma in sigmas:
        mutation = GaussianMutation(mutation_rate=1.0, mutation_strength=sigma).set_input_length(len(population))
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, mutation.num_keys((1,)))
        print(f"Testing sigma={sigma} with {len(keys)} keys")
        offspring = mutation(keys, population, config)

        delta = jnp.linalg.norm(offspring.values - parent_vals)
        deltas_by_sigma.append(float(delta))

    # Verify monotonic increase: delta(0.1) < delta(0.5) < delta(1.0)
    assert deltas_by_sigma[0] < deltas_by_sigma[1] < deltas_by_sigma[2], (
        f"Deltas should increase with sigma: got {deltas_by_sigma}"
    )


def test_gaussian_mutation_large_population_scaling():
    """Verify mutation works correctly on large populations."""
    config = RealGenomeConfig(shape=(100,), bounds=(-1.0, 1.0))
    pop_size = 500
    key = jax.random.PRNGKey(999)

    population = RealPopulation.init_random(key, config, pop_size)

    mutation = GaussianMutation(
        mutation_rate=0.5, mutation_strength=0.1, num_offspring=1
    ).set_input_length(pop_size)

    k_mut = jax.random.split(key, mutation.num_keys((pop_size,)))
    offspring = mutation(k_mut, population, config)

    assert len(offspring) == pop_size
    assert offspring.values.shape == population.values.shape
    # Some changes should exist (stochastic)
    assert not jnp.allclose(offspring.values, population.values)
