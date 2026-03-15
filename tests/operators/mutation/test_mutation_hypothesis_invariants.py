"""Property-based tests for mutation operators using Hypothesis.

These tests validate invariants that must hold for ANY valid input,
not just specific example cases. They use Hypothesis to generate random
inputs and check that core properties are maintained.
"""

import jax.numpy as jnp
import jax.random as jar
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.mutation.real import GaussianMutation

# ============================================================================
# GAUSSIAN MUTATION INVARIANTS (Real Genome)
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=50),
    mutation_rate=st.floats(min_value=0.0, max_value=1.0),
    mutation_strength=st.floats(min_value=0.01, max_value=2.0),
)
@settings(max_examples=200, deadline=None)
def test_gaussian_mutation_respects_bounds(
    pop_size: int, genome_dim: int, mutation_rate: float, mutation_strength: float
) -> None:
    """INVARIANT: Mutation respects genome bounds for all parameter combinations.

    For any population within [-5, 5] bounds and any mutation parameters,
    all offspring must remain within bounds.

    This is a critical safety invariant - mutation should never escape the
    solution space defined by bounds.
    """
    bounds = (-5.0, 5.0)
    config = RealGenomeConfig(shape=(genome_dim,), bounds=bounds)

    # Generate random population within bounds
    rng = jar.PRNGKey(42)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    # Apply mutation
    mutation = GaussianMutation(
        mutation_rate=mutation_rate, mutation_strength=mutation_strength, num_offspring=1, clip=True
    ).set_input_length(pop_size)

    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    # Core invariant: all offspring within bounds
    assert offspring.values.shape[0] == pop_size, (
        f"Expected {pop_size} offspring, got {offspring.values.shape[0]}"
    )

    assert jnp.all(offspring.values >= bounds[0]), (
        f"Mutation produced values below lower bound: {jnp.min(offspring.values)} < {bounds[0]}"
    )

    assert jnp.all(offspring.values <= bounds[1]), (
        f"Mutation produced values above upper bound: {jnp.max(offspring.values)} > {bounds[1]}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_gaussian_mutation_zero_rate_is_identity(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: mutation_rate=0 produces no changes (identity operation).

    When mutation_rate is 0, the mutation should have no effect on genomes.
    This tests the edge case where no mutation should occur.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(99)
    population = RealPopulation.init_random(rng, config, size=pop_size)
    original_values = jnp.array(population.values)

    mutation = GaussianMutation(mutation_rate=0.0, mutation_strength=0.5).set_input_length(pop_size)
    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    # With rate=0, offspring should be identical to parent
    assert jnp.allclose(offspring.values, original_values), (
        "Mutation with rate=0 should be identity operation"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_gaussian_mutation_preserves_population_size(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Mutation output has same cardinality as input.

    With num_offspring=1 (1:1 replacement), output population size equals input.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(77)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    mutation = GaussianMutation(
        mutation_rate=0.5, mutation_strength=0.3, num_offspring=1
    ).set_input_length(pop_size)

    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    assert offspring.values.shape[0] == pop_size, (
        f"Expected {pop_size} offspring, got {offspring.values.shape[0]}"
    )


@pytest.mark.invariant
@given(
    mutation_strength=st.floats(min_value=0.05, max_value=1.5),
)
@settings(max_examples=50, deadline=None)
def test_gaussian_mutation_noise_distribution_properties(mutation_strength: float) -> None:
    """INVARIANT: Gaussian noise mean ≈ 0, std ≈ mutation_strength.

    When mutation_rate=1.0, every allele is mutated. The noise distribution
    should match the intended strength parameter.
    """
    config = RealGenomeConfig(shape=(100,), bounds=(-10.0, 10.0))
    mutation = GaussianMutation(
        mutation_rate=1.0, mutation_strength=mutation_strength, num_offspring=1
    )

    # Generate many samples to validate distribution
    noise_samples = []
    for i in range(50):
        rng = jar.PRNGKey(i)
        keys = jar.split(rng, mutation.num_keys_per_atomic_operation)
        noise = mutation._generate_noise(keys, config)
        noise_samples.append(noise)

    all_noise = jnp.concatenate(noise_samples, axis=0)
    empirical_mean = float(jnp.mean(all_noise))
    empirical_std = float(jnp.std(all_noise))

    # Mean should be very close to 0 (Gaussian centered at 0)
    assert abs(empirical_mean) < 0.1, f"Mean={empirical_mean}, expected ≈0"

    # Std should be close to mutation_strength (allow 20% tolerance for sampling variance)
    assert 0.8 * mutation_strength < empirical_std < 1.2 * mutation_strength, (
        f"Std={empirical_std}, expected ≈{mutation_strength}"
    )


@pytest.mark.invariant
@given(
    mutation_strength1=st.floats(min_value=0.05, max_value=0.5),
    mutation_strength2=st.floats(min_value=0.5, max_value=2.0),
)
@settings(max_examples=30, deadline=None)
def test_gaussian_mutation_strength_amplitude(
    mutation_strength1: float, mutation_strength2: float
) -> None:
    """INVARIANT: Higher mutation_strength produces larger deltas.

    Given mutation_rate=1.0 (all alleles mutated), higher strength parameter
    should produce larger average absolute changes.
    """
    assume(mutation_strength2 > mutation_strength1)  # Ensure distinct values

    config = RealGenomeConfig(shape=(50,), bounds=(-5.0, 5.0))
    parent_vals = jnp.zeros((1, 50))
    genes = RealGenome(values=parent_vals)
    population = RealPopulation(genes=genes, fitness=jnp.zeros(1), config=config)

    # Test with strength1
    mutation1 = GaussianMutation(
        mutation_rate=1.0,
        mutation_strength=mutation_strength1,
        num_offspring=1,
        clip=False,  # Don't clip so we see full magnitude
    ).set_input_length(1)

    keys1 = jar.split(jar.PRNGKey(1), mutation1.num_keys((1,)))
    offspring1 = mutation1(keys1, population, config)
    delta1 = jnp.mean(jnp.abs(offspring1.values - population.values))

    # Test with strength2
    mutation2 = GaussianMutation(
        mutation_rate=1.0, mutation_strength=mutation_strength2, num_offspring=1, clip=False
    ).set_input_length(1)

    keys2 = jar.split(jar.PRNGKey(2), mutation2.num_keys((1,)))
    offspring2 = mutation2(keys2, population, config)
    delta2 = jnp.mean(jnp.abs(offspring2.values - population.values))

    # Higher strength should produce larger deltas on average
    assert delta2 > delta1 * 0.9, (
        f"Strength {mutation_strength2} produced delta {delta2}, "
        f"but smaller strength {mutation_strength1} produced {delta1}"
    )


# ============================================================================
# BIT FLIP MUTATION INVARIANTS (Binary Genome)
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_length=st.integers(min_value=5, max_value=50),
    mutation_rate=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=150, deadline=None)
def test_bitflip_mutation_produces_binary_values(
    pop_size: int, genome_length: int, mutation_rate: float
) -> None:
    """INVARIANT: BitFlip mutation output contains only binary values (0 or 1).

    Binary genome values must stay binary - mutation cannot produce
    fractional or out-of-range values.
    """
    config = BinaryGenomeConfig(length=genome_length)
    rng = jar.PRNGKey(42)
    population = BinaryPopulation.init_random(rng, config, size=pop_size)

    mutation = BitFlipMutation(mutation_rate=mutation_rate, num_offspring=1).set_input_length(
        pop_size
    )

    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    # All values must be 0 or 1
    unique_values = jnp.unique(offspring.values)
    assert jnp.all(jnp.isin(unique_values, jnp.array([0, 1]))), (
        f"BitFlip produced non-binary values: {unique_values}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_length=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_bitflip_mutation_zero_rate_is_identity(pop_size: int, genome_length: int) -> None:
    """INVARIANT: mutation_rate=0 produces no bit flips (identity operation).

    With rate=0, all bits should remain unchanged.
    """
    config = BinaryGenomeConfig(length=genome_length)
    rng = jar.PRNGKey(123)
    population = BinaryPopulation.init_random(rng, config, size=pop_size)
    original_values = jnp.array(population.values)

    mutation = BitFlipMutation(mutation_rate=0.0).set_input_length(pop_size)
    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    assert jnp.array_equal(offspring.values, original_values), (
        "BitFlip with rate=0 should not change any bits"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_length=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_bitflip_mutation_preserves_population_size(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Output population size equals input population size."""
    config = BinaryGenomeConfig(length=genome_length)
    rng = jar.PRNGKey(456)
    population = BinaryPopulation.init_random(rng, config, size=pop_size)

    mutation = BitFlipMutation(mutation_rate=0.5, num_offspring=1).set_input_length(pop_size)
    keys = jar.split(rng, mutation.num_keys((pop_size,)))
    offspring = mutation(keys, population, config)

    assert offspring.values.shape[0] == pop_size, (
        f"Expected {pop_size} offspring, got {offspring.values.shape[0]}"
    )


@pytest.mark.invariant
@given(
    genome_length=st.integers(min_value=20, max_value=50),
)
@settings(max_examples=50, deadline=None)
def test_bitflip_mutation_rate_flip_count(genome_length: int) -> None:
    """INVARIANT: With high mutation_rate, expect ≈ rate * genome_length flips per individual.

    For a population of 1, with mutation_rate=1.0 and genome_length bits,
    we expect roughly genome_length flips (every bit flips with probability 1.0).
    """
    config = BinaryGenomeConfig(length=genome_length)
    rng = jar.PRNGKey(789)

    # Create a single individual with all zeros
    genes = BinaryGenome(values=jnp.zeros((1, genome_length), dtype=jnp.int32))
    population = BinaryPopulation(genes=genes, fitness=jnp.zeros(1), config=config)

    # With mutation_rate=1.0, every bit flips
    mutation = BitFlipMutation(mutation_rate=1.0, num_offspring=1).set_input_length(1)
    keys = jar.split(rng, mutation.num_keys((1,)))
    offspring = mutation(keys, population, config)

    # Count flips: since parent is all 0, all 1s in offspring are flips
    num_flips = int(jnp.sum(offspring.values[0]))

    # With rate=1.0, expect all bits to flip
    assert num_flips == genome_length, (
        f"With mutation_rate=1.0, expected {genome_length} flips, got {num_flips}"
    )
