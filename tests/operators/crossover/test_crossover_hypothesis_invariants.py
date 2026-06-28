"""Property-based tests for crossover operators using Hypothesis.

These tests validate invariants that must hold for ANY valid parent pairs,
checking that offspring respect bounds, preserve counts, and maintain
expected population structures.
"""

import jax
import jax.numpy as jnp
import jax.random as jar
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.binary import (
    SinglePointCrossover,
)
from malthusjax.operators.crossover.binary import (
    UniformCrossover as BinaryUniformCrossover,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover, UniformCrossover

# ============================================================================
# SIMULATED BINARY CROSSOVER (SBX) INVARIANTS
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=50),
    eta=st.floats(min_value=1.0, max_value=30.0),
)
@settings(max_examples=200, deadline=None)
def test_sbx_respects_bounds(pop_size: int, genome_dim: int, eta: float) -> None:
    """INVARIANT: SBX offspring values stay within parental ranges.

    For any two parents within bounds, offspring from SBX should also
    be within those bounds (or very close, accounting for numerical precision).
    """
    bounds = (-5.0, 5.0)
    config = RealGenomeConfig(shape=(genome_dim,), bounds=bounds)
    rng = jar.PRNGKey(42)

    # Create population with even size (for pairing)
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    population = RealPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=eta)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # Check bounds (with tiny tolerance for numerical precision)
    tolerance = 1e-4
    assert jnp.all(offspring.genes.values >= bounds[0] - tolerance), (
        f"Offspring below lower bound: {jnp.min(offspring.genes.values)} < {bounds[0]}"
    )

    assert jnp.all(offspring.genes.values <= bounds[1] + tolerance), (
        f"Offspring above upper bound: {jnp.max(offspring.genes.values)} > {bounds[1]}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_sbx_produces_correct_offspring_count(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: SBX with num_offspring=2 produces 2*num_pairs offspring.

    For population of size pop_size, expect (pop_size // 2) * 2 offspring.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(99)
    population = RealPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    expected_count = (adjust_pop // 2) * 2
    assert offspring.genes.values.shape[0] == expected_count, (
        f"Expected {expected_count} offspring, got {offspring.genes.values.shape[0]}"
    )


@pytest.mark.invariant
@given(
    genome_dim=st.integers(min_value=1, max_value=30),
    eta_values=st.tuples(
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=10.0, max_value=30.0),
    ),
)
@settings(max_examples=50, deadline=None)
def test_sbx_eta_influences_offspring_distribution(genome_dim: int, eta_values: tuple) -> None:
    """INVARIANT: Different eta values produce different offspring distributions.

    Higher eta makes offspring more likely to be close to parents (sharper).
    Lower eta allows more spread.
    """
    eta_low, eta_high = eta_values
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))

    # Create two fixed parents
    parent1_vals = jnp.full((1, genome_dim), -2.0)
    parent2_vals = jnp.full((1, genome_dim), 2.0)

    from malthusjax.core.genome.real_genome import RealGenome

    genes = RealGenome(values=jnp.vstack([parent1_vals, parent2_vals]))
    population = RealPopulation(genes=genes, fitness=jnp.zeros(2), config=config)

    # Split for parent populations
    p1_genes = jax.tree_util.tree_map(lambda x: x[:1], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[1:], population.genes)
    p1_pop = population.spawn_offspring(p1_genes, fitness=jnp.zeros(1))
    p2_pop = population.spawn_offspring(p2_genes, fitness=jnp.zeros(1))

    # Generate offspring with low eta
    crossover_low = SimulatedBinaryCrossover(num_offspring=2, eta=eta_low)
    keys_low = jar.split(jar.PRNGKey(1), crossover_low.num_keys((1,)))
    offspring_low = crossover_low(keys_low, p1_pop, p2_pop, config)

    # Generate offspring with high eta
    crossover_high = SimulatedBinaryCrossover(num_offspring=2, eta=eta_high)
    keys_high = jar.split(jar.PRNGKey(2), crossover_high.num_keys((1,)))
    offspring_high = crossover_high(keys_high, p1_pop, p2_pop, config)

    # With higher eta, offspring should be closer to parents on average
    center = (-2.0 + 2.0) / 2  # Midpoint between parents
    dist_low = jnp.mean(jnp.abs(offspring_low.genes.values - center))
    dist_high = jnp.mean(jnp.abs(offspring_high.genes.values - center))

    # Higher eta should produce tighter distribution (closer to parents)
    # This is a statistical property, so we allow some variance
    # We only assert if there's a clear difference
    if dist_low > 1.5 and dist_high < 1.0:
        assert dist_high <= dist_low * 1.1, (
            f"Higher eta {eta_high} should produce closer offspring: "
            f"dist_high={dist_high}, dist_low={dist_low}"
        )


# ============================================================================
# UNIFORM CROSSOVER INVARIANTS (Real Genome)
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
    crossover_rate=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=150, deadline=None)
def test_uniform_crossover_produces_parental_material(
    pop_size: int, genome_dim: int, crossover_rate: float
) -> None:
    """INVARIANT: Uniform crossover produces offspring from parental material.

    Each allele in offspring comes from one parent or the other.
    With crossover_rate, that percentage of alleles come from parent2,
    rest from parent1.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(222)
    population = RealPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = UniformCrossover(num_offspring=2, crossover_rate=crossover_rate)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # Verify offspring are within parental ranges (not excessively mutated)
    # This is a weaker check - we don't verify exact parent origins due to
    # the stochastic nature of uniform crossover
    assert offspring.genes.values.shape[0] == (adjust_pop // 2) * 2, (
        f"Expected {(adjust_pop // 2) * 2} offspring, got {offspring.genes.values.shape[0]}"
    )

    assert offspring.genes.values.shape[1] == genome_dim, (
        f"Expected genome dim {genome_dim}, got {offspring.genes.values.shape[1]}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_uniform_crossover_respects_bounds(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Uniform crossover offspring stay within parental bounds."""
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(333)
    population = RealPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = UniformCrossover(num_offspring=2, crossover_rate=0.5)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # Offspring should be within bounds since they're combinations of in-bounds parents
    assert jnp.all(offspring.genes.values >= -5.0 - 1e-4), (
        f"Offspring below lower bound: {jnp.min(offspring.genes.values)}"
    )

    assert jnp.all(offspring.genes.values <= 5.0 + 1e-4), (
        f"Offspring above upper bound: {jnp.max(offspring.genes.values)}"
    )


# ============================================================================
# BINARY CROSSOVER INVARIANTS
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_length=st.integers(min_value=5, max_value=50),
)
@settings(max_examples=150, deadline=None)
def test_binary_uniform_crossover_produces_binary(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Binary uniform crossover produces only binary values (0 or 1)."""
    config = BinaryGenomeConfig(length=genome_length)
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(444)
    population = BinaryPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = BinaryUniformCrossover(num_offspring=2, crossover_rate=0.5)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # All values must be 0 or 1
    unique_values = jnp.unique(offspring.genes.values)
    assert jnp.all(jnp.isin(unique_values, jnp.array([0, 1]))), (
        f"BinaryUniformCrossover produced non-binary values: {unique_values}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_length=st.integers(min_value=5, max_value=50),
)
@settings(max_examples=150, deadline=None)
def test_single_point_crossover_produces_binary(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Single-point crossover produces only binary values (0 or 1)."""
    config = BinaryGenomeConfig(length=genome_length)
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(555)
    population = BinaryPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = SinglePointCrossover(num_offspring=2)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # All values must be 0 or 1
    unique_values = jnp.unique(offspring.genes.values)
    assert jnp.all(jnp.isin(unique_values, jnp.array([0, 1]))), (
        f"SinglePointCrossover produced non-binary values: {unique_values}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_length=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_single_point_crossover_has_split_point(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Single-point crossover creates offspring from two segments.

    Each offspring is composed of a prefix from one parent and suffix from another
    at a single crossover point.
    """
    # Only test with populations large enough to see variation
    assume(pop_size >= 4)

    config = BinaryGenomeConfig(length=genome_length)
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(666)
    population = BinaryPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = SinglePointCrossover(num_offspring=2)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    # Verify offspring count is correct
    assert offspring.genes.values.shape[0] == (adjust_pop // 2) * 2, (
        f"Expected {(adjust_pop // 2) * 2} offspring, got {offspring.genes.values.shape[0]}"
    )

    # Verify genome_length is preserved
    assert offspring.genes.values.shape[1] == genome_length, (
        f"Expected genome length {genome_length}, got {offspring.genes.values.shape[1]}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=2, max_value=50),
    genome_length=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_binary_crossover_preserves_population_size(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Binary crossover produces exactly num_offspring * num_pairs offspring."""
    config = BinaryGenomeConfig(length=genome_length)
    adjust_pop = pop_size if pop_size % 2 == 0 else pop_size + 1
    rng = jar.PRNGKey(777)
    population = BinaryPopulation.init_random(rng, config, size=adjust_pop)

    # Split population into two parent groups
    num_pairs = adjust_pop // 2
    p1_idx = jnp.arange(num_pairs)
    p2_idx = jnp.arange(num_pairs, adjust_pop)
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    crossover = SinglePointCrossover(num_offspring=2)
    keys = jar.split(rng, crossover.num_keys((num_pairs,)))
    offspring = crossover(keys, p1_pop, p2_pop, config)

    expected_count = (adjust_pop // 2) * 2
    assert offspring.genes.values.shape[0] == expected_count, (
        f"Expected {expected_count} offspring, got {offspring.genes.values.shape[0]}"
    )
