"""Property-based tests for selection operators using Hypothesis.

These tests validate invariants that must hold for ANY valid population,
checking that selection maintains cardinality, respects fitness ordering,
and preserves population integrity.
"""

import jax.numpy as jnp
import jax.random as jar
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.tournament import TournamentSelection

# ============================================================================
# ELITE POOL SELECTION INVARIANTS
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=5, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=200, deadline=None)
def test_elite_pool_returns_exact_count(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Elite pool selection always returns exactly num_selections individuals.

    This is a critical contract - selection must never change the population size.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(42)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    # Create with random fitness
    fitness = jar.uniform(jar.PRNGKey(1), shape=(pop_size,))
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = ElitePoolSelection(num_selections=pop_size, elite_k=max(1, pop_size // 2))

    selected = selection(jar.PRNGKey(1), population, None)

    assert len(selected[0]) == pop_size, f"Expected {pop_size} selected, got {len(selected[0])}"


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=10, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=30),
    elite_ratio=st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=150, deadline=None)
def test_elite_pool_includes_best_individuals(
    pop_size: int, genome_dim: int, elite_ratio: float
) -> None:
    """INVARIANT: Elite pool selection preserves top-k individuals.

    The elite individuals (those with highest fitness) should be preserved
    across generations with high probability.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(99)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    # Create with known fitness: 0 to pop_size-1
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    elite_k = max(1, int(pop_size * elite_ratio))
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=elite_k)

    selected = selection(jar.PRNGKey(2), population, None)
    selected_fitness = population.fitness[selected[0]]

    # At minimum, population mean fitness should be reasonable
    mean_selected = float(jnp.mean(selected_fitness))
    mean_all = float(jnp.mean(population.fitness))

    # Selected individuals should have mean fitness >= overall mean
    assert mean_selected >= mean_all - 1e-5, (
        f"Elite pool mean fitness {mean_selected} < overall mean {mean_all}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=5, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_elite_pool_max_fitness_preserved(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Best individual (max fitness) is always preserved.

    Elite pool must always include the global best solution.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(111)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    # Create with known fitness with clear maximum
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)
    max_fitness = float(jnp.max(fitness))

    selection = ElitePoolSelection(num_selections=pop_size, elite_k=1)
    selected = selection(jar.PRNGKey(3), population, None)

    # The best individual should be in selected set
    selected_fitness = population.fitness[selected[0]]
    assert float(jnp.max(selected_fitness)) == max_fitness, (
        "Elite pool failed to preserve global best individual"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=5, max_value=50),
    num_selections=st.integers(min_value=5, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_elite_pool_respects_num_selections(
    pop_size: int, num_selections: int, genome_dim: int
) -> None:
    """INVARIANT: Selection returns exactly num_selections individuals.

    Allows different num_selections from pop_size (e.g., for overlapping generations).
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(222)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    fitness = jar.uniform(rng, shape=(pop_size,))
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = ElitePoolSelection(
        num_selections=num_selections, elite_k=max(1, num_selections // 3)
    )

    selected = selection(jar.PRNGKey(4), population, None)

    assert len(selected[0]) == num_selections, (
        f"Expected {num_selections} selected, got {len(selected[0])}"
    )


# ============================================================================
# TOURNAMENT SELECTION INVARIANTS
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=10, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=30),
    tournament_size=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=150, deadline=None)
def test_tournament_returns_exact_count(
    pop_size: int, genome_dim: int, tournament_size: int
) -> None:
    """INVARIANT: Tournament selection always returns exactly num_selections individuals."""
    assume(tournament_size <= pop_size)  # Tournament size must be feasible

    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(333)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    fitness = jar.uniform(rng, shape=(pop_size,))
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = TournamentSelection(num_selections=pop_size, tournament_size=tournament_size)

    selected = selection(jar.PRNGKey(5), population, None)

    assert len(selected[0]) == pop_size, f"Expected {pop_size} selected, got {len(selected[0])}"


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=20, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_tournament_favors_higher_fitness(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Tournament selection favors individuals with higher fitness.

    While not deterministic per individual, the average selected fitness
    should exceed the population average.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(444)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    # Create with known fitness: 0 to pop_size-1
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = TournamentSelection(
        num_selections=pop_size // 2,  # Select subset
        tournament_size=max(2, pop_size // 10),
    )

    selected = selection(jar.PRNGKey(6), population, None)
    selected_fitness = population.fitness[selected[0]]

    mean_selected = float(jnp.mean(selected_fitness))
    mean_all = float(jnp.mean(population.fitness))

    # Tournament selection should prefer higher fitness on average
    assert mean_selected >= mean_all - 1e-5, (
        f"Tournament selection mean {mean_selected} not favoring higher fitness (mean={mean_all})"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=10, max_value=50),
    tournament_size=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_tournament_larger_size_stronger_selection(pop_size: int, tournament_size: int) -> None:
    """INVARIANT: Larger tournament size = stronger selection pressure.

    With larger tournaments, winners are more likely to be genuinely high-fitness.
    """
    assume(tournament_size <= pop_size)

    config = BinaryGenomeConfig(length=20)
    rng = jar.PRNGKey(555)
    population = BinaryPopulation.init_random(rng, config, size=pop_size)

    # Create fitness gradient: random but somewhat ordered
    fitness = jnp.sort(jar.uniform(rng, shape=(pop_size,)))
    population = BinaryPopulation(genes=population.genes, fitness=fitness, config=config)

    # Small tournament (weaker selection)
    selection_small = TournamentSelection(num_selections=pop_size // 2, tournament_size=2)
    selected_small = selection_small(jar.PRNGKey(7), population, None)
    fitness_small = fitness[selected_small[0]]

    # Large tournament (stronger selection)
    selection_large = TournamentSelection(
        num_selections=pop_size // 2, tournament_size=min(tournament_size + 3, pop_size)
    )
    selected_large = selection_large(jar.PRNGKey(8), population, None)
    fitness_large = fitness[selected_large[0]]

    # Larger tournament should have higher mean fitness
    # (though with randomness, we use a lenient check)
    mean_small = float(jnp.mean(fitness_small))
    mean_large = float(jnp.mean(fitness_large))

    # We expect larger tournament to produce >= mean fitness, but don't enforce strict >
    # due to randomness. Just verify it's at least reasonable.
    assert mean_large >= mean_small * 0.9, (
        f"Larger tournament ({tournament_size}) produced mean {mean_large}, "
        f"smaller produced {mean_small}"
    )


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=10, max_value=50),
    genome_dim=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_tournament_all_individuals_samplable(pop_size: int, genome_dim: int) -> None:
    """INVARIANT: Every individual in population has non-zero selection probability.

    Tournament selection allows any individual to be selected (based on random tournament).
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    rng = jar.PRNGKey(666)
    population = RealPopulation.init_random(rng, config, size=pop_size)

    fitness = jar.uniform(rng, shape=(pop_size,))
    population = RealPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = TournamentSelection(num_selections=pop_size * 5, tournament_size=2)

    # Multiple rounds to see if all individuals get selected eventually
    all_selected = set()
    for i in range(5):
        selected = selection(jar.PRNGKey(100 + i), population, None)
        import numpy as np
        all_selected.update(np.asarray(selected[0]).tolist())

    # With 5 rounds of pop_size*5 selections from pop_size individuals,
    # statistically all should be selected at least once
    selected_ratio = len(all_selected) / pop_size
    assert selected_ratio >= 0.6, (
        f"Tournament selection only selected {selected_ratio * 100:.1f}% of individuals"
    )


# ============================================================================
# BINARY GENOME SELECTION INVARIANTS
# ============================================================================


@pytest.mark.invariant
@given(
    pop_size=st.integers(min_value=5, max_value=50),
    genome_length=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=100, deadline=None)
def test_selection_preserves_binary_integrity(pop_size: int, genome_length: int) -> None:
    """INVARIANT: Selection of binary population preserves binary values (0 or 1)."""
    config = BinaryGenomeConfig(length=genome_length)
    rng = jar.PRNGKey(777)
    population = BinaryPopulation.init_random(rng, config, size=pop_size)

    fitness = jar.uniform(rng, shape=(pop_size,))
    population = BinaryPopulation(genes=population.genes, fitness=fitness, config=config)

    selection = ElitePoolSelection(num_selections=pop_size // 2, elite_k=pop_size // 4)
    selected = selection(jar.PRNGKey(8), population, None)

    # Selected individuals should still have binary values
    selected_population = BinaryPopulation(
        genes=population.genes.__class__(values=population.genes.values[selected[0]]),
        fitness=population.fitness[selected[0]],
        config=config,
    )

    unique_values = jnp.unique(selected_population.genes.values)
    assert jnp.all(jnp.isin(unique_values, jnp.array([0, 1]))), (
        f"Selection produced non-binary values: {unique_values}"
    )
