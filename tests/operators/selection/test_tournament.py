import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.selection.tournament import TournamentSelection


@pytest.fixture
def setup_tournament_pop():
    """Create a population where index 3 is clearly the 'Alpha' (best fitness)."""
    key = jr.PRNGKey(42)
    config = RealGenomeConfig(shape=(5,), bounds=(-10.0, 10.0), dtype=jnp.float32)
    pop_size = 10
    pop = RealPopulation.init_random(key, config, pop_size)
    # Fitness: Index 3 is best (100.0), others are low.
    fitness = jnp.array([1.0, 2.0, 1.5, 100.0, 0.5, 1.2, 0.8, 2.1, 1.1, 0.9])
    pop = pop.replace(fitness=fitness)
    return pop, key


class TestTournamentSelection:
    """Rigorous tests for TournamentSelection operator."""

    def test_basic_functionality(self, setup_tournament_pop):
        """Verifies output shapes and index bounds."""
        pop, key = setup_tournament_pop
        num_selections = 4
        sel = TournamentSelection(num_selections=num_selections, tournament_size=3)

        # Test keys as provided by ResourceMapper (often sliced)
        k1, _ = jr.split(key)
        indices = sel._select(k1, pop.fitness)

        assert indices.shape == (num_selections,)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < len(pop))

    def test_jit_stability(self, setup_tournament_pop):
        """Ensures the operator compiles and remains deterministic under JIT."""
        pop, key = setup_tournament_pop
        sel = TournamentSelection(num_selections=5, tournament_size=2)
        k1, _ = jr.split(key)

        @jax.jit
        def compiled_sel(f, k):
            return sel._select(k, f)

        indices_jit = compiled_sel(pop.fitness, k1)
        indices_raw = sel._select(k1, pop.fitness)

        np.testing.assert_array_equal(indices_jit, indices_raw)

    def test_selection_pressure(self, setup_tournament_pop):
        """
        PROOFS: Larger tournament_size must increase the probability
        of selecting the best individual.
        """
        pop, key = setup_tournament_pop
        num_trials = 500
        best_idx = 3  # The Alpha in our fixture

        # Case A: Low Pressure (Tournament Size = 1, should be Random Selection)
        sel_low = TournamentSelection(num_selections=num_trials, tournament_size=1)
        idx_low = sel_low._select(key, pop.fitness)
        count_low = jnp.sum(idx_low == best_idx)

        # Case B: High Pressure (Tournament Size = 5)
        sel_high = TournamentSelection(num_selections=num_trials, tournament_size=5)
        idx_high = sel_high._select(key, pop.fitness)
        count_high = jnp.sum(idx_high == best_idx)

        # Statistical check: High pressure should pick the best significantly more often
        assert count_high > count_low
        print(f"Picks of Best (T=1): {count_low} | Picks of Best (T=5): {count_high}")

    def test_resource_mapping_compatibility(self):
        """Verifies the operator satisfies the BaseSelection contract for ResourceMapper."""
        sel = TournamentSelection(num_selections=10, tournament_size=3)

        # Test the method currently causing AttributeErrors in your engine
        reconfigured = sel.set_input_length(100)
        assert reconfigured.input_length == 100
        assert isinstance(reconfigured, TournamentSelection)

        # Verify RNG budget calculation
        assert sel.num_keys((100,)) == sel.num_keys_per_atomic_operation

    def test_key_shape_resilience(self, setup_tournament_pop):
        """Ensures logic handles both (2,) and (1, 2) key shapes from the Engine."""
        pop, key = setup_tournament_pop
        sel = TournamentSelection(num_selections=2)

        # Test raw key (2,)
        k_raw = key
        res_raw = sel._select(k_raw, pop.fitness)

        # Test engine-sliced key (1, 2)
        k_sliced = key.reshape(1, 2)
        res_sliced = sel._select(k_sliced, pop.fitness)

        assert res_raw.shape == (2,)
        assert res_sliced.shape == (2,)
