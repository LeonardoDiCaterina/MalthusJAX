import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.operators.selection.roulette import RouletteSelection


@pytest.fixture
def setup_roulette_data():
    key = jr.PRNGKey(123)
    fitness = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 50.0, 0.1, 0.1])

    class MockPop:
        def __init__(self, f):
            self.fitness = f

    return MockPop(fitness), key


class TestRouletteSelection:
    def test_temperature_scaling(self, setup_roulette_data):
        """Verifies that temperature correctly adjusts selection pressure."""
        pop, key = setup_roulette_data
        num_trials = 1000

        # High Temperature -> Near Random (Index 5 should appear ~12.5% of time)
        sel_hot = RouletteSelection(num_selections=num_trials, temperature=100.0)
        parent_hot, _ = sel_hot(key, pop)
        hot_rate = jnp.sum(parent_hot == 5) / num_trials

        # Low Temperature -> Near Deterministic (Index 5 should appear ~100% of time)
        sel_cold = RouletteSelection(num_selections=num_trials, temperature=0.01)
        parent_cold, _ = sel_cold(key, pop)
        cold_rate = jnp.sum(parent_cold == 5) / num_trials

        assert cold_rate > hot_rate
        assert cold_rate > 0.95

    def test_gumbel_trick_toggle(self, setup_roulette_data):
        """Verifies both execution paths (Gumbel vs Categorical) produce valid results."""
        pop, key = setup_roulette_data
        pop_size = len(pop.fitness)

        # Path 1: Gumbel-Max (num_selections == pop_size)
        sel_gumbel = RouletteSelection(num_selections=pop_size, use_gumbel_trick=True)
        parent_gumbel, elite_gumbel = sel_gumbel(key, pop)

        # Path 2: Categorical (Force standard path via flag)
        sel_std = RouletteSelection(num_selections=pop_size, use_gumbel_trick=False)
        parent_std, elite_std = sel_std(key, pop)

        assert parent_gumbel.shape == (pop_size,)
        assert parent_std.shape == (pop_size,)
        # n_elites=0 by default
        assert elite_gumbel.shape == (0,)
        assert elite_std.shape == (0,)

    def test_elite_extraction(self, setup_roulette_data):
        """Verifies elite indices via default get_elite_indices path."""
        pop, key = setup_roulette_data
        sel = RouletteSelection(num_selections=5, temperature=1.0).set_n_elites(2)
        parent_idx, elite_idx = sel(key, pop)
        assert parent_idx.shape == (5,)
        assert elite_idx.shape == (2,)
        # Top-2 should include index 5 (fitness 50.0)
        assert 5 in elite_idx

    def test_large_population_safety(self):
        """Tests that the operator can be reconfigured for large populations."""
        sel = RouletteSelection(num_selections=10000, use_gumbel_trick=False)
        assert sel.use_gumbel_trick is False
