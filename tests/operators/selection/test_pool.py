import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@pytest.fixture
def setup_elite_pop():
    key = jr.PRNGKey(42)
    config = RealGenomeConfig(shape=(5,), bounds=(-10.0, 10.0), dtype=jnp.float32)
    # Create 10 individuals with linear fitness 0-9
    pop = RealPopulation.init_random(key, config, 10)
    fitness = jnp.arange(10.0)  # Indices 8 and 9 are the best
    pop = pop.replace(fitness=fitness)
    return pop, key


class TestElitePoolSelection:
    def test_parameter_k_boundary(self, setup_elite_pop):
        """Verifies that only the top K indices are ever selected."""
        pop, key = setup_elite_pop
        elite_k = 2
        num_selections = 100
        sel = ElitePoolSelection(num_selections=num_selections, elite_k=elite_k)

        parent_idx, elite_idx = sel(key, pop)
        # With elite_k=2 and fitness 0-9, only indices 8 and 9 should be picked
        unique_indices = jnp.unique(parent_idx)
        assert jnp.all(jnp.isin(unique_indices, jnp.array([8, 9])))
        # n_elites defaults to 0 → empty elite_idx
        assert elite_idx.shape == (0,)

    def test_jit_and_shapes(self, setup_elite_pop):
        """Verifies XLA compatibility and output shapes."""
        pop, key = setup_elite_pop
        sel = ElitePoolSelection(num_selections=5, elite_k=3)

        @jax.jit
        def compiled_call(p, k):
            return sel(k, p)

        parent_idx, elite_idx = compiled_call(pop, key)
        assert parent_idx.shape == (5,)
        assert parent_idx.dtype == jnp.int32
        assert elite_idx.shape == (0,)  # n_elites=0 default

    def test_fused_elite_extraction(self, setup_elite_pop):
        """Verifies fused argpartition returns correct parent AND elite indices."""
        pop, key = setup_elite_pop
        # elite_k=3 (parent pool), n_elites=2 (preservation)
        sel = ElitePoolSelection(num_selections=50, elite_k=3).set_n_elites(2)

        parent_idx, elite_idx = sel(key, pop)
        assert parent_idx.shape == (50,)
        assert elite_idx.shape == (2,)
        # Elites must be the top-2 fitness individuals (indices 8 and 9)
        assert jnp.all(jnp.isin(elite_idx, jnp.array([8, 9])))
        # Parents must come from top-3 pool (indices 7, 8, 9)
        assert jnp.all(jnp.isin(jnp.unique(parent_idx), jnp.array([7, 8, 9])))

    def test_fused_same_k(self, setup_elite_pop):
        """When elite_k == n_elites, no secondary sort needed."""
        pop, key = setup_elite_pop
        sel = ElitePoolSelection(num_selections=20, elite_k=3).set_n_elites(3)

        parent_idx, elite_idx = sel(key, pop)
        assert parent_idx.shape == (20,)
        assert elite_idx.shape == (3,)
        # Both parent pool and elites are the top-3
        assert jnp.all(jnp.isin(elite_idx, jnp.array([7, 8, 9])))

    def test_resource_mapper_integration(self):
        """Tests the set_input_length hook used by the ResourceMapper."""
        sel = ElitePoolSelection(num_selections=10)
        reconfigured = sel.set_input_length(100)
        assert reconfigured.input_length == 100
        assert isinstance(reconfigured, ElitePoolSelection)

    def test_set_n_elites(self):
        """Tests the set_n_elites hook used by the engine."""
        sel = ElitePoolSelection(num_selections=10, elite_k=5)
        reconfigured = sel.set_n_elites(3)
        assert reconfigured.n_elites == 3
        assert isinstance(reconfigured, ElitePoolSelection)
