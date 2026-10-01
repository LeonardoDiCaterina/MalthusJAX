import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.neutral_selection import NeutralSelection


@pytest.fixture
def config():
    return CartesianGenomeConfig(
        num_rows=1,
        num_cols=10,
        num_inputs=2,
        num_outputs=1,
        num_ops=4,
        max_arity=2,
        levels_back=-1,
    )


def test_neutral_selection_drift_on_ties(config):
    """Verifies that offspring replaces parent when fitness ties occur."""
    pop = config.init_population(jax.random.PRNGKey(0), 5)

    # All 5 individuals have identical fitness (0.5)
    # Individual 0 is parent, individuals 1..4 are offspring
    tied_pop = pop.replace(fitness=jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]))

    selector = NeutralSelection(num_selections=4, elite_k=1, n_elites=1, maximize=False)
    key = jax.random.PRNGKey(42)
    parents, elite_idx = selector(key, tied_pop, config)

    # In minimization, modified fitness for tied offspring (higher index) is lower
    # so offspring 4 should have the lowest modified fitness and win elite_idx!
    assert elite_idx[0] == 4  # Offspring replaced the parent!
    assert parents[0] == 4


def test_neutral_selection_superior_parent_preserved(config):
    """Verifies that when parent has strictly better fitness, it is retained."""
    pop = config.init_population(jax.random.PRNGKey(0), 5)
    # Parent at index 0 has fitness 0.1, offspring have 0.5
    superior_pop = pop.replace(fitness=jnp.array([0.1, 0.5, 0.5, 0.5, 0.5]))

    selector = NeutralSelection(num_selections=4, elite_k=1, n_elites=1, maximize=False)
    key = jax.random.PRNGKey(42)
    parents, elite_idx = selector(key, superior_pop, config)

    assert elite_idx[0] == 0
    assert parents[0] == 0
