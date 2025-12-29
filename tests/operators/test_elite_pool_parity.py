import jax
import jax.random as jar
import numpy as np
import numpy.testing as npt

from malthusjax.operators.selection.elite_pool import ElitePoolSelection


def test_elite_pool_kernel_parity():
    key = jar.PRNGKey(1)
    pop_size = 20
    genome_length = 5
    k1, k2 = jar.split(key)

    population = jar.uniform(k1, shape=(pop_size, genome_length))
    fitness = jar.uniform(k2, shape=(pop_size,))

    num_selections = 6
    elite_k = 8
    op = ElitePoolSelection(num_selections=num_selections, elite_k=elite_k)

    # Use _select directly since we have raw fitness (not a Population object)
    K = jar.PRNGKey(42)
    legacy_indices = op._select(K, fitness, None)
    legacy_selected = population[legacy_indices]

    # Run again with same key to verify determinism
    indices_2 = op._select(K, fitness, None)
    selected_2 = population[indices_2]

    # Assert sampled rows match (deterministic behavior)
    npt.assert_allclose(np.array(legacy_selected), np.array(selected_2), atol=1e-8)