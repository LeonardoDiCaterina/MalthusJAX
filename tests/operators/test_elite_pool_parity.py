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

    # Legacy: pass master key K
    K = jar.PRNGKey(42)
    legacy_indices = op(K, fitness)
    legacy_selected = population[legacy_indices]

    # For kernel, we must compute sample_key = split(K)[0] to match legacy behavior
    sample_key = jax.random.split(K)[0]
    kernel_out = op.apply_kernel(sample_key, (population, fitness), op)

    # Assert sampled rows match
    npt.assert_allclose(np.array(legacy_selected), np.array(kernel_out), atol=1e-8)
