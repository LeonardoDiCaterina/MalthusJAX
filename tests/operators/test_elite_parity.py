import jax
import jax.random as jar
import numpy as np
import numpy.testing as npt

from malthusjax.operators.selection.truncation import Truncation


def test_truncation_kernel_parity():
    key = jar.PRNGKey(0)
    pop_size = 10
    genome_length = 4
    k1, k2 = jar.split(key)

    # create a deterministic population and fitness
    population = jar.uniform(k1, shape=(pop_size, genome_length))
    fitness = jar.uniform(k2, shape=(pop_size,))

    num_select = 3
    op = Truncation(num_selections=num_select)

    # Legacy indices
    indices = op(None, fitness)

    legacy_selected = population[indices]

    # Kernel out
    kernel_out = op.apply_kernel(None, (population, fitness), op)

    npt.assert_allclose(np.array(legacy_selected), np.array(kernel_out), atol=1e-8)
