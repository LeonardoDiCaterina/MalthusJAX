import jax
import jax.random as jar
import numpy as np
import numpy.testing as npt

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.crossover.real import UniformCrossover


def test_uniform_crossover_kernel_parity_single_pair():
    """Parity test for a single parent pair to ensure RNG-aligned outputs.

    For a single pair, legacy `_cross_one(key, p1, p2, config)` and
    `apply_kernel(key, (p1_arr, p2_arr), op)` should produce identical offspring
    if passed the same PRNG key and shapes.
    """
    K = jar.PRNGKey(123)

    config = RealGenomeConfig(length=6, bounds=(-1.0, 1.0))
    k1, k2 = jar.split(K)
    p1 = RealGenome.random_init(k1, config)
    p2 = RealGenome.random_init(k2, config)

    op = UniformCrossover(crossover_rate=0.5)

    legacy_child = op._cross_one(K, p1, p2, config)

    # kernel expects (p1_array, p2_array) as batched arrays: use batch dim =1
    p1_arr = p1.values[None, ...]
    p2_arr = p2.values[None, ...]

    kernel_out = op.apply_kernel(K, (p1_arr, p2_arr), op)

    # kernel returns batched output (1, length)
    npt.assert_allclose(np.array(legacy_child.values), np.array(kernel_out[0]), atol=1e-6)
