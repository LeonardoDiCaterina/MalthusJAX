import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jar

from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig


def test_gaussian_kernel_parity_with_legacy():
    """Parity test: legacy `_mutate_one` (with mutation_rate=1.0) vs `apply_kernel`.

    Strategy:
    - Use fixed master key K and split it once: k1, k2 = split(K).
    - Legacy `_mutate_one` will internally split(K) to k1', k2' and use k2' for normal.
      With deterministic split, k2' == k2.
    - Call legacy with K and kernel with k2, compare resulting arrays.
    """

    key = jar.PRNGKey(0)

    # Create config and genome
    config = RealGenomeConfig(length=8, bounds=(-1.0, 1.0))
    gkey, init_key = jar.split(key)
    genome = RealGenome.random_init(init_key, config)

    # Create operator with mutation_rate=1.0 so mask always applies
    op = GaussianMutation(mutation_rate=1.0, mutation_strength=0.05)

    # Legacy path: pass master key K
    K = jar.PRNGKey(42)

    # For kernel, we must pass the same key that legacy uses for the normal draw.
    # Legacy does: k1, k2 = split(K); noise = normal(k2, shape)
    k1, k2 = jar.split(K)

    legacy_mutated = op._mutate_one(K, genome, config)
    # kernel takes raw array data and returns mutated array
    kernel_out = op.apply_kernel(k2, genome.values, op)

    # legacy_mutated is RealGenome; compare values
    np.testing.assert_allclose(np.array(legacy_mutated.values), np.array(kernel_out), atol=1e-6)
