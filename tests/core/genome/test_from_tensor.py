import jax
import jax.numpy as jnp

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig


def test_real_from_tensor_basic():
    arr = jnp.arange(12.0).reshape((2, 6))
    g = RealGenome.from_tensor(arr)
    assert isinstance(g, RealGenome)
    assert jnp.allclose(g.values, arr)


def test_real_from_tensor_jit():
    arr = jnp.ones((3, 4), dtype=jnp.float32)

    @jax.jit
    def build(x):
        g = RealGenome.from_tensor(x)
        return g.values

    out = build(arr)
    assert jnp.allclose(out, arr)


def test_real_from_tensor_with_config_ignored():
    arr = jnp.zeros((1, 6))
    cfg = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
    g = RealGenome.from_tensor(arr, cfg)
    assert jnp.allclose(g.values, arr)
