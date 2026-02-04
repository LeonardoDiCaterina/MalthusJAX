import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig


def test_binary_from_tensor_basic():
    arr = jnp.array([[0, 1, 1], [1, 0, 1]], dtype=jnp.int32)
    g = BinaryGenome.from_tensor(arr)
    assert isinstance(g, BinaryGenome)
    assert jnp.array_equal(g.values, arr)


def test_binary_from_tensor_jit():
    arr = jnp.ones((2, 4), dtype=jnp.int32)

    @jax.jit
    def build(x):
        g = BinaryGenome.from_tensor(x)
        return g.values

    out = build(arr)
    assert jnp.array_equal(out, arr)


def test_categorical_from_tensor_basic():
    arr = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    g = CategoricalGenome.from_tensor(arr)
    assert isinstance(g, CategoricalGenome)
    assert jnp.array_equal(g.values, arr)


def test_categorical_from_tensor_jit():
    arr = jnp.zeros((3, 5), dtype=jnp.int32)

    @jax.jit
    def build(x):
        g = CategoricalGenome.from_tensor(x)
        return g.values

    out = build(arr)
    assert jnp.array_equal(out, arr)


def test_linear_from_tensor_basic():
    ops = jnp.array([0, 1, 2], dtype=jnp.int32)
    args = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
    g = LinearGenome.from_tensor((ops, args))
    assert isinstance(g, LinearGenome)
    assert jnp.array_equal(g.ops, ops)
    assert jnp.array_equal(g.args, args)


def test_linear_from_tensor_jit():
    ops = jnp.arange(4, dtype=jnp.int32)
    args = jnp.tile(jnp.arange(3, dtype=jnp.int32), (4, 1))

    @jax.jit
    def build(o, a):
        g = LinearGenome.from_tensor((o, a))
        return g.ops, g.args

    out_ops, out_args = build(ops, args)
    assert jnp.array_equal(out_ops, ops)
    assert jnp.array_equal(out_args, args)
