import inspect

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("tensorneat")
import tensorneat.algorithm as algo
import tensorneat.genome as geno
import tensorneat.problem as prob

from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome


def test_tensorneat_upstream_signatures():
    """Verify that TensorNEAT upstream classes adhere to expected method signatures.

    Acts as an early canary alert for breaking upstream changes.
    """
    # BaseAlgorithm signatures
    ask_sig = inspect.signature(algo.BaseAlgorithm.ask)
    assert "state" in ask_sig.parameters

    tell_sig = inspect.signature(algo.BaseAlgorithm.tell)
    assert "state" in tell_sig.parameters
    assert "fitness" in tell_sig.parameters

    transform_sig = inspect.signature(algo.BaseAlgorithm.transform)
    assert "state" in transform_sig.parameters

    forward_sig = inspect.signature(algo.BaseAlgorithm.forward)
    assert "state" in forward_sig.parameters
    assert "inputs" in forward_sig.parameters

    # BaseProblem signatures
    eval_sig = inspect.signature(prob.BaseProblem.evaluate)
    assert "state" in eval_sig.parameters
    assert "randkey" in eval_sig.parameters

    # BaseGenome signatures
    init_sig = inspect.signature(geno.BaseGenome.initialize)
    assert "state" in init_sig.parameters
    assert "randkey" in init_sig.parameters


def test_tensorneat_genome_pytree_roundtrip():
    """Verify lossless PyTree round-trip for TensorNeatGenome."""
    nodes = jax.random.normal(jax.random.PRNGKey(1), (15, 4))
    conns = jax.random.normal(jax.random.PRNGKey(2), (30, 5))
    genome = TensorNeatGenome(values=(nodes, conns))

    leaves, treedef = jax.tree_util.tree_flatten(genome)
    assert len(leaves) == 2
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, TensorNeatGenome)
    assert jnp.array_equal(genome.values[0], restored.values[0])
    assert jnp.array_equal(genome.values[1], restored.values[1])


def test_tensorneat_genome_jit_and_vmap():
    """Verify JIT and VMAP transformations on TensorNeatGenome."""

    # JIT transformation
    @jax.jit
    def scale_genome(g: TensorNeatGenome, factor: float) -> TensorNeatGenome:
        return TensorNeatGenome(values=(g.values[0] * factor, g.values[1] * factor))

    nodes = jnp.ones((10, 4))
    conns = jnp.ones((20, 5)) * 2.0
    g = TensorNeatGenome(values=(nodes, conns))

    scaled = scale_genome(g, 3.5)
    assert jnp.allclose(scaled.values[0], nodes * 3.5)
    assert jnp.allclose(scaled.values[1], conns * 3.5)

    # VMAP transformation
    batch_size = 6
    batch_nodes = jax.random.normal(jax.random.PRNGKey(10), (batch_size, 10, 4))
    batch_conns = jax.random.normal(jax.random.PRNGKey(11), (batch_size, 20, 5))
    batch_genome = TensorNeatGenome(values=(batch_nodes, batch_conns))

    @jax.vmap
    def compute_energy(single_g: TensorNeatGenome) -> jnp.ndarray:
        return jnp.sum(jnp.square(single_g.values[0])) + jnp.sum(jnp.square(single_g.values[1]))

    energies = compute_energy(batch_genome)
    assert energies.shape == (batch_size,)
    for i in range(batch_size):
        expected_energy = jnp.sum(jnp.square(batch_nodes[i])) + jnp.sum(jnp.square(batch_conns[i]))
        assert jnp.isclose(energies[i], expected_energy)
