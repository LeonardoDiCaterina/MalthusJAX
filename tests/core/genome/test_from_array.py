"""Tests for BasePopulation.from_array class method.

Verifies that populations can be constructed from raw JAX arrays with
arbitrary population-axis placement.
"""

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.binary_genome import (
    BinaryGenome,
    BinaryGenomeConfig,
    BinaryPopulation,
)
from malthusjax.core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
    CategoricalPopulation,
)
from malthusjax.core.genome.real_genome import (
    RealGenome,
    RealGenomeConfig,
    RealPopulation,
)


class TestRealPopulationFromArray:
    """from_array on RealPopulation."""

    def test_axis0_default(self):
        """axis=0 (default): leading dim is population."""
        arr = jnp.arange(20.0).reshape((4, 5))
        cfg = RealGenomeConfig(shape=(5,), bounds=(-10.0, 10.0))
        pop = RealPopulation.from_array(arr, cfg, RealGenome)

        assert len(pop) == 4
        assert pop.genes.values.shape == (4, 5)
        assert jnp.allclose(pop.genes.values, arr)
        assert pop.fitness.shape == (4,)
        assert jnp.all(pop.fitness == -jnp.inf)

    def test_axis1(self):
        """axis=1: pop dim is in the middle → each genome has shape (x, z)."""
        arr = jax.random.uniform(jax.random.PRNGKey(0), (3, 6, 4))
        cfg = RealGenomeConfig(shape=(3, 4), bounds=(-5.0, 5.0))
        pop = RealPopulation.from_array(arr, cfg, RealGenome, axis=1)

        assert len(pop) == 6
        assert pop.genes.values.shape == (6, 3, 4)
        expected = jnp.moveaxis(arr, 1, 0)
        assert jnp.allclose(pop.genes.values, expected)

    def test_axis_last(self):
        """axis=-1: trailing dim is population."""
        arr = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 8))
        cfg = RealGenomeConfig(shape=(2, 3))
        pop = RealPopulation.from_array(arr, cfg, RealGenome, axis=-1)

        assert len(pop) == 8
        assert pop.genes.values.shape == (8, 2, 3)

    def test_1d_genome(self):
        """Simple 2-D array → axis=0 gives 1-D genomes."""
        arr = jnp.ones((10, 7))
        cfg = RealGenomeConfig(shape=(7,))
        pop = RealPopulation.from_array(arr, cfg, RealGenome)

        assert len(pop) == 10
        assert pop.genes.values.shape == (10, 7)

    @pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="float64 not enabled (JAX_ENABLE_X64=1 required)",
    )
    def test_preserves_dtype_f64(self):
        """Output preserves float64 when x64 mode is on."""
        arr = jnp.zeros((3, 4), dtype=jnp.float64)
        cfg = RealGenomeConfig(shape=(4,), dtype=jnp.float64)
        pop = RealPopulation.from_array(arr, cfg, RealGenome)
        assert pop.genes.values.dtype == jnp.float64

    def test_preserves_dtype_f32(self):
        """Output preserves float32 dtype of the input array."""
        arr = jnp.zeros((3, 4), dtype=jnp.float32)
        cfg = RealGenomeConfig(shape=(4,), dtype=jnp.float32)
        pop = RealPopulation.from_array(arr, cfg, RealGenome)
        assert pop.genes.values.dtype == jnp.float32


class TestBinaryPopulationFromArray:
    """from_array on BinaryPopulation."""

    def test_axis0(self):
        bits = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=jnp.int32)
        cfg = BinaryGenomeConfig(shape=(3,))
        pop = BinaryPopulation.from_array(bits, cfg, BinaryGenome)

        assert len(pop) == 3
        assert pop.genes.values.shape == (3, 3)
        assert jnp.allclose(pop.genes.values, bits)

    def test_axis1(self):
        bits = jax.random.bernoulli(jax.random.PRNGKey(2), shape=(4, 5, 2)).astype(jnp.int32)
        cfg = BinaryGenomeConfig(shape=(4, 2))
        pop = BinaryPopulation.from_array(bits, cfg, BinaryGenome, axis=1)

        assert len(pop) == 5
        assert pop.genes.values.shape == (5, 4, 2)


class TestCategoricalPopulationFromArray:
    """from_array on CategoricalPopulation."""

    def test_axis0(self):
        cats = jax.random.randint(jax.random.PRNGKey(3), (6, 10), 0, 5)
        cfg = CategoricalGenomeConfig(num_categories=5, shape=(10,))
        pop = CategoricalPopulation.from_array(cats, cfg, CategoricalGenome)

        assert len(pop) == 6
        assert pop.genes.values.shape == (6, 10)

    def test_axis_last(self):
        cats = jax.random.randint(jax.random.PRNGKey(4), (3, 7, 12), 0, 8)
        cfg = CategoricalGenomeConfig(num_categories=8, shape=(3, 7))
        pop = CategoricalPopulation.from_array(cats, cfg, CategoricalGenome, axis=-1)

        assert len(pop) == 12
        assert pop.genes.values.shape == (12, 3, 7)


class TestFromArrayEdgeCases:
    """Edge cases and integration tests."""

    def test_single_individual(self):
        """Population of 1 still works."""
        arr = jnp.ones((1, 5))
        cfg = RealGenomeConfig(shape=(5,))
        pop = RealPopulation.from_array(arr, cfg, RealGenome)
        assert len(pop) == 1

    def test_round_trip_with_init_random(self):
        """from_array should reconstruct a population that init_random made."""
        key = jax.random.PRNGKey(42)
        cfg = RealGenomeConfig(shape=(6,), bounds=(-1.0, 1.0))
        original = RealPopulation.init_random(key, cfg, 8)

        raw = original.genes.values
        rebuilt = RealPopulation.from_array(raw, cfg, RealGenome, axis=0)

        assert jnp.allclose(rebuilt.genes.values, original.genes.values)
        assert len(rebuilt) == len(original)

    def test_jit_compatible(self):
        """from_array works inside jax.jit."""
        cfg = RealGenomeConfig(shape=(4,))

        @jax.jit
        def build(arr):
            pop = RealPopulation.from_array(arr, cfg, RealGenome)
            return pop.genes.values, pop.fitness

        arr = jnp.ones((3, 4))
        vals, fit = build(arr)
        assert vals.shape == (3, 4)
        assert fit.shape == (3,)
