import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.linear import LinearGenome, LinearPopulation


def test_linear_genome_init(rng_key, linear_genome_config):
    """Verifies topological initialization where args respect row limits."""
    genome = LinearGenome.random_init(rng_key, linear_genome_config)
    assert isinstance(genome, LinearGenome)
    assert genome.ops.shape == (linear_genome_config.length,)
    assert genome.args.shape == (linear_genome_config.length, linear_genome_config.max_arity)

    # Verify topological constraints: arg indices must be < num_inputs + row_index
    for i in range(linear_genome_config.length):
        limit = linear_genome_config.num_inputs + i
        assert jnp.all(genome.args[i] < limit), f"Row {i} references index out of DAG bounds"


def test_linear_population_soa(linear_population, linear_genome_config):
    """Verifies SoA batching for matrix-based linear genomes."""
    assert isinstance(linear_population, LinearPopulation)
    # Batch dimension (10, length) and (10, length, arity)
    assert linear_population.genes.ops.shape == (10, linear_genome_config.length)
    assert linear_population.genes.args.shape == (
        10,
        linear_genome_config.length,
        linear_genome_config.max_arity,
    )


def test_linear_autocorrect_dag_repair(linear_population, linear_genome_config):
    """Verifies that autocorrect repairs invalid out-of-order references."""
    # Manually create a broken genome where row 0 references row 5 (illegal)
    # We use .at[].set() for JAX array modification
    broken_args = linear_population.genes.args.at[:, 0, :].set(100)

    # Use the Any-cast pattern to bypass Flax replace attribute errors
    from typing import Any, cast

    broken_genes = cast(Any, linear_population.genes).replace(args=broken_args)
    broken_pop = cast(Any, linear_population).replace(genes=broken_genes)

    corrected_pop = broken_pop.autocorrect(linear_genome_config)

    # After repair, row 0 args must be < num_inputs
    max_legal_idx = linear_genome_config.num_inputs - 1
    assert jnp.all(corrected_pop.genes.args[:, 0, :] <= max_legal_idx)


def test_linear_distance_jit(linear_population):
    """Verifies JIT-stable structural Hamming distance for programs."""
    g1 = linear_population[0]
    g2 = linear_population[1]

    @jax.jit
    def get_dist(a, b):
        return a.distance(b, metric=DistanceMetric.HAMMING)

    dist = get_dist(g1, g2)
    assert dist >= 0
    # Manual check: sum of mismatched ops and args
    expected = jnp.sum(g1.ops != g2.ops) + jnp.sum(g1.args != g2.args)
    assert float(dist) == pytest.approx(float(expected))


def test_linear_render_smoke(linear_population, linear_genome_config):
    """Smoke test for the non-JIT human-readable rendering."""
    # This is a CPU-side operation
    genome = linear_population[0]
    output = genome.render(linear_genome_config)
    assert "Row" in output
    assert "Expression" in output
    assert f"v_{linear_genome_config.length - 1}" in output
