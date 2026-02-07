import jax
import jax.numpy as jnp

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig, LinearPopulation


def test_linear_genome_init(rng_key):
    """Verifies LGP initialization and topological DAG constraints."""
    config = LinearGenomeConfig(length=10, num_inputs=5, num_ops=8, max_arity=2)
    genome = LinearGenome.random_init(rng_key, config)

    assert isinstance(genome, LinearGenome)
    # Check structural shapes
    assert genome.ops.shape == (config.length,)
    assert genome.args.shape == (config.length, config.max_arity)

    # Verify topological validity: instruction i can only reference inputs or previous instructions
    # Legal indices for row i are [0, num_inputs + i - 1]
    for i in range(config.length):
        max_legal_idx = config.num_inputs + i - 1
        assert jnp.all(genome.args[i] <= max_legal_idx)
        assert jnp.all(genome.args[i] >= 0)


def test_linear_population_soa(rng_key):
    """Verifies SoA batching for structural programs (ops and args)."""
    pop_size = 8
    config = LinearGenomeConfig(length=12, num_inputs=3, num_ops=10, max_arity=2)
    population = LinearPopulation.init_random(rng_key, config, size=pop_size)

    assert isinstance(population, LinearPopulation)
    # Both ops and args should have a leading population dimension
    assert population.genes.ops.shape == (pop_size, 12)
    assert population.genes.args.shape == (pop_size, 12, 2)
    assert population.fitness.shape == (pop_size,)


def test_linear_values_property(rng_key):
    """Checks the unified .values interface for the structural genome."""
    config = LinearGenomeConfig(length=5, num_inputs=2, num_ops=4, max_arity=2)
    genome = LinearGenome.random_init(rng_key, config)

    # .values should return the (ops, args) tuple
    payload = genome.values
    assert isinstance(payload, tuple)
    assert len(payload) == 2
    assert jnp.array_equal(payload[0], genome.ops)
    assert jnp.array_equal(payload[1], genome.args)


def test_linear_autocorrect_clipping():
    """Verifies that out-of-bounds ops and illegal references are corrected."""
    config = LinearGenomeConfig(length=3, num_inputs=2, num_ops=5, max_arity=1)

    # Manually create a "broken" genome:
    # Row 0: op 10 (invalid), arg 5 (invalid, max legal is num_inputs-1 = 1)
    broken_ops = jnp.array([10, 0, 0])
    broken_args = jnp.array([[5], [0], [0]])
    genome = LinearGenome(ops=broken_ops, args=broken_args)

    corrected = genome.autocorrect(config)

    # Op should be clipped to [0, 4]
    assert corrected.ops[0] == 4
    # Arg at row 0 should be clipped to [0, 1]
    assert corrected.args[0, 0] == 1


def test_linear_distance_metrics(rng_key):
    """Tests structural Hamming distance between programs."""
    config = LinearGenomeConfig(length=5, num_inputs=2, num_ops=10, max_arity=2)
    g1 = LinearGenome.random_init(rng_key, config)

    # Create g2 as a copy of g1 with exactly one op changed
    g2_ops = g1.ops.at[0].set((g1.ops[0] + 1) % config.num_ops)
    g2 = LinearGenome(ops=g2_ops, args=g1.args)

    dist = g1.distance(g2, metric=DistanceMetric.HAMMING)
    # Hamming distance counts mismatches in both ops and args
    assert int(dist) == 1


def test_linear_render():
    """Ensures the assembly-like rendering does not crash."""
    config = LinearGenomeConfig(length=3, num_inputs=2, num_ops=2, max_arity=1)
    genome = LinearGenome(ops=jnp.array([0, 1, 0]), args=jnp.array([[0], [2], [1]]))

    # Test rendering with default op names
    output = genome.render(config)
    assert "v_0 = OP_0(x_0)" in output
    assert "v_1 = OP_1(v_0)" in output


def test_linear_jit_stability(rng_key):
    """Verifies that structural operations are JIT-compatible."""
    config = LinearGenomeConfig(length=10, num_inputs=5, num_ops=8, max_arity=2)
    genome = LinearGenome.random_init(rng_key, config)

    @jax.jit
    def structural_step(g):
        # Trigger autocorrect and distance inside JIT
        g_corr = g.autocorrect(config)
        d = g_corr.distance(g, metric=DistanceMetric.HAMMING)
        return g_corr, d

    corrected, dist = structural_step(genome)
    assert isinstance(corrected, LinearGenome)
    assert dist >= 0
