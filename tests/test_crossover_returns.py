import jax
import jax.random as jr
import jax.numpy as jnp

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.crossover.binary import UniformCrossover as BinaryUniformCrossover
from malthusjax.operators.crossover.real import UniformCrossover as RealUniformCrossover


def test_binary_single_pair_returns_batched_genome():
    key = jr.PRNGKey(0)
    cfg = BinaryGenomeConfig(shape=(10,), dtype=jnp.bool_)

    p1 = BinaryGenome.random_init(jr.PRNGKey(1), cfg)
    p2 = BinaryGenome.random_init(jr.PRNGKey(2), cfg)

    op = BinaryUniformCrossover(crossover_rate=0.5, num_offspring=3)

    out = op(jr.PRNGKey(3), p1, p2, cfg)

    # Should return a batched BinaryGenome
    assert isinstance(out, BinaryGenome)
    assert out.values.shape[0] == 3
    assert out.values.shape[1] == cfg.shape[0]


def test_real_single_pair_returns_batched_genome_and_vmap():
    key = jr.PRNGKey(10)
    cfg = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))

    p1 = RealGenome.random_init(jr.PRNGKey(11), cfg)
    p2 = RealGenome.random_init(jr.PRNGKey(12), cfg)

    op = RealUniformCrossover(crossover_rate=0.5)  # num_offspring defaults to 1

    out = op(jr.PRNGKey(13), p1, p2, cfg)

    assert isinstance(out, RealGenome)
    assert out.values.shape[0] == 1
    assert out.values.shape[1] == cfg.shape[0]

    # Test vmap of jit(op) returns a batched RealGenome with leading axis num_trials
    jit_op = jax.jit(op)
    keys = jr.split(jr.PRNGKey(20), 4)
    results = jax.vmap(lambda k: jit_op(k, p1, p2, cfg))(keys)

    # results should be a RealGenome dataclass with values shape (num_trials, num_offspring, dims)
    assert hasattr(results, "values")
    assert results.values.shape[0] == 4
    assert results.values.shape[1] == 1
    assert results.values.shape[2] == cfg.shape[0]