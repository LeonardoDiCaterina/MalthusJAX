import jax.numpy as jnp

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.crossover.binary import UniformCrossover as BinaryUniformCrossover
from malthusjax.operators.crossover.real import (
    UniformCrossover as RealUniformCrossover,
)
from malthusjax.operators.crossover.real import (
    UniformCrossover_injection as RealUniformCrossover_injection,
)


def test_default_num_offspring_is_one():
    """Ensure operator defaults match the expected single-offspring contract."""
    assert RealUniformCrossover().num_offspring == 1
    assert BinaryUniformCrossover().num_offspring == 1


def test_real_uniform_mask_semantics():
    """mask=False -> p1, mask=True -> p2 for fused and injection variants."""
    cfg = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0), dtype=jnp.float32)
    p1 = RealGenome(values=jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32))
    p2 = RealGenome(values=jnp.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=jnp.float32))

    op = RealUniformCrossover()

    mask0 = jnp.zeros(cfg.shape, dtype=jnp.bool_)
    off0 = op._recombine_one(p1, p2, mask0, cfg)
    assert jnp.allclose(off0.values, p1.values)

    mask1 = jnp.ones(cfg.shape, dtype=jnp.bool_)
    off1 = op._recombine_one(p1, p2, mask1, cfg)
    assert jnp.allclose(off1.values, p2.values)

    mask_mixed = jnp.array([False, True, False, True, False], dtype=jnp.bool_)
    offm = op._recombine_one(p1, p2, mask_mixed, cfg)
    expected = jnp.where(mask_mixed, p2.values, p1.values)
    assert jnp.allclose(offm.values, expected)

    # Injection-mode should follow same semantics
    op_inj = RealUniformCrossover_injection(num_offspring=1)
    off0i = op_inj._recombine_one(p1, p2, mask0, cfg)
    off1i = op_inj._recombine_one(p1, p2, mask1, cfg)
    assert jnp.allclose(off0i.values, p1.values)
    assert jnp.allclose(off1i.values, p2.values)


def test_binary_uniform_mask_semantics():
    """Binary uniform crossover follows same mask -> parent mapping."""
    cfg = BinaryGenomeConfig(shape=(6,), p=0.5)
    p1 = BinaryGenome(values=jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32))
    p2 = BinaryGenome(values=jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.int32))

    op = BinaryUniformCrossover()

    mask0 = jnp.zeros(cfg.shape, dtype=jnp.bool_)
    off0 = op._recombine_one(p1, p2, mask0, cfg)
    assert jnp.array_equal(off0.values, p1.values)

    mask1 = jnp.ones(cfg.shape, dtype=jnp.bool_)
    off1 = op._recombine_one(p1, p2, mask1, cfg)
    assert jnp.array_equal(off1.values, p2.values)

    mask_mixed = jnp.array([False, True, False, True, False, True], dtype=jnp.bool_)
    offm = op._recombine_one(p1, p2, mask_mixed, cfg)
    expected = jnp.where(mask_mixed, p2.values, p1.values)
    assert jnp.array_equal(offm.values, expected)


def test_offspring_major_flattening():
    """Regression test to ensure offspring-major flattening ordering remains stable."""
    input_length = 3
    num_offspring = 2
    gene_dim = 4

    nested = jnp.arange(input_length * num_offspring * gene_dim).reshape(
        input_length, num_offspring, gene_dim
    )

    # same transformation used by BaseCrossover: transpose then reshape
    transposed = jnp.transpose(nested, (1, 0, 2))
    flattened = transposed.reshape((-1, gene_dim))

    # expected concatenation order:
    #   - offspring 0 across all pairs
    #   - offspring 1 across all pairs
    expected = jnp.concatenate([nested[:, o, :] for o in range(num_offspring)], axis=0)

    assert jnp.array_equal(flattened, expected)
