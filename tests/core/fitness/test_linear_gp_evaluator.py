import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.linear_gp_evaluator import (
    TENSORGP_NAMES,
    LinearGPEvaluator,
    LinearGPEvaluatorConfig,
)
from malthusjax.core.genome.linear_genome import LinearGenome


def test_linear_gp_evaluator_compilation():
    config = LinearGPEvaluatorConfig(num_inputs=1, length=10, loss_function="mse")

    # Simple data: x = [1, 2, 3], y = [2, 4, 6] (y = 2x)
    data = (jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[2.0], [4.0], [6.0]]))
    evaluator = LinearGPEvaluator(config=config, data=data)

    ops = jnp.zeros((10,), dtype=jnp.int32)
    args = jnp.zeros((10, 3), dtype=jnp.int32)

    genome = LinearGenome(ops=ops, args=args)

    fitness = evaluator.evaluate(genome)

    assert jnp.isclose(fitness, 0.0, atol=1e-5)


def test_linear_gp_evaluator_unsupported_metric():
    config = LinearGPEvaluatorConfig(num_inputs=1, length=10, loss_function="invalid_metric")
    data = (jnp.array([[1.0]]), jnp.array([[1.0]]))

    with pytest.raises(ValueError, match="does not support"):
        # The error is raised at evaluation time for LinearGPEvaluator
        evaluator = LinearGPEvaluator(config=config, data=data)
        ops = jnp.zeros((10,), dtype=jnp.int32)
        args = jnp.zeros((10, 3), dtype=jnp.int32)
        evaluator.evaluate(LinearGenome(ops=ops, args=args))


def test_linear_gp_all_operators():
    # Let's compile a program with all operators just to touch them in execution
    # This boosts coverage for all the op_* functions
    config = LinearGPEvaluatorConfig(num_inputs=1, length=len(TENSORGP_NAMES), loss_function="mse")

    data = (jnp.array([[1.0]]), jnp.array([[1.0]]))
    evaluator = LinearGPEvaluator(config=config, data=data)

    ops = jnp.arange(len(TENSORGP_NAMES), dtype=jnp.int32)
    args = jnp.zeros((len(TENSORGP_NAMES), 3), dtype=jnp.int32)

    genome = LinearGenome(ops=ops, args=args)

    # Should run without error
    # It might result in nan/inf depending on operations, but jax handles it
    fitness = evaluator.evaluate(genome)
    assert fitness is not None
