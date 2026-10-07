import jax.numpy as jnp
from malthus_gp.evaluators.cartesian_gp_evaluator import (
    CartesianGPEvaluator,
    CartesianGPEvaluatorConfig,
)
from malthus_gp.genomes.cartesian_genome import CartesianGenome
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter


def test_evaluator_mse_loss():
    # Model computes x0 + x1
    ops = jnp.array([0], dtype=jnp.int32)
    args = jnp.array([[0, 1]], dtype=jnp.int32)
    out_nodes = jnp.array([2], dtype=jnp.int32)
    genome = CartesianGenome(ops=ops, args=args, out_nodes=out_nodes)

    interpreter = CartesianInterpreter(num_inputs=2, num_outputs=1)
    config = CartesianGPEvaluatorConfig(loss_function="mse", maximize=False)

    # Data: x0 + x1 = [3, 5], actual y = [3, 7] -> error = [0, -2] -> MSE = (0 + 4)/2 = 2.0
    X = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    y = jnp.array([[3.0], [7.0]])

    evaluator = CartesianGPEvaluator(config=config, data=(X, y), interpreter=interpreter)
    loss = evaluator.evaluate(genome)
    assert jnp.isclose(loss, 2.0)
