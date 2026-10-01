import jax.numpy as jnp
from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenome
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)


def test_differentiable_interpreter_autodiff():
    # 1 input (index 0)
    # Node 1: tanh(2.0 * x0 + 0.5)
    # Target function S(x) = tanh(2x + 0.5)
    # Derivative dS/dx = 2 * (1 - tanh(2x + 0.5)^2) = 2 * sech^2(2x + 0.5)
    ops = jnp.array([0], dtype=jnp.int32)  # op 0 is tanh
    args = jnp.array([[0, 0]], dtype=jnp.int32)
    out_nodes = jnp.array([1], dtype=jnp.int32)
    weights = jnp.array([[2.0, 0.0]], dtype=jnp.float32)
    biases = jnp.array([0.5], dtype=jnp.float32)

    genome = DifferentiableCartesianGenome(
        ops=ops, args=args, out_nodes=out_nodes, weights=weights, biases=biases
    )
    interpreter = DifferentiableCartesianInterpreter(num_inputs=1, num_outputs=1)

    x = jnp.array([0.25], dtype=jnp.float32)
    # 2*0.25 + 0.5 = 1.0
    expected_val = jnp.tanh(1.0)
    expected_grad = 2.0 * (1.0 - jnp.tanh(1.0) ** 2)

    # 1. Test forward pass
    y = interpreter.evaluate_single(genome, x)
    assert jnp.isclose(y[0], expected_val, atol=1e-5)

    # 2. Test analytical derivative dS/dx via JAX AutoDiff
    grad_val = interpreter.grad_x(genome, x)
    assert jnp.isclose(grad_val[0], expected_grad, atol=1e-5)

    # 3. Test analytical Hessian d^2S/dx^2 via JAX AutoDiff
    # d/dx [ 2 * (1 - tanh(2x+0.5)^2) ] = 2 * (-2 * tanh * 2 * (1 - tanh^2)) = -8 * tanh(1.0) * sech^2(1.0)
    expected_hess = -8.0 * jnp.tanh(1.0) * (1.0 - jnp.tanh(1.0) ** 2)
    hess_val = interpreter.hessian_x(genome, x)
    assert jnp.isclose(hess_val[0, 0], expected_hess, atol=1e-5)

    # 4. Test formula extraction
    formula = interpreter.to_formula(genome, feature_names=["t"])
    assert "tanh" in formula
    assert "t" in formula
