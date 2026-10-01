import jax.numpy as jnp
import optax
from malthus_gp.evaluators.cartesian_gp_evaluator import (
    CartesianGPEvaluator,
    CartesianGPEvaluatorConfig,
)
from malthus_gp.evaluators.lsmf_evaluator import LSMFEvaluator, LSMFEvaluatorConfig
from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenome
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)


def test_lsmf_gradient_descent_decreases_loss():
    """Validates that LSMF inner gradient descent decreases loss on continuous parameters."""
    # Target function: y = 2.0 * x
    # Linear activation is op 3 in ACTIVATION_FUNCTIONS
    # Sub-optimal initial weight: 0.1 instead of 2.0
    ops = jnp.array([3], dtype=jnp.int32)
    args = jnp.array([[0, 0]], dtype=jnp.int32)
    out_nodes = jnp.array([1], dtype=jnp.int32)
    weights = jnp.array([[0.1, 0.0]], dtype=jnp.float32)
    biases = jnp.array([0.0], dtype=jnp.float32)

    genome = DifferentiableCartesianGenome(
        ops=ops, args=args, out_nodes=out_nodes, weights=weights, biases=biases
    )

    interpreter = DifferentiableCartesianInterpreter(num_inputs=1, num_outputs=1)
    X = jnp.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2.0 * X

    base_evaluator = CartesianGPEvaluator(
        config=CartesianGPEvaluatorConfig(loss_function="mse"),
        data=(X, y),
        interpreter=interpreter,
    )

    initial_loss = base_evaluator.evaluate(genome)

    # Wrap in LSMFEvaluator with 25 Adam steps
    lsmf_evaluator = LSMFEvaluator(
        config=LSMFEvaluatorConfig(epochs=25),
        data=base_evaluator.data,
        base_evaluator=base_evaluator,
        optimizer=optax.adam(0.1),
    )

    optimized_loss = lsmf_evaluator.evaluate(genome)

    # Gradient descent must have reduced the error significantly
    assert optimized_loss < initial_loss
    assert optimized_loss < 0.1
