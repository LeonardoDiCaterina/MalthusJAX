import jax.numpy as jnp
from malthus_gp.evaluators.ode_evaluator import (
    ODEResidualEvaluator,
    ODEResidualEvaluatorConfig,
)
from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenome
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)


def test_ode_residual_on_exact_solution():
    """Validates that y(t) = exp(-t) has zero residual for dy/dt = -y, y(0)=1."""
    # Target ODE: dy/dt = -y, y(0) = 1
    # Genome computes: exp(-1.0 * t + 0.0)
    # op 7 is act_exp in ACTIVATION_FUNCTIONS
    ops = jnp.array([7], dtype=jnp.int32)
    args = jnp.array([[0, 0]], dtype=jnp.int32)
    out_nodes = jnp.array([1], dtype=jnp.int32)
    weights = jnp.array([[-1.0, 0.0]], dtype=jnp.float32)
    biases = jnp.array([0.0], dtype=jnp.float32)

    genome = DifferentiableCartesianGenome(
        ops=ops, args=args, out_nodes=out_nodes, weights=weights, biases=biases
    )

    interpreter = DifferentiableCartesianInterpreter(num_inputs=1, num_outputs=1)
    config = ODEResidualEvaluatorConfig(bc_weight=10.0, maximize=False)
    t_domain = jnp.linspace(0.0, 2.0, 20)

    evaluator = ODEResidualEvaluator(
        config=config,
        data=t_domain,
        interpreter=interpreter,
        ode_fn=lambda t, y: -y,
        initial_condition=(0.0, 1.0),
    )

    loss = evaluator.evaluate(genome)
    # For y(t) = exp(-t): dy/dt = -exp(-t) = -y, so residual is identically zero!
    # y(0) = exp(0) = 1, so boundary penalty is zero!
    assert jnp.isclose(loss, 0.0, atol=1e-5)
