import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig
from malthusjax.core.genome.linear import LinearGenome


def test_linear_gp_execution_logic(linear_genome_config):
    """Tests if a simple ADD program produces the expected mathematical result."""
    # Create a simple program: v0 = ADD(x0, x1)
    # OpCode 0 is ADD in your TENSORGP_FUNCTIONS
    ops = jnp.array([0])
    # args: x0 (idx 0), x1 (idx 1), dummy (idx 0)
    args = jnp.array([[0, 1, 0]])
    genome = LinearGenome(ops=ops, args=args)

    # Configure evaluator with 2 inputs and 1 instruction
    config = LinearGPEvaluatorConfig(maximize=True, num_inputs=2, length=1)
    # Dummy data for init
    evaluator = LinearGPEvaluator(config=config, data=(jnp.zeros((1, 2)), jnp.zeros(1)))

    # Test on input [10.0, 5.0]
    x_input = jnp.array([10.0, 5.0])
    outputs = evaluator.predict_one(genome, x_input)

    # v0 = 10.0 + 5.0 = 15.0
    assert float(outputs[0]) == pytest.approx(15.0)


def test_symbiotic_fitness_calculation(linear_genome_config):
    """Verifies that fitness correctly identifies the best instruction (MSE)."""
    X = jnp.array([[1.0], [2.0]])
    y = jnp.array([2.0, 4.0])  # Target function: y = 2x

    # Instruction 0: v0 = x0 + x0 (Perfect: 2x)
    # Instruction 1: v1 = x0 - x0 (Bad: 0)
    ops = jnp.array([0, 1])
    args = jnp.array([[0, 0, 0], [0, 0, 0]])
    genome = LinearGenome(ops=ops, args=args)

    config = LinearGPEvaluatorConfig(maximize=True, num_inputs=1, length=2)
    evaluator = LinearGPEvaluator(config=config, data=(X, y))

    # evaluate() should return the fitness of the BEST instruction.
    # Instruction 0 has MSE = 0. Instruction 1 has MSE = 10.
    # Since maximize=True, it should return -0.0
    fitness = evaluator.evaluate(genome)
    assert float(fitness) == pytest.approx(0.0)


def test_linear_gp_jit_stability(linear_population, linear_genome_config):
    """Ensures the complex Scan/Switch loop is stable for XLA compilation."""
    X = jnp.ones((5, linear_genome_config.num_inputs))
    y = jnp.ones(5)

    eval_config = LinearGPEvaluatorConfig(
        maximize=True,
        num_inputs=linear_genome_config.num_inputs,
        length=linear_genome_config.length,
    )

    evaluator = LinearGPEvaluator(config=eval_config, data=(X, y))

    @jax.jit
    def batched_eval(pop):
        return evaluator.evaluate_population(pop)

    result_pop = batched_eval(linear_population)
    assert result_pop.fitness.shape == (10,)
    assert not jnp.any(jnp.isnan(result_pop.fitness))
