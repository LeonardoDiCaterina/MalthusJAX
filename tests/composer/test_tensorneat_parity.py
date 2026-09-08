"""Parity tests for TensorNEAT evaluators."""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("tensorneat")

try:
    import tensorneat  # noqa: F401
    from tensorneat.algorithm import NEAT
    from tensorneat.common import State
    from tensorneat.genome import DefaultGenome
    from tensorneat.problem import XOR

    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False


@pytest.mark.skipif(not TENSORNEAT_AVAILABLE, reason="tensorneat is not installed")
def test_tensorneat_evaluator_parity():
    """Verify the new TensorNeatEvaluator is mathematically equivalent to the legacy TensorNeatQDEvaluator."""
    from malthusjax.core.fitness.composable.base import ScalarOutput
    from malthusjax.core.fitness.composable.environments import TensorNEATProblemWrapper
    from malthusjax.core.fitness.composable.evaluators import TensorNeatEvaluator
    from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
    from malthusjax.core.fitness.qd.tensorneat_evaluator import (
        TensorNeatEvaluatorConfig,
        TensorNeatQDEvaluator,
    )
    from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
    from plugins.tensor_neat_transform import TensorNeatTransform

    problem = XOR()
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)
    algorithm = NEAT(pop_size=10, species_size=1, genome=genome)

    key = jax.random.PRNGKey(0)
    state = State(randkey=key)
    state = algorithm.setup(state)

    # 1. Ask
    pop_values = algorithm.ask(state)

    # 2. Package in MalthusJAX
    genes = TensorNeatGenome(values=pop_values)
    dummy_pop = TensorNeatPopulation(genes=genes, fitness=jnp.zeros(10), config=None)

    # 3. Evaluate Legacy
    legacy_evaluator = TensorNeatQDEvaluator.create(
        algorithm=algorithm,
        problem=problem,
        forward_fn=algorithm.forward,
        config=TensorNeatEvaluatorConfig(seed=42),
    )
    legacy_evaluated_pop = legacy_evaluator.evaluate_population(dummy_pop)

    # 4. Evaluate New Composable
    new_evaluator = TensorNeatEvaluator(
        env=TensorNEATProblemWrapper(problem=problem),
        transform=TensorNeatTransform(algorithm=algorithm),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
        forward_fn=algorithm.forward,
        seed=42,
    )
    new_evaluated_pop = new_evaluator.evaluate_population(dummy_pop)

    # 5. Assert Parity
    assert jnp.allclose(legacy_evaluated_pop.fitness, new_evaluated_pop.fitness, equal_nan=True)
    assert jnp.allclose(legacy_evaluated_pop.info["descriptors"], new_evaluated_pop.info["descriptors"])
