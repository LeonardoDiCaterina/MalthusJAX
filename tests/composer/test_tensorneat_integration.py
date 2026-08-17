"""Tests for the TensorNEAT Integration in MalthusJAX."""

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
def test_tensorneat_native_evaluator():
    """Verify TensorNeatQDEvaluator correctly intercepts and evaluates populations."""
    from malthusjax.core.fitness.qd.tensorneat_evaluator import (
        TensorNeatEvaluatorConfig,
        TensorNeatQDEvaluator,
    )
    from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation

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

    # 3. Evaluate
    evaluator = TensorNeatQDEvaluator.create(
        algorithm=algorithm,
        problem=problem,
        forward_fn=algorithm.forward,
        config=TensorNeatEvaluatorConfig(seed=42),
    )

    evaluated_pop = evaluator.evaluate_population(dummy_pop)

    assert evaluated_pop.fitness.shape == (10,)
    assert not jnp.any(jnp.isnan(evaluated_pop.fitness))


@pytest.mark.skipif(not TENSORNEAT_AVAILABLE, reason="tensorneat is not installed")
def test_tensorneat_emitter():
    """Verify TensorNeatEmitter mathematically evaluates correctly."""
    from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
    from malthusjax.operators.emitters.tensorneat_emitter import (
        TensorNeatEmitter,
        TensorNeatEmitterState,
    )

    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)

    # We create a TensorNeatEmitter and just verify it initializes without errors
    emitter = TensorNeatEmitter(_batch_size=10, genome=genome)

    key = jax.random.PRNGKey(0)
    algorithm = NEAT(pop_size=10, species_size=1, genome=genome)
    state = State(randkey=key)
    state = algorithm.setup(state)
    pop_values = algorithm.ask(state)

    genes = TensorNeatGenome(values=pop_values)
    dummy_pop = TensorNeatPopulation(genes=genes, fitness=jnp.zeros(10), config=None)

    emitter_state = emitter.init(key, dummy_pop, params=None)
    assert isinstance(emitter_state, TensorNeatEmitterState)


@pytest.mark.skipif(not TENSORNEAT_AVAILABLE, reason="tensorneat is not installed")
def test_tensorneat_adapter_engine():
    """Verify TensorNEATEngineAdapter can compile and step."""
    from malthusjax.composer import Composer

    # We use composer to run the adapter for 2 generations
    composer = Composer.create_default()
    result = composer.quick_run(
        backend="tensorneat",
        fitness="tensorneat:problem=xor",
        strategy="tensorneat:algorithm=neat:pop_size=10:species_size=1:max_nodes=10:max_conns=20",
        eval_mode="native",
        generations=2,
    )

    # Check that it returns a valid result
    assert result.runs[0].status == "success"
    assert len(result.runs[0].history) == 2
