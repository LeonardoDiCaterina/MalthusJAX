from typing import Any

import jax
import optax
from flax import struct
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.genome.neural_cartesian import NeuralCartesianGenomeConfig
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig

from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators import BaseMutation
from malthusjax.operators.base import BaseCrossover
from malthusjax.operators.selection.tournament import TournamentSelection


@struct.dataclass
class DummyCrossover(BaseCrossover[Any, Any]):
    @property
    def num_keys_per_atomic_operation(self) -> int: return 1
    def _generate_noise(self, keys, config, generation=0): return keys[0]
    def _recombine_one(self, p1, p2, noise_data, config, **kwargs): return p1

@struct.dataclass
class DummyMutation(BaseMutation[Any, Any]):
    @property
    def num_keys_per_atomic_operation(self) -> int: return 1
    def _generate_noise(self, keys, config, generation=0): return keys[0]
    def _mutate_one(self, genome, noise_data, config, **kwargs): return genome

def test_dmep_engine_step():
    """Verify that the GeneticEngine successfully jits and steps the dMEP engine."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Dataset
    X = jax.random.normal(k1, (50, 2))
    y = jax.random.normal(k2, (50,))

    # Configs
    gc = NeuralPrefixGenomeConfig(length=10, num_inputs=2, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=2, length=10, batch_size=8, loss_function="mse")

    base_evaluator = NeuralPrefixEvaluator(config=ec, data=(X, y))
    optimizer = optax.adam(learning_rate=0.1)
    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_evaluator, optimizer=optimizer, epochs=2)

    # Engine Config
    from lsp.operators.neural_mutation import ArchitectureMutation, HybridMutation, WeightMutation

    mut_op = HybridMutation(
        arch_mutation=ArchitectureMutation(op_rate=0.1, arg_rate=0.1),
        weight_mutation=WeightMutation(mutation_rate=0.1, sigma=0.1)
    )

    engine_config = GeneticEngineParams(pop_size=10, elitism=2, num_generations=5)

    selection = TournamentSelection(num_selections=8, tournament_size=2)

    engine = GeneticEngine(
        engine_params=engine_config,
        genome_config=gc,
        evaluator=lsmf,
        selection=selection,
        crossover=DummyCrossover(),
        mutation=mut_op
    )

    # Test Init
    state = engine.init_state(k3)
    assert state.population.genes.weights.shape == (10, 10, 2) # Pop, L, max_arity

    # Test Step (ensure it JIT compiles)
    jitted_step = jax.jit(engine.step)

    # Step 1
    new_state, _ = jitted_step(state)
    assert new_state.generation == 1

    # Step 2
    new_state, _ = jitted_step(new_state)
    assert new_state.generation == 2

def test_dcgpann_engine_step():
    """Verify that the GeneticEngine successfully jits and steps the dCGPANN engine."""
    key = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(key, 3)

    X = jax.random.normal(k1, (50, 3))
    y = jax.random.normal(k2, (50,))

    gc = NeuralCartesianGenomeConfig(num_inputs=3, num_outputs=1, num_rows=2, num_cols=2, num_ops=5, levels_back=2, max_arity=2)
    ec = NeuralCartesianEvaluatorConfig(genome_config=gc, batch_size=5, loss_function="mse")

    base_evaluator = NeuralCartesianEvaluator(config=ec, data=(X, y))
    optimizer = optax.adam(learning_rate=0.01)
    lsmf = LSMFEvaluator(config=ec, data=(X, y), base_evaluator=base_evaluator, optimizer=optimizer, epochs=1)

    from lsp.operators.neural_mutation import ArchitectureMutation, HybridMutation, WeightMutation

    HybridMutation(
        arch_mutation=ArchitectureMutation(op_rate=0.2, arg_rate=0.2),
        weight_mutation=WeightMutation(mutation_rate=0.2, sigma=0.1)
    )

    engine_config = GeneticEngineParams(pop_size=4, elitism=1, num_generations=5)

    selection = TournamentSelection(num_selections=3, tournament_size=2)


    engine = GeneticEngine(
        engine_params=engine_config,
        genome_config=gc,
        evaluator=lsmf,
        selection=selection,
        crossover=DummyCrossover(),
        mutation=DummyMutation()
    )

    state = engine.init_state(k3)
    jitted_step = jax.jit(engine.step)
    new_state, _ = jitted_step(state)
    assert new_state.generation == 1

def test_engine_batch_mismatch():
    """Verify evaluator gracefully handles datasets not perfectly divisible by batch size."""
    key = jax.random.PRNGKey(999)
    k1, k2, k3 = jax.random.split(key, 3)

    # Dataset size 25
    X = jax.random.normal(k1, (25, 2))
    y = jax.random.normal(k2, (25,))

    # Batch size 8 (25 / 8 = 3.125 batches)
    gc = NeuralPrefixGenomeConfig(length=10, num_inputs=2, num_ops=5, max_arity=2, mep_output_strategy="dynamic")
    ec = NeuralPrefixEvaluatorConfig(num_inputs=2, length=10, batch_size=8, loss_function="mse")

    base_evaluator = NeuralPrefixEvaluator(config=ec, data=(X, y))

    engine_config = GeneticEngineParams(pop_size=4, elitism=1, num_generations=5)
    selection = TournamentSelection(num_selections=3, tournament_size=2)

    engine = GeneticEngine(
        engine_params=engine_config,
        genome_config=gc,
        evaluator=base_evaluator,
        selection=selection,
        crossover=DummyCrossover(),
        mutation=DummyMutation()
    )

    state = engine.init_state(k3)
    jitted_step = jax.jit(engine.step)

    # Should not crash on boundary batches
    new_state, _ = jitted_step(state)
    assert new_state.generation == 1
