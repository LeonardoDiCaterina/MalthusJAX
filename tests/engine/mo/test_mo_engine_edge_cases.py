import pytest
import jax
import jax.numpy as jnp

from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams

from malthusjax.operators.emitters.mixing import MixingEmitter
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter, GeneticCrossoverEmitter
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.core.fitness.base import BaseEvaluatorConfig

@struct.dataclass
class DummyMOConfig(BaseEvaluatorConfig):
    maximize: bool = struct.field(pytree_node=False, default=True)

@struct.dataclass
class DummyMOEvaluator(BaseMOEvaluator[BinaryGenome, DummyMOConfig, None]):
    def evaluate(self, genome: BinaryGenome) -> jnp.ndarray:
        return jnp.stack([jnp.sum(genome.values), jnp.sum(genome.values)], axis=-1)

def test_mo_engine_jit_compilation():
    """Ensure the MOEngine cleanly JIT compiles using the run() loop."""
    key = jax.random.PRNGKey(0)
    pop_size = 10
    
    genome_config = BinaryGenomeConfig(length=5)
    evaluator = DummyMOEvaluator(config=DummyMOConfig(maximize=True), data=None)
    emitter = MixingEmitter(
        emitter_a=GeneticMutationEmitter(
            mutation=BitFlipMutation(mutation_rate=0.1),
            genome_config=genome_config,
            _batch_size=pop_size // 2
        ),
        emitter_b=GeneticCrossoverEmitter(
            crossover=UniformCrossover(crossover_rate=1.0),
            genome_config=genome_config,
            _batch_size=pop_size - (pop_size // 2)
        )
    )
    
    # Test with just 2 generations to verify scan unroll / execution under JIT
    engine_params = MOEngineParams(
        pop_size=pop_size,
        num_generations=2
    )
    engine = MOEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    k_init, k_engine = jax.random.split(key)
    initial_pop = genome_config.init_population(k_init, pop_size)
    state = engine.init_state(k_engine, initial_pop)
    
    # Call JIT explicitly on the engine's run function
    jitted_run = jax.jit(engine.run)
    final_state, metrics, _ = jitted_run(state)
    
    # If it completed without crashing, JIT constraints (static shapes, pure functions) are satisfied.
    assert metrics.generation.shape == (2,)
    assert final_state.generation == 2

def test_mo_engine_zero_generations():
    """Ensure it handles a 0-generation edge case safely."""
    key = jax.random.PRNGKey(0)
    pop_size = 10
    
    genome_config = BinaryGenomeConfig(length=5)
    evaluator = DummyMOEvaluator(config=DummyMOConfig(maximize=True), data=None)
    emitter = MixingEmitter(
        emitter_a=GeneticMutationEmitter(
            mutation=BitFlipMutation(mutation_rate=0.1),
            genome_config=genome_config,
            _batch_size=pop_size // 2
        ),
        emitter_b=GeneticCrossoverEmitter(
            crossover=UniformCrossover(crossover_rate=1.0),
            genome_config=genome_config,
            _batch_size=pop_size - (pop_size // 2)
        )
    )
    
    engine_params = MOEngineParams(
        pop_size=pop_size,
        num_generations=0
    )
    engine = MOEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    k_init, k_engine = jax.random.split(key)
    initial_pop = genome_config.init_population(k_init, pop_size)
    state = engine.init_state(k_engine, initial_pop)
    
    final_state, metrics, _ = engine.run(state)
    
    # State should be identical to generation 0
    assert final_state.generation == 0
    assert metrics.generation.shape == (0,)
