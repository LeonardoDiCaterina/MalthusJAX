"""Tests for MOEngine integrating MOPopulation, BaseMOEvaluator, and Emitters."""

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

# 1. Create a dummy configuration
@struct.dataclass
class DummyMOConfig(BaseEvaluatorConfig):
    maximize: bool = struct.field(pytree_node=False, default=True)

# 2. Implement the BaseMOEvaluator
@struct.dataclass
class DummyMOEvaluator(BaseMOEvaluator[BinaryGenome, DummyMOConfig, None]):
    """
    A simple evaluator that returns 2 objectives.
    Objective 1: Number of 1s in the first half of the genome.
    Objective 2: Number of 1s in the second half of the genome.
    """
    def evaluate(self, genome: BinaryGenome) -> jnp.ndarray:
        mid = genome.values.shape[-1] // 2
        obj1 = jnp.sum(genome.values[:mid])
        obj2 = jnp.sum(genome.values[mid:])
        return jnp.stack([obj1, obj2], axis=-1)

def test_mo_engine_integration():
    """
    Test a full end-to-end integration of MOEngine.
    """
    key = jax.random.PRNGKey(42)
    pop_size = 20
    genome_length = 10
    num_generations = 5
    
    # Setup config and initialize base population
    genome_config = BinaryGenomeConfig(length=genome_length)
    k_init, k_engine = jax.random.split(key)
    initial_pop = genome_config.init_population(k_init, pop_size)
    
    # Setup evaluator
    eval_config = DummyMOConfig(maximize=True)
    evaluator = DummyMOEvaluator(config=eval_config, data=None)
    
    # Setup emitter
    crossover = UniformCrossover(crossover_rate=1.0)
    mutation = BitFlipMutation(mutation_rate=0.1)
    emitter = MixingEmitter(
        emitter_a=GeneticMutationEmitter(
            mutation=mutation,
            genome_config=genome_config,
            _batch_size=pop_size // 2
        ),
        emitter_b=GeneticCrossoverEmitter(
            crossover=crossover,
            genome_config=genome_config,
            _batch_size=pop_size - (pop_size // 2)
        )
    )
    
    # Setup engine
    engine_params = MOEngineParams(
        pop_size=pop_size,
        num_generations=num_generations
    )
    engine = MOEngine(
        emitter=emitter,
        evaluator=evaluator,
        engine_params=engine_params
    )
    
    # 1. Initialize State
    state = engine.init_state(k_engine, initial_pop)
    
    # Verify initial state has correctly computed MO metrics
    assert state.population.fitness.shape == (pop_size, 2)
    assert state.population.pareto_rank.shape == (pop_size,)
    assert state.population.crowding_distance.shape == (pop_size,)
    
    # 2. Run Scan
    final_state, metrics, _ = engine.run(state)
    
    # Assert metrics were recorded for each generation
    assert metrics.num_pareto_optimal.shape == (num_generations,)
    assert metrics.best_fitness.shape == (num_generations,)
    
    # Assert survival mechanism preserved exact population size
    assert final_state.population.fitness.shape == (pop_size, 2)
    assert final_state.population.genes.values.shape == (pop_size, genome_length)
    
    # Verify the Pareto front improved or stayed the same
    # The number of optimal individuals should generally increase or stabilize
    assert jnp.all(metrics.num_pareto_optimal > 0)
