import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.engine.genetic_engine import GeneticEngine, GeneticEngineParams
from tests.conftest import get_batch_shape
@pytest.fixture
def standard_engine():
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True), data=None)
    
    return GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(num_selections=20, tournament_size=2),
        crossover=UniformCrossover(crossover_rate=0.8),
        mutation=BitFlipMutation(mutation_rate=0.1)
    )

@pytest.fixture
def engine_params():
    return GeneticEngineParams(pop_size=20, num_generations=5, elitism=2)

def test_component_execution(standard_engine, engine_params):
    """Test individual components."""
    key = jar.PRNGKey(42)
    state = standard_engine.init_state(key, engine_params)
    
    # 1. Test Elites
    elites = standard_engine._select_elites(key, state, engine_params)
    assert get_batch_shape(elites)[0] == engine_params.elitism

    # 2. Test Selection
    parents = standard_engine._select_parents(key, state, engine_params)
    assert len(parents) == engine_params.pop_size
    assert get_batch_shape(parents.genes)[0] == engine_params.pop_size

    # 3. Test Offspring Creation
    offspring = standard_engine._create_offspring(key, parents, state, engine_params)
    assert get_batch_shape(offspring)[0] == engine_params.pop_size

    # 4. Test Merge & Evaluate
    new_pop, fitness = standard_engine._merge_and_evaluate(key, elites, offspring, state, engine_params)
    assert len(new_pop) == engine_params.pop_size
    assert fitness.shape == (engine_params.pop_size,)

def test_hall_of_fame_update(standard_engine, engine_params):
    key = jar.PRNGKey(0)
    state = standard_engine.init_state(key, engine_params)
    
    new_pop = state.population
    fake_fitness = jnp.zeros(engine_params.pop_size).at[0].set(9999.0)
    new_pop = new_pop.replace(fitness=fake_fitness)
    
    best_genome, best_fit, stagnation = standard_engine._update_hall_of_fame(state, new_pop, engine_params)
    
    assert best_fit == 9999.0
    assert stagnation == 0 

def test_full_step_execution(standard_engine, engine_params):
    key = jar.PRNGKey(1)
    state = standard_engine.init_state(key, engine_params)
    
    next_key, next_state, metrics = standard_engine.step(key, state, engine_params)
    
    assert next_state.generation == 1
    assert metrics.best_fitness > -1