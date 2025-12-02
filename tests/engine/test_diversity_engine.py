import pytest
import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.engine.diversity_engine import DiversityAwareEngine
from malthusjax.engine.genetic_engine import GeneticEngineParams
import pytest
import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.engine.diversity_engine import DiversityAwareEngine
from malthusjax.engine.genetic_engine import GeneticEngineParams

from tests.conftest import get_batch_shape

@pytest.fixture
def diversity_engine():
    genome_config = BinaryGenomeConfig(length=5)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True), data=None)
    
    return DiversityAwareEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(num_selections=10, tournament_size=2),
        crossover=UniformCrossover(crossover_rate=0.8),
        mutation=BitFlipMutation(mutation_rate=0.1),
        diversity_weight=0.5
    )

def test_crowding_calculation(diversity_engine):
    """Test crowding distance logic."""
    genes = jnp.array([
        [0, 0, 0, 0, 0], # A
        [0, 0, 0, 0, 0], # B
        [1, 1, 1, 1, 1]  # C
    ])
    fitness = jnp.array([1.0, 1.0, 5.0])
    config = BinaryGenomeConfig(length=5)
    
    # WRAP ARRAY IN GENOME OBJECT!
    genome_obj = BinaryGenome(bits=genes)
    pop = BinaryPopulation(genes=genome_obj, fitness=fitness, config=config)
    
    crowding = diversity_engine._compute_crowding_scores(pop)
    
    assert crowding[2] > crowding[0]
    assert crowding[2] > crowding[1]

def test_diversity_parent_selection(diversity_engine):
    """Test parent selection override."""
    params = GeneticEngineParams(pop_size=10, num_generations=5, elitism=1)
    key = jar.PRNGKey(42)
    
    state = diversity_engine.init_state(key, params)
    parents = diversity_engine._select_parents(key, state, params)
    
    assert len(parents) == params.pop_size
    # Check shape of the bits array
    assert get_batch_shape(parents.genes)[0] == params.pop_size 

def test_diversity_elite_selection(diversity_engine):
    """Test elite selection override."""
    params = GeneticEngineParams(pop_size=10, num_generations=5, elitism=4)
    key = jar.PRNGKey(99)
    state = diversity_engine.init_state(key, params)
    
    elites = diversity_engine._select_elites(key, state, params)
    
    # Use helper
    assert get_batch_shape(elites)[0] == 4
    # Check length (2nd dim)
    assert get_batch_shape(elites)[1] == 5