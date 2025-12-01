"""
Tests for StandardGeneticEngine.
"""
import pytest
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.engine.genetic_engine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.base import AbstractEngineParams
from malthusjax.core.fitness.binary_evaluators import BinarySumConfig, BinarySumEvaluator
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import UniformCrossover  
from malthusjax.operators.mutation.binary import BitFlipMutation


@pytest.fixture
def standard_engine_components():
    """Create components for GeneticEngine."""
    # 1. Define the genome structure
    genome_config = BinaryGenomeConfig(length=10)
    
    # 2. Define the evaluator config
    eval_config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=eval_config, data=None)
    
    selection = TournamentSelection(num_selections=20, tournament_size=3)
    crossover = UniformCrossover(num_offspring=1, crossover_rate=0.5)
    mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
    
    return {
        'genome_config': genome_config,
        'evaluator': evaluator,
        'selection': selection,
        'crossover': crossover,
        'mutation': mutation
    }


def test_genetic_engine_initialization(standard_engine_components):
    """Test that GeneticEngine initializes correctly."""
    engine = GeneticEngine(**standard_engine_components)
    assert engine is not None
    assert not engine.is_compiled()


def test_genetic_engine_compilation_cache(standard_engine_components):
    """Test that compile_evolution() caches the function."""
    engine = GeneticEngine(**standard_engine_components)
    params = GeneticEngineParams(pop_size=20, num_generations=5, elitism=2)
    
    assert not engine.is_compiled(params)
    engine.compile_evolution(params)
    assert engine.is_compiled(params)


def test_genetic_engine_run_workflow(standard_engine_components):
    """Test complete GeneticEngine run workflow."""
    engine = GeneticEngine(**standard_engine_components)
    params = GeneticEngineParams(pop_size=20, num_generations=3, elitism=2)
    
    # Initialize state
    rng_key = jar.PRNGKey(42)
    initial_state = engine.init_state(rng_key, params)
    
    # This should work without AttributeError
    final_state, history, elapsed_time = engine.run(
        initial_state, 
        params, 
        time_it=True, 
        compile=True, 
        verbose=False
    )
    
    # Verify results
    assert final_state.generation == params.num_generations
    assert elapsed_time is not None
    assert elapsed_time > 0
    assert len(history.generation) == params.num_generations