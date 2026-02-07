"""
Tests for PR1 Engine fixes:
- Dataclass defaults validation
- Elitism==0 edge case handling
"""

import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.fitness.binary_evaluators import BinarySumConfig, BinarySumEvaluator
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.binary import SinglePointCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def test_engine_dataclass_defaults_valid():
    """
    Ensure dataclass fields for crossover/mutation are proper struct
    fields, not function objects.
    """
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    engine_params = GeneticEngineParams(pop_size=20)

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(num_selections=20, tournament_size=3),
        crossover=SinglePointCrossover(num_offspring=2),
        mutation=BitFlipMutation(num_offspring=1, mutation_rate=0.1),
    )

    # Dataclass defaults were wrong before — ensure fields are operator instances
    assert hasattr(engine, "crossover")
    assert hasattr(engine, "mutation")
    # Provided instances should be present
    assert isinstance(engine.crossover, SinglePointCrossover)
    assert isinstance(engine.mutation, BitFlipMutation)


def test_engine_elitism_zero_runs():
    """Ensure engine runs correctly with elitism=0 without top_k errors."""
    genome_config = BinaryGenomeConfig(length=10)
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=True))
    engine_params = GeneticEngineParams(pop_size=17, elitism=0, num_generations=2)

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(num_selections=34, tournament_size=3),
        crossover=SinglePointCrossover(num_offspring=2),
        mutation=BitFlipMutation(num_offspring=1, mutation_rate=0.1),
    )

    state = engine.init_state(jar.PRNGKey(42))
    new_state, metrics = engine.step(state)

    # Verify population size is preserved
    assert new_state.population.genes.values.shape[0] == 17
    # Verify generation incremented
    assert new_state.generation == 1
    # Verify no NaN values in population (bits are int, but check anyway)
    assert not jnp.any(jnp.isnan(new_state.population.fitness))
