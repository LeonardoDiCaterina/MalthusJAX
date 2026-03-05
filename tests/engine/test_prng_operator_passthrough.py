"""Tests for operator-level PRNG key integrity and determinism."""

import jax
import pytest

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.random import create_key, is_new_style_key
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@pytest.fixture
def small_engine():
    params = GeneticEngineParams(pop_size=16, elitism=1, num_generations=1)
    genome_config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    bbob = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=5, maximize=False))

    engine = GeneticEngine(
        engine_params=params,
        genome_config=genome_config,
        evaluator=bbob,
        selection=ElitePoolSelection(num_selections=16, elite_k=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1),
    )
    return engine


def test_allocate_entropy_preserves_impl(prng_impl, small_engine):
    try:
        key = create_key(1, impl=prng_impl)
    except ValueError:
        pytest.skip("impl not supported by this JAX build")

    state = small_engine.init_state(key)
    k_sel, k_cross, k_mut, k_next = small_engine._allocate_entropy(state)

    # Check keys are new-style typed keys
    for k in [k_sel[0], k_cross[0], k_mut[0], k_next]:
        assert is_new_style_key(k)

def test_selection_deterministic_given_key(small_engine):
    key = create_key(99)
    state = small_engine.init_state(key)
    k_sel, k_cross, k_mut, k_next = small_engine._allocate_entropy(state)

    pop = state.population
    sel = state.operators.selection  # Use baked operator (typed_keys set by engine)

    parent1, elite1 = sel(k_sel, pop)
    parent2, elite2 = sel(k_sel, pop)
    assert jax.numpy.array_equal(parent1, parent2)
    assert jax.numpy.array_equal(elite1, elite2)


def test_crossover_mutation_keys_reshape_and_usage(small_engine):
    key = create_key(7)
    state = small_engine.init_state(key)
    k_sel, k_cross, k_mut, k_next = small_engine._allocate_entropy(state)

    # Ensure shapes match ResourceMap expectations (non-empty)
    assert k_cross.shape[0] >= 1
    assert k_mut.shape[0] >= 1

    # Run a reproduction step to ensure operators accept keys and run
    elites, selected_idx = small_engine._selection_phase(
        k_sel,
        state.population,
        state.operators,
        small_engine.engine_params,
    )

    mutants = small_engine._reproduction_phase(
        k_cross,
        k_mut,
        selected_idx,
        state.population,
        state.operators,
        state.resource_map,
    )
    assert mutants.genes is not None
