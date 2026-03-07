"""Regression tests for the Evosax compatibility wrappers.

These wrappers previously mishandled the ``typed_keys`` flag, causing
``jax.random`` primitives to receive a shape ``(1,)`` key instead of a
scalar.  The symptom was an error message along the lines of
``uniform accepts a single key, but was given a key array of shape (1,) !=
()`` when the engine was driven with new-style PRNG keys.

The tests below exercise both wrappers under a tiny genetic engine with a
new-style key to ensure no exceptions are raised and that evolution
proceeds normally.
"""

from __future__ import annotations

import pytest

import jax.random as jr

from malthusjax.core.random import PRNGImpl, create_key, is_new_style_key
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.real_evaluators import SphereEvaluator
from malthusjax.operators.selection import ElitePoolSelection
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxGaussianWrapper
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams


@pytest.mark.parametrize("prng_impl", [PRNGImpl.PHILOX, PRNGImpl.RBG])
def test_evosax_wrappers_with_typed_keys(prng_impl: PRNGImpl):
    """Ensure wrappers run under an engine when the PRNG key is "typed".

    The test builds a minimal engine (pop_size=4, one generation) using the
    Evosax adapters for crossover and mutation.  If the constructed key is
    legacy, we skip the test since the bug only manifested for new-style
    keys.
    """
    key = create_key(0, impl=prng_impl)
    if not is_new_style_key(key):
        pytest.skip(f"PRNG impl {prng_impl} produced legacy key; skipping")

    # small problem setup
    genome_config = RealGenomeConfig(shape=(2,), bounds=(-1.0, 1.0))
    evaluator = SphereEvaluator(dim=2)
    selection = ElitePoolSelection(num_selections=4, elite_k=2)
    crossover = EvosaxUniformCrossoverWrapper(num_offspring=1, crossover_rate=0.5)
    mutation = EvosaxGaussianWrapper(num_offspring=1, mutation_strength=0.1)

    engine = GeneticEngine(
        engine_params=GeneticEngineParams(pop_size=4, num_generations=1),
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        enable_progress_bar=False,
    )

    # initialization should succeed and produce a state
    state = engine.init_state(key)
    assert state is not None

    # perform a single step; errors would surface here
    state, output = engine.step(state)
    assert state.generation == 1
    assert "best_fitness" in output
