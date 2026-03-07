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
from malthusjax.core.fitness.real_evaluators import SphereConfig, SphereEvaluator
from malthusjax.operators.selection import ElitePoolSelection
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxGaussianWrapper
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams


# We used to parameterize over several impls, but some JAX versions
# (notably the one installed in CI) raise a ``ValueError`` when asked to
# construct a key with an unexpected implementation string.  The only
# requirement for this regression test is that we obtain **any** new-style
# key; so we try the default first and fall back to a few known aliases.
# If none of them yield a typed key we simply skip the test.

PRNG_CANDIDATES = [None, PRNGImpl.THREEFRY, PRNGImpl.PHILOX, PRNGImpl.RBG]

def test_evosax_wrappers_with_typed_keys():
    """Ensure wrappers run under an engine when the PRNG key is "typed".

    Build a minimal engine (pop_size=4, one generation) using the Evosax
    adapters for crossover and mutation.  The test is skipped if we cannot
    obtain a new-style key from the environment.
    """

    key = None
    for impl in PRNG_CANDIDATES:
        try:
            key = create_key(0, impl=impl)
        except Exception:
            continue
        if is_new_style_key(key):
            break
    if key is None or not is_new_style_key(key):
        pytest.skip("unable to generate a new-style PRNG key; skipping")

    # small problem setup
    genome_config = RealGenomeConfig(shape=(2,), bounds=(-1.0, 1.0))
    evaluator = SphereEvaluator(SphereConfig(maximize=False))
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
    # output is a GeneticGenerationOutput dataclass; just verify attr exists
    assert hasattr(output, "best_fitness")
