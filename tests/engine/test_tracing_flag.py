import jax
from malthusjax.engine import genetic_fastengine as ge
from malthusjax.engine.genetic_fastengine import (
    GeneticEngineParams,
    GeneticEngine,
    enable_tracing,
    disable_tracing,
)
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.real_evaluators import SphereEvaluator
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.real import BlendCrossover
from malthusjax.operators.mutation.real import GaussianMutation


def make_engine(debug: bool) -> GeneticEngine:
    cfg = RealGenomeConfig(shape=(2,), bounds=(-1.0, 1.0))
    # SphereEvaluator takes a config with maximize flag
    eval_cfg = type('C', (), {'maximize': False})()
    evaluator = SphereEvaluator(eval_cfg)
    sel = TournamentSelection(num_selections=4, tournament_size=2)
    cross = BlendCrossover(num_offspring=2)
    mut = GaussianMutation(num_offspring=2)
    params = GeneticEngineParams(pop_size=8, num_generations=1, debug_tracing=debug)
    return GeneticEngine(
        genome_config=cfg,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        engine_params=params,
    )


def test_default_tracing_disabled():
    # flag is off globally and via default params
    disable_tracing()
    assert not ge._TRACING_ENABLED
    eng = make_engine(debug=False)
    st = eng.init_state(0)
    assert not ge._TRACING_ENABLED
    # even after a step, should remain off
    eng.step(st)
    assert not ge._TRACING_ENABLED


def test_debug_tracing_on():
    disable_tracing()
    eng = make_engine(debug=True)
    assert not ge._TRACING_ENABLED  # not yet initialized
    st = eng.init_state(0)
    assert ge._TRACING_ENABLED, "init_state should enable tracing"
    # turning it off manually also works
    disable_tracing()
    assert not ge._TRACING_ENABLED


def test_helpers_toggle():
    disable_tracing()
    assert not ge._TRACING_ENABLED
    enable_tracing()
    assert ge._TRACING_ENABLED
    disable_tracing()
    assert not ge._TRACING_ENABLED
