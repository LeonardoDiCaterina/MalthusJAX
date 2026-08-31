import pytest

from malthusjax.composer.engine_factory import GeneticEngineAdapter
from malthusjax.composer.factory import (
    build_evosax_engine,
    build_map_elites_engine,
    build_qdax_engine,
    build_real_engine,
    build_tensorneat_engine,
    has_real_operators,
)
from malthusjax.composer.strategies.core import (
    GeneticStrategy,
    MapElitesStrategy,
    QDAXStrategy,
    TensorNEATStrategy,
)
from malthusjax.engine.genetic_fastengine import GeneticEngine


def test_has_real_operators():
    assert has_real_operators(None, None, None, None, None) is False
    assert has_real_operators("real", None, None, None, None) is True
    assert has_real_operators(None, "sphere", None, None, None) is True


def test_build_real_engine_defaults():
    strategy = GeneticStrategy(selection=None, crossover=None, mutation=None)
    adapter = build_real_engine(strategy, fitness="sphere:dim=5")

    assert isinstance(adapter, GeneticEngineAdapter)
    assert isinstance(adapter.genetic_engine, GeneticEngine)


def test_build_real_engine_custom_operators():
    strategy = GeneticStrategy(
        selection="tournament:num_selections=4",
        crossover="blend:alpha=0.2",
        mutation="gaussian:mutation_rate=0.05",
    )
    adapter = build_real_engine(strategy, fitness="sphere:dim=5", pop_size=10, generations=2)
    assert isinstance(adapter, GeneticEngineAdapter)
    assert adapter.genetic_engine.engine_params.pop_size == 10


def test_build_evosax_engine_basic():
    pytest.importorskip("evosax")
    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec="sphere",
        pop_size=10,
        generations=5,
        num_dims=3,
        bounds=(-5.0, 5.0),
        maximize=False,
    )
    # The return object is EvosaxEngineAdapter, but we just verify it didn't crash and has basic attributes.
    assert adapter.pop_size == 10
    assert adapter.num_generations == 5
    assert adapter.num_dims == 3
    assert adapter.bounds == (-5.0, 5.0)


def test_build_evosax_engine_with_bbob():
    pytest.importorskip("evosax")
    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        fitness_spec="bbob:fn=1",
        pop_size=10,
        generations=2,
        num_dims=2,
        bounds=(-5.0, 5.0),
        maximize=False,
    )
    assert adapter.num_dims == 2


def test_build_qdax_engine_basic():
    pytest.importorskip("qdax")
    strategy = QDAXStrategy(
        strategy_cls="MAPElites",
        emitter="mixing",
        mutation_sigma=0.1,
        num_descriptors=2,
        num_centroids=10,
    )
    adapter = build_qdax_engine(
        strategy=strategy,
        fitness_spec="sphere",
        pop_size=16,
        generations=2,
        genome_length=5,
        bounds=(-1.0, 1.0),
        maximize=True,
        history_metrics=None,
    )
    # QDaxEngineAdapter
    assert adapter.pop_size == 16
    assert adapter.generations == 2


def test_build_tensorneat_engine_basic():
    pytest.importorskip("tensorneat")
    strategy = TensorNEATStrategy(
        algorithm_name="NEAT",
        genome_name="Default",
        problem_name="XOR",
        num_inputs=2,
        num_outputs=1,
    )
    adapter = build_tensorneat_engine(
        strategy=strategy,
        fitness_spec=None,
        pop_size=10,
        generations=2,
        maximize=True,
        history_metrics=None,
    )
    # TensorNeatEngineAdapter
    assert adapter.pop_size == 10
    assert adapter.generations == 2


def test_build_map_elites_engine_basic():
    strategy = MapElitesStrategy(
        emitter="mixing", mutation_sigma=0.1, num_descriptors=2, num_centroids=10
    )
    # This requires qdax internally for centroids
    pytest.importorskip("qdax")
    adapter = build_map_elites_engine(
        strategy=strategy,
        fitness_spec="sphere:dim=5",
        pop_size=10,
        generations=2,
        maximize=False,
        history_metrics=None,
        genome_length=5,
    )
    from malthusjax.composer.adapters.map_elites_adapter import MapElitesEngineAdapter

    assert isinstance(adapter, MapElitesEngineAdapter)
    assert adapter.pop_size == 10


def test_build_map_elites_engine_tensorneat():
    pytest.importorskip("tensorneat")
    pytest.importorskip("qdax")

    import tensorneat.algorithm
    import tensorneat.genome

    from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter

    genome = tensorneat.genome.DefaultGenome(num_inputs=2, num_outputs=1)
    alg = tensorneat.algorithm.NEAT(pop_size=10, genome=genome)

    emitter = TensorNeatEmitter(
        algorithm=alg, genome=genome, pop_size=10, mut_rate=0.5, cx_rate=0.5
    )

    strategy = MapElitesStrategy(emitter=emitter, num_descriptors=2, num_centroids=10)

    adapter = build_map_elites_engine(
        strategy=strategy,
        fitness_spec=None,
        tensorneat_problem="XOR",
        pop_size=10,
        generations=2,
        maximize=True,
        history_metrics=None,
    )

    from malthusjax.composer.adapters.map_elites_adapter import MapElitesEngineAdapter

    assert isinstance(adapter, MapElitesEngineAdapter)
