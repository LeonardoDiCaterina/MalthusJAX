import pytest
from malthusjax.composer.strategies.core import GeneticStrategy, EvoSAXStrategy, MapElitesStrategy, QDAXStrategy

def test_genetic_strategy():
    strategy = GeneticStrategy(
        selection="tournament:size=3",
        crossover="sbx",
        mutation="polynomial"
    )
    assert strategy.selection == "tournament:size=3"
    assert strategy.crossover == "sbx"
    assert strategy.mutation == "polynomial"

    strategy_default = GeneticStrategy()
    assert strategy_default.selection is None
    assert strategy_default.crossover is None
    assert strategy_default.mutation is None

def test_map_elites_strategy():
    strategy = MapElitesStrategy(emitter="mixing")
    assert strategy.emitter == "mixing"
    
    strategy_default = MapElitesStrategy()
    assert strategy_default.emitter is None

def test_evosax_strategy():
    strategy = EvoSAXStrategy(algorithm_name="SimpleGA", algorithm_kwargs={"popsize": 100})
    assert strategy.algorithm_name == "SimpleGA"
    assert strategy.algorithm_kwargs == {"popsize": 100}

    strategy_no_kwargs = EvoSAXStrategy(algorithm_name="CMA_ES")
    assert strategy_no_kwargs.algorithm_name == "CMA_ES"
    assert strategy_no_kwargs.algorithm_kwargs == {}

def test_qdax_strategy():
    strategy = QDAXStrategy(
        strategy_cls="MAPElites",
        emitter="mixing",
        metrics_function="dummy_metrics",
        centroids="dummy_centroids",
        init_variables="dummy_init",
        algorithm_kwargs={"param": "value"}
    )
    assert strategy.strategy_cls == "MAPElites"
    assert strategy.emitter == "mixing"
    assert strategy.metrics_function == "dummy_metrics"
    assert strategy.centroids == "dummy_centroids"
    assert strategy.init_variables == "dummy_init"
    assert strategy.algorithm_kwargs == {"param": "value"}
