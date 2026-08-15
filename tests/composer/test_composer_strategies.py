from unittest.mock import MagicMock, patch

from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.core import EvoSAXStrategy, QDAXStrategy


@patch("malthusjax.composer.composer.BenchmarkRunner")
@patch("malthusjax.composer.composer.Composer._build_real_engine")
def test_quick_run_with_default_malthusjax(mock_build_real_engine, mock_runner):
    composer = Composer()

    mock_engine = MagicMock()
    mock_build_real_engine.return_value = mock_engine

    # Needs fitness to trigger GeneticStrategy
    composer.quick_run(backend="malthusjax", fitness="dummy")

    assert mock_build_real_engine.called
    args, kwargs = mock_build_real_engine.call_args
    assert kwargs["strategy"] is not None


@patch("malthusjax.composer.composer.BenchmarkRunner")
@patch("malthusjax.composer.composer.Composer._build_evosax_engine")
def test_quick_run_with_evosax_backend(mock_build_evosax_engine, mock_runner):
    composer = Composer()

    mock_engine = MagicMock()
    mock_build_evosax_engine.return_value = mock_engine

    composer.quick_run(backend="evosax", evosax_strategy="CMA_ES")

    assert mock_build_evosax_engine.called
    args, kwargs = mock_build_evosax_engine.call_args
    assert kwargs["strategy_name"] == "CMA_ES"


@patch("malthusjax.composer.composer.BenchmarkRunner")
@patch("malthusjax.composer.composer.Composer._build_evosax_engine")
def test_quick_run_with_evosax_strategy_explicit(mock_build_evosax_engine, mock_runner):
    composer = Composer()

    mock_engine = MagicMock()
    mock_build_evosax_engine.return_value = mock_engine

    strategy = EvoSAXStrategy(algorithm_name="LM_MA_ES")
    composer.quick_run(strategy=strategy)

    assert mock_build_evosax_engine.called
    args, kwargs = mock_build_evosax_engine.call_args
    assert kwargs["strategy_name"] == "LM_MA_ES"


@patch("malthusjax.composer.composer.BenchmarkRunner")
@patch("malthusjax.composer.qdax_adapter.build_qdax_engine")
def test_quick_run_with_qdax_backend(mock_build_qdax_engine, mock_runner):
    composer = Composer()

    mock_engine = MagicMock()
    mock_build_qdax_engine.return_value = mock_engine

    with patch.dict("sys.modules", {"qdax.core.map_elites": MagicMock()}):
        composer.quick_run(backend="qdax", qdax_strategy="MAPElites")

        assert mock_build_qdax_engine.called


@patch("malthusjax.composer.composer.BenchmarkRunner")
@patch("malthusjax.composer.qdax_adapter.build_qdax_engine")
def test_quick_run_with_qdax_strategy_explicit(mock_build_qdax_engine, mock_runner):
    composer = Composer()

    mock_engine = MagicMock()
    mock_build_qdax_engine.return_value = mock_engine

    mock_strategy_cls = MagicMock()
    strategy = QDAXStrategy(
        strategy_cls=mock_strategy_cls,
        emitter=MagicMock(),
        metrics_function=MagicMock(),
        centroids=MagicMock(),
        init_variables=MagicMock(),
    )

    composer.quick_run(strategy=strategy)

    assert mock_build_qdax_engine.called
    args, kwargs = mock_build_qdax_engine.call_args
    assert kwargs["strategy_cls"] == mock_strategy_cls
