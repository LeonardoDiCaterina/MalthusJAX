"""Coverage tests for composer __main__, map_elites_adapter, and factory functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from malthusjax.composer.__main__ import main as composer_main
from malthusjax.composer.adapters.map_elites_adapter import MapElitesEngineAdapter
from malthusjax.composer.factory import (
    build_map_elites_engine,
    build_tensorneat_engine,
    resolve_tensorneat_problem,
)
from malthusjax.composer.strategies.core import MapElitesStrategy, TensorNEATStrategy


def test_composer_main():
    # 1. Less than 2 args
    with patch("sys.argv", ["composer"]):
        with pytest.raises(SystemExit) as exc_info:
            composer_main()
        assert exc_info.value.code == 1

    # 2. Config only
    mock_result = MagicMock()
    mock_result.summary.return_value = "Run Summary OK"
    with patch("sys.argv", ["composer", "config.toml"]):
        with patch(
            "malthusjax.composer.Composer.from_toml", return_value=mock_result
        ) as mock_from_toml:
            composer_main()
            mock_from_toml.assert_called_once_with("config.toml", pipelines=None)

    # 3. Config with pipelines
    with patch("sys.argv", ["composer", "config.toml", "pipe_a", "pipe_b"]):
        with patch(
            "malthusjax.composer.Composer.from_toml", return_value=mock_result
        ) as mock_from_toml:
            composer_main()
            mock_from_toml.assert_called_once_with("config.toml", pipelines=["pipe_a", "pipe_b"])


def test_map_elites_adapter_run_once_standard():
    key = jax.random.PRNGKey(0)
    mock_emitter = MagicMock()
    mock_pop = MagicMock()
    mock_pop.copy.return_value = mock_pop
    mock_emitter.genome_config.init_population.return_value = mock_pop

    mock_engine = MagicMock()
    mock_engine.emitter = mock_emitter
    mock_engine.engine_params = SimpleNamespace(num_generations=2, key_derivation="standard")

    mock_state = MagicMock()
    mock_state.best_fitness = 10.5
    mock_state.generation = 2
    mock_state.qd_score = 42.0
    mock_state.coverage = 0.8

    mock_history = SimpleNamespace(
        best_fitness=jnp.array([5.0, 10.5]),
        qd_score=jnp.array([20.0, 42.0]),
        coverage=jnp.array([0.5, 0.8]),
    )
    mock_engine.init_state.return_value = mock_state
    mock_engine.run.return_value = (mock_state, mock_history, {})

    adapter = MapElitesEngineAdapter(
        engine=mock_engine,
        pop_size=10,
        maximize=True,
        history_metrics=["best_fitness", "qd_score", "coverage"],
        centroids=jnp.ones((5, 2)),
    )
    result = adapter.run_once(key)

    assert len(result["history"]) == 2
    assert result["summary"]["best_fitness"] == pytest.approx(-10.5)
    assert result["summary"]["qd_score"] == 42.0
    assert result["summary"]["coverage"] == 0.8
    assert "timings" in result


def test_map_elites_adapter_run_once_qdax_replica():
    key = jax.random.PRNGKey(1)
    mock_emitter = MagicMock()
    mock_pop = MagicMock()
    mock_pop.copy.return_value = mock_pop
    mock_emitter.genome_config.init_population.return_value = mock_pop

    mock_engine = MagicMock()
    mock_engine.emitter = mock_emitter
    mock_engine.engine_params = SimpleNamespace(num_generations=1, key_derivation="qdax_replica")

    mock_state = MagicMock()
    mock_state.best_fitness = -2.0
    mock_state.generation = 1
    mock_state.qd_score = 10.0
    mock_state.coverage = 0.2
    mock_state.replace.return_value = mock_state

    mock_history = SimpleNamespace(
        best_fitness=jnp.array([-2.0]),
        qd_score=jnp.array([10.0]),
        coverage=jnp.array([0.2]),
    )
    mock_engine.init_state.return_value = mock_state
    mock_engine.run.return_value = (mock_state, mock_history, {})

    adapter = MapElitesEngineAdapter(
        engine=mock_engine,
        pop_size=5,
        maximize=False,
        history_metrics=None,
    )
    result = adapter.run_once(key)
    assert result["summary"]["best_fitness"] == pytest.approx(-2.0)
    assert result["summary"]["qd_score"] == 10.0


def test_map_elites_adapter_init_pop_ndarray_and_tuple():
    key = jax.random.PRNGKey(2)

    # 1. ndarray init_pop with genome_config
    mock_emitter = MagicMock()
    dummy_pop = MagicMock()
    mock_emitter.genome_config.init_population.return_value = dummy_pop
    dummy_pop.genes = MagicMock()
    dummy_pop.genes.replace.return_value = dummy_pop.genes
    dummy_pop.replace.return_value = dummy_pop
    dummy_pop.copy.return_value = dummy_pop

    mock_engine = MagicMock()
    mock_engine.emitter = mock_emitter
    mock_engine.engine_params = SimpleNamespace(num_generations=1, key_derivation="standard")
    mock_state = MagicMock(best_fitness=1.0, generation=1, qd_score=1.0, coverage=0.5)
    mock_engine.init_state.return_value = mock_state
    mock_engine.run.return_value = (mock_state, SimpleNamespace(), {})

    adapter = MapElitesEngineAdapter(
        engine=mock_engine,
        pop_size=4,
        maximize=True,
        history_metrics=[],
        initial_population=jnp.zeros((4, 3)),
    )
    res = adapter.run_once(key)
    assert "summary" in res

    # 2. ndarray with emitter without genome_config -> raises NotImplementedError
    mock_engine2 = MagicMock()
    mock_engine2.emitter = object()
    mock_engine2.engine_params = SimpleNamespace(num_generations=1)
    adapter2 = MapElitesEngineAdapter(
        engine=mock_engine2,
        pop_size=4,
        maximize=True,
        history_metrics=[],
        initial_population=jnp.zeros((4, 3)),
    )
    with pytest.raises(NotImplementedError):
        adapter2.run_once(key)

    # 3. Emitter lacks genome_config and genome when init_pop is None -> raises AttributeError
    adapter3 = MapElitesEngineAdapter(
        engine=mock_engine2,
        pop_size=4,
        maximize=True,
        history_metrics=[],
        initial_population=None,
    )
    with pytest.raises(AttributeError, match="Emitter lacks genome_config or genome"):
        adapter3.run_once(key)


def test_resolve_tensorneat_problem():
    # Valid "xor"
    prob, prob_state = resolve_tensorneat_problem("xor", None)
    assert prob is not None

    # Gymnax with env_name
    prob_gym, prob_gym_state = resolve_tensorneat_problem("gymnaxenv:env_name=CartPole-v1", None)
    assert prob_gym is not None

    # Unknown problem
    with pytest.raises(ValueError, match="Unknown TensorNEAT problem"):
        resolve_tensorneat_problem("nonexistent_unknown_problem", None)


def test_build_tensorneat_engine_errors():
    strat_bad_algo = TensorNEATStrategy(
        algorithm_name="nonexistent_algo",
        genome_name="default",
        problem_name="xor",
    )
    with pytest.raises(ValueError, match="Unknown TensorNEAT algorithm"):
        build_tensorneat_engine(
            strategy=strat_bad_algo,
            fitness_spec=None,
            pop_size=10,
            generations=5,
            maximize=True,
            history_metrics=None,
        )

    strat_bad_genome = TensorNEATStrategy(
        algorithm_name="neat",
        genome_name="nonexistent_genome",
        problem_name="xor",
    )
    with pytest.raises(ValueError, match="Unknown TensorNEAT genome"):
        build_tensorneat_engine(
            strategy=strat_bad_genome,
            fitness_spec=None,
            pop_size=10,
            generations=5,
            maximize=True,
            history_metrics=None,
        )


def test_build_map_elites_engine_validation():
    strat_no_emitter = MapElitesStrategy(emitter=None)
    with pytest.raises(ValueError, match="MapElitesStrategy requires an explicit emitter"):
        build_map_elites_engine(
            strategy=strat_no_emitter,
            fitness_spec=None,
            pop_size=10,
            generations=5,
            maximize=True,
            history_metrics=None,
        )
