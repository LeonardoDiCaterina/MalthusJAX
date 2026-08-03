"""Tests for the EngineRegistry catalog class."""

import pytest

from malthusjax.composer.engine_catalog import EngineRegistry
from malthusjax.composer.engine_registry import _ENGINE_REGISTRY


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore the global engine registry around each test."""
    saved = dict(_ENGINE_REGISTRY)
    yield
    _ENGINE_REGISTRY.clear()
    _ENGINE_REGISTRY.update(saved)


# -- parse_spec -----------------------------------------------------------


def test_parse_spec_simple():
    """Plain engine name, no params."""
    registry = EngineRegistry()
    name, params = registry.parse_spec("ga")
    assert name == "ga"
    assert params == {}


def test_parse_spec_with_params():
    """Engine name with colon-separated params."""
    registry = EngineRegistry()
    name, params = registry.parse_spec("ga:pop_size=200,elitism=4")
    assert name == "ga"
    assert params == {"pop_size": 200, "elitism": 4}


def test_parse_spec_float_param():
    """Float values are converted correctly."""
    registry = EngineRegistry()
    _, params = registry.parse_spec("ga:sigma=0.3")
    assert params["sigma"] == 0.3
    assert isinstance(params["sigma"], float)


def test_parse_spec_bool_param():
    """Boolean values are converted correctly."""
    registry = EngineRegistry()
    _, params = registry.parse_spec("ga:enable=True,disable=False")
    assert params["enable"] is True
    assert params["disable"] is False


def test_parse_spec_string_param():
    """Quoted strings are unquoted."""
    registry = EngineRegistry()
    _, params = registry.parse_spec('ga:name="custom"')
    assert params["name"] == "custom"


def test_parse_spec_empty_raises():
    """Empty spec raises ValueError."""
    registry = EngineRegistry()
    with pytest.raises(ValueError, match="Empty engine specification"):
        registry.parse_spec("")


def test_parse_spec_bad_param_raises():
    """Missing '=' in param pair raises ValueError."""
    registry = EngineRegistry()
    with pytest.raises(ValueError, match="Invalid parameter format"):
        registry.parse_spec("ga:badparam")


# -- get -------------------------------------------------------------------


def test_get_ga_default():
    """'ga' is pre-registered and returns a GeneticEngineAdapter."""
    from malthusjax.composer.catalog import OperatorCatalog

    catalog = OperatorCatalog()
    registry = EngineRegistry()

    engine = registry.get(
        "ga",
        evaluator=catalog.get("sphere:dim=5"),
        selection=catalog.get("tournament:num_selections=10,tournament_size=3"),
        crossover=catalog.get("blend:alpha=0.5"),
        mutation=catalog.get("gaussian:mutation_rate=0.1"),
        pop_size=20,
        generations=5,
        genome_length=5,
    )

    # Should return a GeneticEngineAdapter with run_once method
    assert hasattr(engine, "run_once"), "Engine must satisfy Engine protocol"
    assert callable(engine.run_once)


def test_get_with_spec_overrides():
    """Spec-level params override defaults."""
    from malthusjax.composer.catalog import OperatorCatalog

    catalog = OperatorCatalog()
    registry = EngineRegistry()

    engine = registry.get(
        "ga:pop_size=30,elitism=1",
        evaluator=catalog.get("sphere:dim=5"),
        selection=catalog.get("tournament:num_selections=10,tournament_size=3"),
        crossover=catalog.get("blend:alpha=0.5"),
        mutation=catalog.get("gaussian:mutation_rate=0.1"),
        generations=3,
        genome_length=5,
    )

    # Verify it built with the overridden pop_size
    assert engine.genetic_engine.engine_params.pop_size == 30
    assert engine.genetic_engine.engine_params.elitism == 1


def test_get_unknown_engine_raises():
    """Requesting an unregistered engine raises KeyError."""
    registry = EngineRegistry()
    with pytest.raises(KeyError, match="Unknown engine"):
        registry.get(
            "nonexistent",
            evaluator=None,
            selection=None,
            crossover=None,
            mutation=None,
        )


# -- list_available --------------------------------------------------------


def test_list_available_includes_ga():
    """'ga' appears in available engines."""
    registry = EngineRegistry()
    available = registry.list_available()
    assert "ga" in available


def test_list_available_sorted():
    """Available list is sorted."""
    registry = EngineRegistry()
    available = registry.list_available()
    assert available == sorted(available)


# -- register (runtime) ---------------------------------------------------


def test_runtime_register():
    """Custom engine can be registered at runtime."""
    registry = EngineRegistry()

    def custom_factory(evaluator, selection, crossover, mutation, **kwargs):
        return {"custom": True, **kwargs}

    registry.register("custom_test", custom_factory)
    assert "custom_test" in registry.list_available()

    engine = registry.get(
        "custom_test",
        evaluator=None,
        selection=None,
        crossover=None,
        mutation=None,
        x=42,
    )
    assert engine["custom"] is True
    assert engine["x"] == 42


def test_runtime_register_duplicate_raises():
    """Duplicate runtime registration raises without override."""
    registry = EngineRegistry()

    registry.register("dup_test", lambda **kw: None)
    with pytest.raises(KeyError, match="already registered"):
        registry.register("dup_test", lambda **kw: None)


def test_runtime_register_persists():
    """Runtime registration is visible to new EngineRegistry instances."""
    registry1 = EngineRegistry()
    registry1.register(
        "persist_test",
        lambda evaluator, selection, crossover, mutation, **kw: "persisted",
    )

    registry2 = EngineRegistry()
    assert "persist_test" in registry2.list_available()


# -- get_help --------------------------------------------------------------


def test_get_help_ga():
    """get_help returns documentation for 'ga'."""
    registry = EngineRegistry()
    help_text = registry.get_help("ga")
    assert "ga" in help_text
    assert "Defaults:" in help_text


def test_get_help_unknown_raises():
    """get_help for unknown engine raises KeyError."""
    registry = EngineRegistry()
    with pytest.raises(KeyError, match="Unknown engine"):
        registry.get_help("nonexistent")


# -- integration with Composer -------------------------------------------


def test_composer_engine_type_param():
    """Composer.quick_run accepts engine_type parameter."""
    from malthusjax.composer import Composer

    composer = Composer.create_default()

    # Run with explicit engine_type="ga" (the default)
    result = composer.quick_run(
        engine_type="ga",
        fitness="sphere:dim=5",
        selection="tournament:num_selections=10,tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.1",
        pop_size=20,
        generations=3,
        genome_length=5,
        seeds=(42,),
        experiment_name="engine_type_test",
    )

    assert result is not None
    assert len(result.runs) == 1
