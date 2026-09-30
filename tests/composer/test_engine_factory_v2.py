"""Tests for EngineFactory config -> engine resolution."""

import pytest

from malthusjax.composer.engine_factory import GeneticEngineAdapter
from malthusjax.composer.engine_factory_v2 import EngineFactory
from malthusjax.composer.evosax_adapter import EvosaxEngineAdapter
from malthusjax.composer.experiment_config import (
    BackendConfig,
    EvosaxBackendConfig,
    ExecutionConfig,
    ExperimentConfig,
    MalthusJAXBackendConfig,
    PopulationConfig,
)


def test_build_malthusjax_from_config():
    """MalthusJAX config builds GeneticEngineAdapter."""
    factory = EngineFactory()
    cfg = ExperimentConfig(
        population=PopulationConfig(size=32, genome_length=5),
        execution=ExecutionConfig(generations=10),
        backend=MalthusJAXBackendConfig(
            fitness="sphere:dim=5",
            selection="tournament:num_selections=16,tournament_size=2",
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1,mutation_strength=0.05",
        ),
    )
    engine = factory.build(cfg)
    assert isinstance(engine, GeneticEngineAdapter)
    assert hasattr(engine, "run_once")
    assert callable(engine.run_once)


def test_build_evosax_from_config():
    """Evosax config builds EvosaxEngineAdapter."""
    factory = EngineFactory()
    cfg = ExperimentConfig(
        population=PopulationConfig(size=32, genome_length=5),
        execution=ExecutionConfig(generations=10),
        backend=EvosaxBackendConfig(
            strategy="SimpleGA",
            fitness="sphere:dim=5",
        ),
    )
    engine = factory.build(cfg)
    assert isinstance(engine, EvosaxEngineAdapter)
    assert hasattr(engine, "run_once")
    assert callable(engine.run_once)


def test_build_stub_from_config():
    """Config without operators builds StubEngine."""
    factory = EngineFactory()
    cfg = ExperimentConfig(
        backend=MalthusJAXBackendConfig(fitness=None),
    )
    engine = factory.build(cfg)
    assert hasattr(engine, "run_once")
    assert callable(engine.run_once)


def test_unknown_backend_raises():
    """Unknown backend raises ValueError."""
    factory = EngineFactory()
    cfg = ExperimentConfig(backend=BackendConfig(name="nonexistent_backend_xyz"))
    with pytest.raises(ValueError, match="Unknown backend"):
        factory.build(cfg)


def test_invalid_config_type_raises():
    """Passing invalid object type raises TypeError."""
    factory = EngineFactory()
    with pytest.raises(TypeError, match="Expected ExperimentConfig"):
        factory.build("invalid_config_type")  # type: ignore[arg-type]


def test_build_with_evaluator():
    """build_with_evaluator returns both engine and evaluator."""
    factory = EngineFactory()
    cfg = ExperimentConfig(
        population=PopulationConfig(size=32, genome_length=5),
        execution=ExecutionConfig(generations=10),
        backend=MalthusJAXBackendConfig(
            fitness="sphere:dim=5",
            selection="tournament:num_selections=16,tournament_size=2",
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1,mutation_strength=0.05",
        ),
    )
    engine, evaluator = factory.build_with_evaluator(cfg)
    assert isinstance(engine, GeneticEngineAdapter)
    assert evaluator is not None


def test_config_not_mutated():
    """Original ExperimentConfig remains unmodified during build."""
    factory = EngineFactory()
    cfg = ExperimentConfig(
        population=PopulationConfig(size=32, genome_length=5),
        execution=ExecutionConfig(generations=10),
        backend=MalthusJAXBackendConfig(
            fitness="sphere:dim=5",
            selection="tournament:num_selections=16,tournament_size=2",
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1,mutation_strength=0.05",
        ),
    )
    orig_dict = cfg.to_dict()
    factory.build(cfg)
    assert cfg.to_dict() == orig_dict
