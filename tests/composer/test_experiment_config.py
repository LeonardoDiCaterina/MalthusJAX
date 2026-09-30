"""Tests for ExperimentConfig and structured config tree."""

from dataclasses import FrozenInstanceError

import pytest

from malthusjax.composer.experiment_config import (
    EvosaxBackendConfig,
    ExecutionConfig,
    ExperimentConfig,
    LoggingConfig,
    MalthusJAXBackendConfig,
    OutputConfig,
    PopulationConfig,
    QdaxBackendConfig,
    TensorneatBackendConfig,
)


def test_default_config_is_valid():
    """Default ExperimentConfig() passes validate() without error."""
    config = ExperimentConfig()
    config.validate()
    assert config.population.size == 50
    assert config.execution.generations == 100
    assert isinstance(config.backend, MalthusJAXBackendConfig)


def test_from_quick_run_kwargs_malthusjax():
    """Flat kwargs convert cleanly to structured MalthusJAXBackendConfig."""
    config = ExperimentConfig.from_quick_run_kwargs(
        pop_size=64,
        generations=200,
        seeds=(10, 20),
        fitness="sphere:dim=10",
        selection="tournament:num_selections=20,tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.1,mutation_strength=0.05",
        bounds=(-10.0, 10.0),
        genome_length=10,
        genome_type="real",
        maximize=True,
    )
    assert config.population.size == 64
    assert config.population.bounds == (-10.0, 10.0)
    assert config.population.genome_length == 10
    assert config.execution.generations == 200
    assert config.execution.seeds == (10, 20)
    assert config.execution.maximize is True
    assert isinstance(config.backend, MalthusJAXBackendConfig)
    assert config.backend.fitness == "sphere:dim=10"
    assert config.backend.crossover == "blend:alpha=0.5"


def test_from_quick_run_kwargs_evosax():
    """backend='evosax' properly routes to EvosaxBackendConfig."""
    config = ExperimentConfig.from_quick_run_kwargs(
        backend="evosax",
        evosax_strategy="OpenES",
        fitness="sphere:dim=5",
        pop_size=32,
        generations=50,
        seeds=(42,),
    )
    assert isinstance(config.backend, EvosaxBackendConfig)
    assert config.backend.strategy == "OpenES"
    assert config.backend.fitness == "sphere:dim=5"
    assert config.backend.composable is False
    assert config.population.size == 32


def test_from_quick_run_kwargs_composable_evosax():
    """backend='composable_evosax' sets composable=True."""
    config = ExperimentConfig.from_quick_run_kwargs(
        backend="composable_evosax",
        strategy="SimpleGA",
    )
    assert isinstance(config.backend, EvosaxBackendConfig)
    assert config.backend.composable is True


def test_from_quick_run_kwargs_tensorneat():
    """backend='tensorneat' properly routes and parses TensorneatBackendConfig."""
    config = ExperimentConfig.from_quick_run_kwargs(
        backend="tensorneat",
        tensorneat_algorithm="NEAT",
        tensorneat_genome="DefaultGenome",
        tensorneat_num_inputs=4,
        tensorneat_num_outputs=2,
    )
    assert isinstance(config.backend, TensorneatBackendConfig)
    assert config.backend.algorithm == "NEAT"
    assert config.backend.num_inputs == 4
    assert config.backend.num_outputs == 2


def test_from_quick_run_kwargs_qdax():
    """backend='qdax' properly routes and parses QdaxBackendConfig."""
    config = ExperimentConfig.from_quick_run_kwargs(
        backend="qdax",
        qdax_strategy="MAPElites",
        qdax_num_descriptors=3,
        qdax_num_centroids=150,
        qdax_mutation_sigma=0.2,
    )
    assert isinstance(config.backend, QdaxBackendConfig)
    assert config.backend.strategy_cls == "MAPElites"
    assert config.backend.num_descriptors == 3
    assert config.backend.num_centroids == 150
    assert config.backend.mutation_sigma == 0.2


def test_to_dict_from_dict_roundtrip():
    """Converting to dict and back produces identical configuration."""
    cfg = ExperimentConfig(
        population=PopulationConfig(size=128, genome_length=20, bounds=(-2.0, 2.0)),
        execution=ExecutionConfig(generations=50, seeds=(7, 8)),
        backend=EvosaxBackendConfig(strategy="CMA_ES", composable=True),
        logging=LoggingConfig(log_interval=10),
        output=OutputConfig(experiment_name="test_roundtrip"),
    )
    d = cfg.to_dict()
    restored = ExperimentConfig.from_dict(d)
    assert restored == cfg


def test_from_toml_produces_typed_configs(tmp_path):
    """ExperimentConfig.from_toml parses a TOML into typed ExperimentConfig instances."""
    toml_content = """
    [experiment]
    name = "toml_exp"
    output_dir = "results/custom"

    [experiment.shared]
    pop_size = 64
    bounds = [-3.0, 3.0]
    seeds = [1, 2]

    [pipelines.pipe_malthus]
    strategy = "ga"
    fitness = "sphere:dim=10"

    [pipelines.pipe_evosax]
    backend = "evosax"
    evosax_strategy = "OpenES"
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    configs = ExperimentConfig.from_toml(str(toml_file))
    assert "pipe_malthus" in configs
    assert "pipe_evosax" in configs

    pipe_m = configs["pipe_malthus"]
    assert pipe_m.population.size == 64
    assert pipe_m.population.bounds == (-3.0, 3.0)
    assert pipe_m.output.experiment_name == "pipe_malthus"
    assert pipe_m.output.output_dir == "results/custom"
    assert isinstance(pipe_m.backend, MalthusJAXBackendConfig)

    pipe_e = configs["pipe_evosax"]
    assert pipe_e.population.size == 64
    assert isinstance(pipe_e.backend, EvosaxBackendConfig)
    assert pipe_e.backend.strategy == "OpenES"


def test_validate_bad_bounds():
    """Bounds with lower >= upper raise ValueError."""
    cfg = ExperimentConfig(population=PopulationConfig(bounds=(5.0, -5.0)))
    with pytest.raises(ValueError, match="Invalid bounds"):
        cfg.validate()

    cfg2 = ExperimentConfig(population=PopulationConfig(bounds=(5.0, 5.0)))
    with pytest.raises(ValueError, match="Invalid bounds"):
        cfg2.validate()


def test_validate_empty_seeds():
    """Empty seeds tuple raises ValueError."""
    cfg = ExperimentConfig(execution=ExecutionConfig(seeds=()))
    with pytest.raises(ValueError, match="seeds tuple must not be empty"):
        cfg.validate()


def test_validate_invalid_pop_size_and_generations():
    """Non-positive pop size and generations raise ValueError."""
    cfg_pop = ExperimentConfig(population=PopulationConfig(size=0))
    with pytest.raises(ValueError, match="Population size must be positive"):
        cfg_pop.validate()

    cfg_gen = ExperimentConfig(execution=ExecutionConfig(generations=-1))
    with pytest.raises(ValueError, match="Generations must be positive"):
        cfg_gen.validate()


def test_validate_binary_genome_with_real_crossover():
    """Binary genome with real-only crossover raises ValueError."""
    cfg = ExperimentConfig(
        population=PopulationConfig(genome_type="binary"),
        backend=MalthusJAXBackendConfig(crossover="blend:alpha=0.5"),
    )
    with pytest.raises(ValueError, match="cannot be used with genome_type='binary'"):
        cfg.validate()


def test_validate_real_genome_with_binary_crossover():
    """Real genome with binary-only crossover raises ValueError."""
    cfg = ExperimentConfig(
        population=PopulationConfig(genome_type="real"),
        backend=MalthusJAXBackendConfig(crossover="single_point"),
    )
    with pytest.raises(ValueError, match="cannot be used with genome_type='real'"):
        cfg.validate()


def test_frozen_immutability():
    """Modifying fields on frozen configs raises FrozenInstanceError."""
    cfg = ExperimentConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.population = PopulationConfig(size=100)  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        cfg.population.size = 100  # type: ignore[misc]
