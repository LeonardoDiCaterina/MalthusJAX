"""Tests for config integration through Composer, quick_run, compare, and TOML."""

from malthusjax.composer.composer import Composer


def test_quick_run_via_config_and_factory():
    """quick_run produces valid results using ExperimentConfig and EngineFactory internally."""
    composer = Composer.create_default()
    result = composer.quick_run(
        fitness="sphere:dim=5",
        pop_size=20,
        generations=5,
        seeds=(1,),
    )
    assert result is not None
    assert len(result.runs) == 1
    assert result.runs[0].status == "success"
    assert "best_fitness" in result.canonical_summary


def test_compare_uses_unified_pipeline():
    """compare() executes multiple pipelines seamlessly."""
    composer = Composer.create_default()
    comp = composer.compare(
        pipelines={
            "pipe1": dict(fitness="sphere:dim=5", crossover="blend:alpha=0.5"),
            "pipe2": dict(fitness="sphere:dim=5", crossover="uniform_real"),
        },
        pop_size=20,
        generations=5,
        seeds=(42,),
        shared_initial_population=False,
    )
    assert len(comp.pipelines) == 2
    assert "pipe1" in comp.pipelines
    assert "pipe2" in comp.pipelines


def test_from_toml_uses_typed_configs(tmp_path):
    """Composer.from_toml executes pipelines parsed as ExperimentConfig."""
    toml_content = """
    [experiment]
    name = "integration_exp"

    [experiment.shared]
    pop_size = 20
    generations = 5
    seeds = [1]
    bounds = [-5.0, 5.0]

    [pipelines.pipe_a]
    fitness = "sphere:dim=5"
    crossover = "blend:alpha=0.5"

    [pipelines.pipe_b]
    fitness = "sphere:dim=5"
    crossover = "uniform_real"
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    comp = Composer.from_toml(str(toml_file), shared_initial_population=False)
    assert len(comp.pipelines) == 2
    assert "pipe_a" in comp.pipelines
    assert "pipe_b" in comp.pipelines
