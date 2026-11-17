"""Regression tests for the wheel-to-research workflow."""

import csv
import json
from pathlib import Path

import pytest

from malthusjax.benchmarking.cli import _load_comparison, main
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import Composer

CONFIG = """
[experiment.shared]
fitness = "native_sphere"
genome_length = 2
pop_size = 8
generations = 2
seeds = [11, 22]

[pipelines.low]
mutation = "gaussian:mutation_rate=0.05"

[pipelines.high]
mutation = "gaussian:mutation_rate=0.50"
"""


def test_cli_results_are_isolated_and_manifest_validated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("comparison.toml")
    source.write_text('[experiment]\noutput_dir = "configured"\n' + CONFIG)

    assert main(["run", str(source)]) == 0
    output = Path("configured")
    records = set(output.glob("data/pipeline_*/seed_*.json"))
    assert len(records) == 4
    seed_record = output / "data/pipeline_low/seed_11.json"
    run = json.loads(seed_record.read_text())
    assert run["status"] == "success"
    solution = run["solution"]
    assert solution["representation"] == "encoded_genome"
    assert sum(gene**2 for gene in solution["value"]) == pytest.approx(
        run["metrics"]["best_fitness"]
    )
    with (output / "histories_combined.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["pipeline"] for row in rows} == {"low", "high"}
    assert (output / "exports/pipeline_low/summary.json").is_file()
    assert (output / "exports/pipeline_high/summary.json").is_file()

    source.write_text(source.read_text().replace("[11, 22]", "[11]"))
    assert main(["parity", str(source)]) == 1
    assert records == set(output.glob("data/pipeline_*/seed_*.json"))
    assert "seeds = [11, 22]" in (output / "metadata/config_snapshot.toml").read_text()

    assert main(["parity", str(source), "--output-dir", "fresh"]) == 0
    fresh = Path("fresh")
    assert len(list(fresh.glob("data/pipeline_*/seed_*.json"))) == 2
    assert _load_comparison(fresh).pipelines["low"].runs[0].solution == solution

    extra = fresh / "data/pipeline_low/seed_22.json"
    extra.write_text(seed_record.read_text())
    with pytest.raises(ValueError, match="Unexpected seed records"):
        _load_comparison(fresh)
    extra.unlink()
    (fresh / "data/pipeline_low/seed_11.json").unlink()
    with pytest.raises(ValueError, match="Missing manifest seed record"):
        _load_comparison(fresh)


@pytest.mark.parametrize("placement", ["experiment", "experiment.shared"])
def test_toml_initialization_reports_actual_population(tmp_path, placement):
    source = tmp_path / "initialization.toml"
    config = (
        "[experiment]\nshared_initial_population = false\n" + CONFIG
        if placement == "experiment"
        else CONFIG.replace(
            "[experiment.shared]", "[experiment.shared]\nshared_initial_population = false"
        )
    )
    source.write_text(
        config.replace("fitness =", "capture_population_fingerprint = true\nfitness =")
    )
    result = Composer.from_toml(source, write_artifacts=False)

    first, second = result.pipelines["low"].runs
    assert first.initialization["policy"] == "backend_seeded"
    assert first.initialization["optimizer_seed"] == first.seed
    assert first.initialization["population_sha256"] != second.initialization["population_sha256"]
    assert [run.initialization for run in result.pipelines["low"].runs] == [
        run.initialization for run in result.pipelines["high"].runs
    ]


def test_pipeline_validation_precedes_optimization(tmp_path, monkeypatch):
    source = tmp_path / "typo.toml"
    source.write_text(CONFIG.replace("mutation_rate=0.50", "mutaton_rate=0.50"))

    def unexpected_run(*args, **kwargs):
        pytest.fail("A pipeline started running before validation completed")

    monkeypatch.setattr(BenchmarkRunner, "run", unexpected_run)
    with pytest.raises(ValueError, match="Pipeline 'high'.*mutaton_rate"):
        Composer.from_toml(source, write_artifacts=False)


def test_qdax_archive_path_is_relative_to_published_results(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("qdax")
    monkeypatch.chdir(tmp_path)
    source = Path("qdax.toml")
    source.write_text(
        CONFIG.split("[pipelines.low]")[0].replace("[11, 22]", "[11]")
        + """
[pipelines.qd]
backend = "qdax"
qdax_num_centroids = 9
export_archive = true
"""
    )
    assert main(["run", str(source)]) == 0

    root = Path("results/qdax")
    run = _load_comparison(root).pipelines["qd"].runs[0]
    assert run.status == "success", run.error
    assert "initial_fitness" in run.unavailable_metrics
    assert "first_generation_best_fitness" in run.metrics
    path = Path(run.artifacts["archive"])
    assert not path.is_absolute()
    with np.load(root / path) as archive:
        occupied = len(archive["cell_index"])
        assert 1 <= occupied <= 9
        assert archive["genotypes"].shape == (occupied, 2)
        assert archive["descriptors"].shape == (occupied, 2)
        assert np.isfinite(archive["archive_fitness"]).all()


def test_evosax_strategy_params_reach_composable_adapter():
    pytest.importorskip("evosax")

    from malthusjax.composer.engine_factory import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig

    config = ExperimentConfig.from_quick_run_kwargs(
        backend="evosax",
        composable=True,
        evosax_strategy="SimpleGA",
        fitness="native_sphere",
        genome_length=2,
        pop_size=8,
        generations=2,
        strategy_params={"mutation_std": 0.125},
    )
    adapter = EngineFactory.build(config)
    assert float(adapter.strategy.std_schedule(0)) == pytest.approx(0.125)
