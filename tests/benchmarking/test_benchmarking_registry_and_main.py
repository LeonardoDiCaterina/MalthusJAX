import runpy
import sys
from pathlib import Path

import pytest

from malthusjax.benchmarking.cli import main
from malthusjax.benchmarking.registry import DataRegistry


class DummyRunResult:
    def __init__(self, runs, metadata):
        self.runs = runs
        self.metadata = metadata

    def aggregated_summary(self):
        return {"best_fitness": {"mean": 0.123}}


class DummyComposer:
    @classmethod
    def create_default(cls):
        return cls()

    def quick_run(self, *args, **kwargs):
        return DummyRunResult(
            runs=[1, 2, 3],
            metadata={
                "artifact_paths": {"summary_json": "results/summary.json"}
            },
        )


def test_data_registry_synthetic_source():
    registry = DataRegistry()
    config = {"source": "synthetic", "shape": [2, 2]}

    registry.register("example", config)
    assert registry.resolve("example") == config


def test_data_registry_unknown_id_raises_key_error():
    registry = DataRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.resolve("missing")


def test_data_registry_file_source(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("1.0,2.0\n3.0,4.0\n")

    registry = DataRegistry()
    registry.register("filedata", {"source": "file", "path": str(path)})

    loaded = registry.resolve("filedata")
    assert hasattr(loaded, "shape")


def test_cli_main_quiet(monkeypatch, capsys):
    from malthusjax.benchmarking import cli

    monkeypatch.setattr(cli, "Composer", DummyComposer)

    result = main(["catalog"])
    assert result == 0

    captured = capsys.readouterr()
    assert "MalthusJAX Operator Catalog" in captured.out


def test_benchmarking_module_main_executes(monkeypatch):
    from malthusjax.benchmarking import cli

    monkeypatch.setattr(cli, "Composer", DummyComposer)
    argv = [
        "malthusjax.benchmarking",
        "catalog",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("malthusjax.benchmarking.__main__", run_name="__main__")

    assert exc.value.code == 0
