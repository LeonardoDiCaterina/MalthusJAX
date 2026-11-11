"""PT5 diagnostic scheduling, rejection, artifact, and Slurm contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest
import tomllib

from malthusjax.core.timing import TimingCapture
from scripts import validate_timing as validation

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = tomllib.loads((ROOT / "configs/timing/pt5.toml").read_text())
    return cfg | {"seeds": [0, 43], "repetitions": 2}


def test_frozen_schedule_retains_first_and_all_repeats():
    cfg = config()
    rows = validation.schedule(cfg)
    assert len(rows) == 12
    assert [r["run_order"] for r in rows] == list(range(12))
    for seed in cfg["seeds"]:
        for name in validation.BACKENDS:
            assert [r["trial_id"] for r in rows if r["seed"] == seed and r["backend"] == name] == [
                0,
                1,
                2,
            ]
    assert [r["backend"] for r in rows[:4]] == list(validation.BACKENDS) + list(
        validation.BACKENDS[::-1]
    )
    assert validation.schedule(cfg) == rows


class Adapter:
    def __init__(self, contaminate=False):
        self.contaminate = contaminate

    def run_once(self, key):
        capture = TimingCapture()
        capture.initialized(key, self)
        capture.prepared()
        if self.contaminate:
            import logging

            logging.getLogger().warning("Finished XLA compilation (intentional test)")
        capture.executed()
        return {
            "history": [{"generation": 1, "best_fitness": 1.0}],
            "summary": {"best_fitness": 1.0},
            "timings": {"execution": 0.01},
            "timing_metadata": capture.metadata(),
        }


def setup_fake(monkeypatch, *, contamination=False, disagree=False):
    real_devices = list(jax.devices())
    monkeypatch.setattr(validation.jax, "devices", lambda: [real_devices[0]])
    monkeypatch.setattr(
        validation,
        "environment_identity",
        lambda: {"revision": "fixture", "backend": "cpu", "device": "fixture"},
    )

    def build(cfg):
        import logging

        logging.getLogger().warning("Compiling jit (intentional preparation signal)")
        return {name: Adapter(contamination) for name in validation.BACKENDS}

    monkeypatch.setattr(validation, "build_adapters", build)

    def reference(name, adapter, key):
        result = adapter.run_once(key)
        if disagree:
            result["summary"]["best_fitness"] = 2.0
        return result

    monkeypatch.setattr(validation, "reference_run", reference)


def execute(tmp_path, **kwargs):
    return validation.run_validation(
        config(),
        tmp_path / "fresh",
        expected_revision="fixture",
        expected_backend="cpu",
        expected_device=jax.devices()[0].device_kind,
        **kwargs,
    )


@pytest.mark.parametrize("contamination,disagree", [(False, False), (True, False), (False, True)])
def test_report_requires_phase_isolation_and_reference_agreement(
    tmp_path, monkeypatch, contamination, disagree
):
    setup_fake(monkeypatch, contamination=contamination, disagree=disagree)
    report = execute(tmp_path)
    assert report["status"] == ("failed" if contamination or disagree else "passed")
    assert len(report["raw_trials"]) == 12
    assert len(report["references"]) == 4
    assert len(report["phase_events"]) == 3 * (12 + 4)
    assert len(report["execution_compilation_events"]) == (12 if contamination else 0)
    assert report["compilation_log_summary"]["compilation_signals"] > 0
    exported = json.loads((tmp_path / "fresh/report.json").read_text())
    assert exported["status"] == report["status"]
    for backend in validation.BACKENDS:
        rows = json.loads((tmp_path / "fresh" / backend / "summary.json").read_text())["runs"]
        assert [r["seed"] for r in rows] == [0, 43]
        assert all(len(r["timing_record"]["trials"]) == 3 for r in rows)
        if contamination or disagree:
            assert all(r["timing_record"]["status"] == "unavailable" for r in rows)


def test_fresh_output_is_mandatory(tmp_path, monkeypatch):
    setup_fake(monkeypatch)
    execute(tmp_path)
    original = (tmp_path / "fresh/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert (tmp_path / "fresh/report.json").read_bytes() == original


def test_wrong_revision_fails_before_adapter_build(tmp_path, monkeypatch):
    setup_fake(monkeypatch)
    monkeypatch.setattr(
        validation, "build_adapters", lambda _: pytest.fail("Must reject before building")
    )
    report = validation.run_validation(
        config(),
        tmp_path / "fresh",
        expected_revision="wrong",
        expected_backend="cpu",
        expected_device="CPU",
    )
    assert report["status"] == "failed"
    assert "revision" in report["error"]
    assert not report["raw_trials"]


@pytest.mark.parametrize("wrapper", ["submit_timing_validation.sh", "submit_empirical_curves.sh"])
def test_slurm_wrapper_preserves_quoted_arguments_and_gpu(tmp_path, wrapper):
    tools = tmp_path / "tools"
    tools.mkdir()
    capture = tmp_path / "argv.jsonl"
    python = tools / "python"
    python.write_text(
        f'#!{sys.executable}\nimport json, os, sys\nwith open(os.environ["PT5_TEST_CAPTURE"], "a") as f:\n f.write(json.dumps({{"args":sys.argv[1:], "gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}})+"\\n")\n'
    )
    python.chmod(0o755)
    env = os.environ | {
        "PATH": f"{tools}:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": "GPU-assigned-by-slurm",
        "SLURM_JOB_ID": "1234",
        "SLURM_SUBMIT_DIR": str(ROOT),
        "PT5_TEST_CAPTURE": str(capture),
    }
    args = ["--output", "results/a path with spaces/new run", "--expect-revision", "exact revision"]
    subprocess.run(
        ["bash", str(ROOT / "scripts" / wrapper), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(capture.read_text().splitlines()[-1])
    assert row["args"][-len(args) :] == args
    assert row["gpu"] == "GPU-assigned-by-slurm"
