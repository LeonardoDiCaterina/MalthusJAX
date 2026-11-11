"""Retained-result identity and no-execution planning guarantees."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_publication_campaign import ROOT, source_hashes, valid_result
from run_publication_study import work_units


def test_retained_unit_requires_matching_inputs_and_success(tmp_path):
    protocol = json.loads((ROOT / "configs/publication/study_v2.json").read_text())
    unit = work_units(protocol, 2)[1]
    sources = source_hashes(2)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "phase": 2,
        "status": "success",
        "exit_code": 0,
        "revision": revision,
        "work_units": [{"index": 1, **unit}],
        "source_sha256": sources,
    }
    (tmp_path / "study_manifest.json").write_text(json.dumps(manifest))
    child = tmp_path / "unit1"
    child.mkdir()
    (child / "manifest.json").write_text(json.dumps({"packages": {}}))
    (child / "summary.json").write_text(
        json.dumps(
            {
                "successful": 1,
                "failed": 0,
                "runs": [{"seed": 2000, "condition": "native_ga_cosine", "status": "success"}],
            }
        )
    )
    valid_result(tmp_path, 2, 1, unit, sources)
    changed = {**unit, "seeds": [2001]}
    with pytest.raises(ValueError, match="arguments differ"):
        valid_result(tmp_path, 2, 1, changed, sources)
    with pytest.raises(ValueError, match="source differs"):
        valid_result(tmp_path, 2, 1, unit, {**sources, "scripts/publication_rl.py": "wrong"})
    manifest["status"] = "running"
    (tmp_path / "study_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a successful"):
        valid_result(tmp_path, 2, 1, unit, sources)


def test_plan_has_no_output_mutation_and_no_unallocated_execution(tmp_path):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    command = [
        sys.executable,
        str(ROOT / "scripts/run_publication_campaign.py"),
        "--phase",
        "2",
        "--expect-revision",
        revision,
        "--output",
        str(tmp_path / "campaign"),
    ]
    planned = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(planned.stdout)
    assert report["remaining_units"] == 80
    assert report["retained_units"] == 0
    assert not (tmp_path / "campaign").exists()
    import os

    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    rejected = subprocess.run([*command, "--execute"], cwd=ROOT, env=env, capture_output=True)
    assert rejected.returncode != 0
    assert not (tmp_path / "campaign").exists()
