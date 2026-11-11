"""End-to-end aggregation from ledgers, including externally retained units."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_publication_campaign import ROOT, source_hashes
from run_publication_study import work_units


def test_aggregate_completed_phases_without_touching_inputs(tmp_path):
    protocol = copy.deepcopy(json.loads((ROOT / "configs/publication/study_v2.json").read_text()))
    protocol["phase1"].update(
        seeds=[2000, 2001],
        functions=["sphere"],
        dims=[10],
        populations=[128],
        seeds_per_work_unit=1,
    )
    campaign = tmp_path / "campaign"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for phase in (1, 2):
        phase_root = campaign / f"phase{phase}"
        phase_root.mkdir(parents=True)
        sources = source_hashes(phase)
        entries = {}
        for index, unit in enumerate(work_units(protocol, phase)):
            # Units0/1 deliberately live outside the campaign directory.
            parent = (
                tmp_path / f"external{index}"
                if phase == 2 and index < 2
                else phase_root / f"result{index}"
            )
            child = parent / f"unit{index}"
            child.mkdir(parents=True)
            manifest = {
                "phase": phase,
                "status": "success",
                "exit_code": 0,
                "revision": revision,
                "source_sha256": sources,
                "work_units": [{"index": index, **unit}],
            }
            (parent / "study_manifest.json").write_text(json.dumps(manifest))
            (child / "manifest.json").write_text(json.dumps({"packages": {}, "versions": {}}))
            seed = unit["seeds"][0]
            if phase == 1:
                block = f"sphere_d10_p128_s{seed}"
                (child / block).mkdir()
                conditions = [
                    "evosax_adapter",
                    "evosax_standard",
                    "evosax_batched",
                    "qdax_adapter",
                    "qdax_native",
                ]
                for i, condition in enumerate(conditions):
                    (child / block / f"{condition}.json").write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "input_population_hash": f"shared{seed}",
                                "metrics": {
                                    "best_fitness": i + seed / 1000,
                                    "coverage": 80 + i,
                                    "qd_score": 100 + i,
                                },
                            }
                        )
                    )
                summary = {
                    "failed_conditions": 0,
                    "blocks": [{"block": block, "conditions": conditions}],
                }
            else:
                condition = unit["arguments"][unit["arguments"].index("--conditions") + 1]
                i = protocol["phase2"]["conditions"].index(condition)
                run = {
                    "seed": seed,
                    "condition": condition,
                    "status": "success",
                    "study_mode": "ranking",
                    "initial_population_sha256": f"shared{seed}",
                    "heldout_mean_return": seed + i * 20 + i * ((seed % 5) - 2),
                    "heldout_episode_returns": [1.0] * 20,
                    "heldout_rollout_diagnostics": [
                        {"rollout_invalid_live_steps": 0, "rollout_invalid_resets": 0}
                    ]
                    * 20,
                    "cost": {"train_end_to_end_seconds": 100 + i},
                }
                summary = {"successful": 1, "failed": 0, "runs": [run]}
            (child / "summary.json").write_text(json.dumps(summary))
            entries[str(index)] = {"status": "success", "result_directory": str(parent)}
        ledger = {
            "phase": phase,
            "status": "complete",
            "revision": revision,
            "protocol": protocol,
            "units": entries,
            "scientific_identity": {"scientific_sources": sources},
        }
        (phase_root / "campaign.json").write_text(json.dumps(ledger))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.json")}
    output = tmp_path / "analysis"
    command = [
        "bash",
        str(ROOT / "scripts/aggregate_publication_phases12.sh"),
        str(campaign),
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(
            os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PHASE 1: 10 runs" in result.stdout
    assert "PHASE 2: 80 runs" in result.stdout
    index = json.loads((output / "input_index.json").read_text())
    assert any("external0" in p for p in index["phase2"]["inputs"])
    assert (output / "phase2/phase2_return_cost.png").stat().st_size > 1000
    assert (output / "phase1_contrasts.csv").exists()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert not (campaign / "phase3").exists()
    repeated = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert repeated.returncode != 0
