import hashlib
import json

import pytest

from malthusjax.benchmarking.publication import rebuild
from malthusjax.composer.adapters.metrics import archive_metric_metadata


def test_rebuild_from_saved_raw_inputs_is_deterministic_and_dry_run_is_readonly(tmp_path):
    paths = []
    for name, offset in (("left", 0.01), ("right", 0.0)):
        runs = [
            {
                "seed": i,
                "status": "success",
                "metric_metadata": archive_metric_metadata(False),
                "history": [
                    {"best_fitness": float(i) * 0.001 + offset, "coverage": 50.0, "qd_score": -1.0}
                ],
            }
            for i in range(12)
        ]
        data = {
            "schema_version": 1,
            "experiment": {
                "metadata": {
                    "timing_environment": {"revision": "fixture", "source_sha256": "fixture"}
                },
                "runs": runs,
            },
        }
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data))
        paths.append(path)
    policy = {
        "family": "fixture",
        "sample_size_rationale": "Synthetic tests only",
        "alpha": 0.05,
        "minimum_pairs": 10,
        "multiplicity": "holm",
        "hypotheses": [
            {
                "label": "fitness",
                "metric": "best_fitness",
                "kind": "location_shift",
                "primary_test": "wilcoxon",
                "estimand": "paired_signed_rank_location",
                "metric_orientation": "lower_is_better",
                "sidedness": "two_sided",
                "expected_direction": "left_lt_right",
            }
        ],
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    rebuild(*paths, policy_path, tmp_path / "out", dry_run=True)
    assert not (tmp_path / "out").exists()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    rebuild(*paths, policy_path, tmp_path / "a")
    rebuild(*paths, policy_path, tmp_path / "b")
    assert (tmp_path / "a/analysis.json").read_bytes() == (tmp_path / "b/analysis.json").read_bytes()
    assert (
        tmp_path / "a/corrected_inference.tex"
    ).read_bytes() == (tmp_path / "b/corrected_inference.tex").read_bytes()
    with pytest.raises(FileExistsError):
        rebuild(*paths, policy_path, tmp_path / "a")


def test_missing_provenance_cannot_make_a_table(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        rebuild(path, path, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
