"""Study-family integrity, actual mutation schedule and Pareto orientation."""

import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_publication_study import CONDITIONS, nondominated, phase1_estimation, ranking
from run_publication_study import work_units

from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.schedules import ScheduleType
from malthusjax.operators.mutation.real import GaussianMutation


def test_actual_cosine_noise_changes_with_generation():
    mutation = GaussianMutation(
        mutation_rate=1.0,
        mutation_strength=0.2,
        schedule_type=ScheduleType.COSINE_ANNEAL,
        final_strength=0.01,
        max_generations=3,
    )
    keys = jax.random.split(jax.random.PRNGKey(2), 2)
    config = RealGenomeConfig(shape=(12,))
    noise = [np.asarray(mutation._generate_noise(keys, config, generation=g)) for g in range(3)]
    assert np.linalg.norm(noise[0]) > np.linalg.norm(noise[1]) > np.linalg.norm(noise[2])
    np.testing.assert_allclose(noise[2], noise[0] * 0.05, rtol=2e-5, atol=1e-7)


def rows():
    result = []
    for seed in range(20):
        for i, condition in enumerate(CONDITIONS):
            result.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "status": "success",
                    "heldout_mean_return": seed + i * 20 + i * ((seed % 5) - 2),
                    "initial_population_sha256": str(seed),
                    "study_mode": "ranking",
                    "heldout_episode_returns": [1.0] * 20,
                    "cost": {"train_end_to_end_seconds": 100 + i},
                }
            )
    return result


def test_ranking_paired_family_and_missingness():
    data = rows()
    result = ranking(data, list(range(20)))
    assert result["inferential_family_available"]
    assert len(result["contrasts"]) == 6
    assert all(c["estimate"]["mean"] < 0 for c in result["contrasts"])
    assert all(c["holm_p"] >= c["raw_p"] for c in result["contrasts"])
    incomplete = ranking(data[:-1], list(range(20)))
    assert not incomplete["inferential_family_available"]
    assert all(c["holm_p"] is None for c in incomplete["contrasts"])


def test_fronts_minimize_negative_return_and_connections():
    np.testing.assert_array_equal(
        nondominated([[-10, 10], [-9, 12], [-8, 8], [-10, 10]]), [True, False, True, True]
    )


def test_unit_schedule_does_not_repeat_or_pool_seeds():
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/publication/study_v2.json").read_text()
    )
    units = work_units(protocol, 1)
    assert len(units) == 240
    assert sum(len(u["seeds"]) * u["conditions"] for u in units) == 6000
    for phase in (2, 3):
        units = work_units(protocol, phase)
        assert len(units) == 80
        assert all(len(u["seeds"]) == 1 for u in units)
    assert set(protocol["phase1"]["seeds"]).isdisjoint({1000, 1001, 1002})


def test_analysis_writes_real_plot_artifacts_from_synthetic_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    phase2 = tmp_path / "summary.json"
    phase2.write_text(json.dumps({"runs": rows()}))
    phase3 = tmp_path / "report.json"
    phase3.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "seed": 17,
                        "topology": "ring",
                        "status": "success",
                        "initial_population_hash": "shared",
                        "heldout": {"objectives_minimize": [[-10, 10], [-8, 8], [-5, 12]]},
                    }
                ]
            }
        )
    )
    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_publication_study.py"),
            "--phase2",
            str(phase2),
            "--phase3",
            str(phase3),
            "--expected-seeds",
            *map(str, range(20)),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert (output / "phase2_return_cost.png").stat().st_size > 1000
    assert (output / "phase3_front_seed17.png").stat().st_size > 1000
    assert json.loads((output / "phase2_ranking.json").read_text())["inferential_family_available"]


def test_phase1_pending_bounds_never_become_equivalence(tmp_path):
    block = tmp_path / "sphere_d10_p128_s2000"
    block.mkdir()
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {"blocks": [{"block": block.name, "conditions": ["evosax_standard", "evosax_adapter"]}]}
        )
    )
    for condition, value in [("evosax_standard", 1.0), ("evosax_adapter", 2.0)]:
        (block / f"{condition}.json").write_text(
            json.dumps({"status": "success", "metrics": {"best_fitness": value}})
        )
    report = phase1_estimation([tmp_path], [2000, 2001])
    assert not report["equivalence_claim"]
    assert report["contrasts"][0]["estimate"]["mean"] == -1
    assert report["contrasts"][0]["missing_or_failed_seeds"] == [2001]
