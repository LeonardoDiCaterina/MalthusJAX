"""Exercise the actual generator/export path with deterministic measured runs."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.timing_records_helpers import record

SCRIPT = Path(__file__).resolve().parents[2] / "examples/generate_empirical_paper_curves.py"
spec = importlib.util.spec_from_file_location("empirical_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
# The live script sets runtime environment defaults; collection must not leak
# those defaults into unrelated tests in the same pytest process.
with patch.dict(os.environ):
    spec.loader.exec_module(generator)


@pytest.fixture
def measured_qd(monkeypatch, tmp_path):
    def experiment(seconds, qd_score):
        return SimpleNamespace(
            runs=[
                SimpleNamespace(
                    seed=seed,
                    status="success",
                    error=None,
                    metrics={},
                    timings={"execution": seconds},
                    timing_record=record(seconds, seed),
                    history=[
                        {"best_fitness": -2.0, "coverage": 50.0, "qd_score": qd_score},
                        {"best_fitness": -1.0, "coverage": 75.0, "qd_score": qd_score},
                    ],
                )
                for seed in (42, 43)
            ]
        )

    comp = SimpleNamespace(
        pipelines={
            "malthusjax_native": experiment(0.004, 10.0),
            "qdax_baseline": experiment(0.002, -20.0),
        }
    )
    composer = SimpleNamespace(compare=lambda **kwargs: comp)
    monkeypatch.setattr(
        generator, "Composer", SimpleNamespace(create_default=lambda: composer), raising=False
    )
    monkeypatch.setattr(
        generator, "OperatorCatalog", lambda: SimpleNamespace(get=lambda _: object()), raising=False
    )
    for name in ("GeneticMutationEmitter", "RealGenomeConfig", "MapElitesStrategy"):
        monkeypatch.setattr(
            generator, name, lambda **kwargs: SimpleNamespace(**kwargs), raising=False
        )
    monkeypatch.setattr(generator, "BENCH_SCRATCH_DIR", tmp_path / "runs")
    return generator.generate_dataset_5a_parity_and_speedup(
        {
            "d5_pop": 4,
            "d5_gens": 2,
            "d5_centroids": 2,
            "d5_seeds": (42, 43),
        }
    )


def payload(qd):
    strategies = (
        "malthusjax_ga_annealed",
        "malthusjax_evosax",
        "evosax_open_es",
        "malthusjax_qdax",
        "qdax_map_elites",
    )
    control = {
        "final_return": -2.0,
        "final_std": 1.0,
        "exec_time_mean": 1.0,
        "exec_time_std": 0.1,
        "throughput": 100,
    }
    pipelines = {name: dict(control) for name in strategies}
    operators = ("sbx_gauss_cosine", "blend_gauss_cosine", "blend_gauss_constant", "mutation_only")
    return {
        "metadata": {"tier": "smoke"},
        "dataset_1a": {
            **pipelines,
            "cartpole": pipelines,
            "meta": {
                "pop_size": 4,
                "generations": 2,
                "num_seeds": 2,
                "cartpole_dim": 3,
                "locomotion_dim": 4,
            },
            "throughput_steps_per_sec": {name: 100 for name in strategies},
        },
        "dataset_2a": {
            "max_niches": 4,
            "meta": {"generations": 2, "pop_size": 4, "num_seeds": 2},
            "table_metrics": {
                name: {"best_fit_mean": 2.0, "best_fit_std": 0.1, "niches": 2}
                for name in ("hybrid", "standalone_tensorneat", "standalone_qdax_mlp")
            },
        },
        "dataset_3a": {
            "meta": {},
            "scaling_results": [
                {
                    "best_fitness_mean": 1.0,
                    "best_fitness_std": 0.1,
                    "num_islands": 2,
                    "total_pop": 4,
                    "compile_time_s": 0.01,
                }
            ],
        },
        "dataset_4a": {
            "curves": {name: [1.0] for name in operators},
            "final_stds": {name: 0.1 for name in operators},
        },
        "dataset_5a": qd,
    }


def test_dataset_and_table_keep_measured_slowdown_and_signed_scores(measured_qd, tmp_path):
    assert measured_qd["speedup_ratio"] == 0.5
    assert measured_qd["fitness_qdax"] == [-2.0, -1.0]
    assert measured_qd["qd_score_qdax"] == [-20.0, -20.0]
    assert measured_qd["qd_score_comparison"]["status"] == "unavailable"
    assert measured_qd["stats"]["p_wilcoxon_qd"] is None
    assert measured_qd["per_seed"]["qdax_baseline"][0]["execution_seconds"] == 0.002
    tables = tmp_path / "tables"
    generator.export_latex_appendix_tables(payload(measured_qd), tables)
    table = (tables / "table_showcase3_qdax_speedup.tex").read_text()
    assert "0.50" in table
    assert "Ratios below one indicate slower" in table
    assert "-20.0" in table
    assert "Not comparable" in table
    assert "score higher" not in table
    assert "no sign. diff." not in table
    assert "Measured workload tier: smoke" in table
    before = {p.name: p.read_bytes() for p in tables.iterdir()}
    with pytest.raises(FileExistsError):
        generator.export_latex_appendix_tables(payload(measured_qd), tables)
    assert before == {p.name: p.read_bytes() for p in tables.iterdir()}


def test_main_keeps_all_outputs_in_explicit_run_directory(monkeypatch, measured_qd, tmp_path):
    data = payload(measured_qd)
    output = tmp_path / "new-run" / "empirical_curves.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--smoke", "--out", str(output)])
    monkeypatch.setattr(generator, "_load_benchmark_dependencies", lambda: None)
    for key, value in data.items():
        if key != "metadata":
            # Existing dataset function names are intentionally exercised through main().
            name = {
                "dataset_1a": "generate_dataset_1a_control_and_parity",
                "dataset_2a": "generate_dataset_2a_dissection",
                "dataset_3a": "generate_dataset_3a_islands",
                "dataset_4a": "generate_dataset_4a_operator_ablation",
                "dataset_5a": "generate_dataset_5a_parity_and_speedup",
            }[key]
            monkeypatch.setattr(generator, name, lambda cfg, value=value: value)
    generator.main()
    saved = json.loads(output.read_text())
    assert saved["metadata"]["smoke"] is True
    assert saved["metadata"]["audit_limitations"]
    assert saved["dataset_5a"]["speedup_ratio"] == 0.5
    assert generator.BENCH_SCRATCH_DIR == output.parent / "runs"
    assert len(list((output.parent / "tables").glob("*.tex"))) == 4
    assert not hasattr(generator, "PAPER_DATA_PATH")
    with pytest.raises(FileExistsError):
        generator.main()


def test_main_requires_output_before_loading_benchmark_dependencies(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--smoke"])
    with pytest.raises(SystemExit) as exc:
        generator.main()
    assert exc.value.code == 2


def test_corrected_inference_is_preserved_in_generated_table(measured_qd, tmp_path, monkeypatch):
    import malthusjax.stats.comparator as comparator
    from malthusjax.stats.core import TOSTResult
    from malthusjax.stats.paper import analyze_archive_records

    monkeypatch.setattr(
        comparator,
        "compute_tost_paired",
        lambda *a, **k: TOSTResult(1.0, -1.0, 1.0, 2.0, -2.0, 0.03, 0.03, 0.03, True),
    )
    hypotheses = [
        {
            "label": label,
            "metric": "best_fitness",
            "kind": "equivalence",
            "primary_test": "paired_t",
            "metric_orientation": "lower_is_better",
            "estimand": "mean_paired_difference",
            "margin": 1.0,
            "sidedness": "two_sided",
            "expected_direction": "left_lt_right",
        }
        for label in ("eq1", "eq2")
    ]
    policy = {
        "family": "synthetic family",
        "sample_size_rationale": "Regression fixture, not a power claim",
        "alpha": 0.05,
        "minimum_pairs": 2,
        "multiplicity": "holm",
        "hypotheses": hypotheses,
    }
    measured_qd["statistical_analysis"] = analyze_archive_records(
        measured_qd["per_seed"]["malthusjax_native"],
        measured_qd["per_seed"]["qdax_baseline"],
        policy,
    )
    generator.export_latex_appendix_tables(payload(measured_qd), tmp_path)
    table = (tmp_path / "table_showcase3_qdax_speedup.tex").read_text()
    assert "equivalence not established; p=0.06 (family corrected)" in table
    assert "p=0.03" not in table
