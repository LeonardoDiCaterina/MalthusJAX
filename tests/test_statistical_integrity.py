"""Regression tests for statistical analyzer, Cartesian pairing, and benchmark config invariants."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from malthusjax.stats.regression_analyzer import OLSRegressionAnalyzer, RegressionSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark_analyzer():
    script_path = REPO_ROOT / "scripts" / "benchmark_analyzer.py"
    spec = importlib.util.spec_from_file_location("benchmark_analyzer", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_lhs_dataframe(seed: int = 42) -> pd.DataFrame:
    """Create a synthetic multi-pipeline, multi-benchmark LHS DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    pipelines = ["ref_pipe", "target_a", "target_b"]
    benchmarks = ["sphere", "rosenbrock"]
    dims = [10, 20, 50, 100, 200]
    pops = [64, 128, 256]
    gens = [50, 100]

    for fn in benchmarks:
        for s in range(1, 31):
            d = int(rng.choice(dims))
            p = int(rng.choice(pops))
            g = int(rng.choice(gens))
            for pipe in pipelines:
                is_t = 0.0 if pipe == "ref_pipe" else (1.0 if pipe == "target_a" else 2.0)
                exec_time = (
                    0.01 * (d**0.5) * (p**0.3) * (1.0 + 0.1 * is_t * np.log(d))
                    + rng.uniform(0.001, 0.005)
                )
                best_fit = (
                    10.0 * d * (1.0 + 0.05 * is_t)
                    + rng.normal(0.0, 1.0)
                )
                rows.append(
                    {
                        "pipeline": pipe,
                        "fn_name": fn,
                        "seed": s,
                        "D": d,
                        "P": p,
                        "G": g,
                        "execution_time": float(max(1e-6, exec_time)),
                        "best_fitness": float(best_fit),
                    }
                )
    return pd.DataFrame(rows)


def test_ols_regression_analyzer_extracts_non_nan_diagnostics(tmp_path: Path):
    """Critique C1: Ensure BP_pval and SW_pval are populated (not NaN) from diagnostics."""
    df_global = _make_synthetic_lhs_dataframe()
    spec = RegressionSpec(dependent_vars=["execution_time", "best_fitness"])
    analyzer = OLSRegressionAnalyzer(spec)

    df_res = analyzer.analyze_suite(
        df_global=df_global,
        ref_pipeline="ref_pipe",
        target_pipelines=["target_a", "target_b"],
        analysis_dir=tmp_path,
    )

    assert not df_res.empty
    assert "BP_pval" in df_res.columns
    assert "SW_pval" in df_res.columns
    assert df_res["BP_pval"].notna().all(), f"Expected non-NaN BP_pval, got:\n{df_res['BP_pval']}"
    assert df_res["SW_pval"].notna().all(), f"Expected non-NaN SW_pval, got:\n{df_res['SW_pval']}"


def test_ols_regression_analyzer_partitions_holm_bonferroni_by_dependent_var(tmp_path: Path):
    """Critique C4: Ensure Holm-Bonferroni correction is computed per Dependent_Var family."""
    df_global = _make_synthetic_lhs_dataframe()

    # Run with both dependent variables together
    spec_combined = RegressionSpec(dependent_vars=["execution_time", "best_fitness"])
    df_combined = OLSRegressionAnalyzer(spec_combined).analyze_suite(
        df_global=df_global,
        ref_pipeline="ref_pipe",
        target_pipelines=["target_a", "target_b"],
        analysis_dir=tmp_path,
    )

    # Run with execution_time alone
    spec_exec_only = RegressionSpec(dependent_vars=["execution_time"])
    df_exec_only = OLSRegressionAnalyzer(spec_exec_only).analyze_suite(
        df_global=df_global,
        ref_pipeline="ref_pipe",
        target_pipelines=["target_a", "target_b"],
        analysis_dir=tmp_path,
    )

    combined_exec = (
        df_combined[df_combined["Dependent_Var"] == "execution_time"]
        .sort_values(["Target", "Benchmark"])
        .reset_index(drop=True)
    )
    solo_exec = df_exec_only.sort_values(["Target", "Benchmark"]).reset_index(drop=True)

    for col in ["beta_3_pval_holm", "beta_3_pval_HC3_holm"]:
        np.testing.assert_allclose(
            combined_exec[col].to_numpy(),
            solo_exec[col].to_numpy(),
            rtol=1e-12,
            err_msg=f"Holm-adjusted column {col} changed when another Dependent_Var was added!",
        )


def test_cartesian_pairing_strictly_aligns_p_g_seed():
    """Critique A5: Ensure multi-(P, G) Cartesian slices never cross-pair across P or G."""
    ba = _load_benchmark_analyzer()

    # Construct a slice for D=50 with two population sizes (P=100, P=500), shuffled row order,
    # and a missing/NaN observation in one cell (P=100, seed=2 for target).
    rows = [
        # P=500, G=100, seed=1
        {"is_treatment": 1, "P": 500, "G": 100, "seed": 1, "best_fitness": 501.0},
        {"is_treatment": 0, "P": 100, "G": 100, "seed": 1, "best_fitness": 101.5},
        # P=100, G=100, seed=1
        {"is_treatment": 1, "P": 100, "G": 100, "seed": 1, "best_fitness": 101.0},
        {"is_treatment": 0, "P": 500, "G": 100, "seed": 1, "best_fitness": 501.5},
        # P=100, G=100, seed=2 (target is NaN -> pair must be dropped cleanly)
        {"is_treatment": 1, "P": 100, "G": 100, "seed": 2, "best_fitness": np.nan},
        {"is_treatment": 0, "P": 100, "G": 100, "seed": 2, "best_fitness": 102.5},
        # P=500, G=100, seed=2
        {"is_treatment": 1, "P": 500, "G": 100, "seed": 2, "best_fitness": 502.0},
        {"is_treatment": 0, "P": 500, "G": 100, "seed": 2, "best_fitness": 502.5},
    ]
    df_d = pd.DataFrame(rows)

    aligned = ba.align_cartesian_pairs(df_d, "best_fitness")

    # Exactly 3 complete (P, G, seed) pairs should survive
    assert len(aligned) == 3
    assert list(aligned["P"]) == [100, 500, 500]
    assert list(aligned["seed"]) == [1, 1, 2]

    # Every paired difference (target - ref) must be -0.5; cross-pairing P=100 with P=500 would give ~400
    diffs = aligned["best_fitness_target"] - aligned["best_fitness_ref"]
    np.testing.assert_allclose(diffs.to_numpy(), np.array([-0.5, -0.5, -0.5]))


def test_h2_ablation_config_crossover_rate_parity():
    """Critique B4: Ensure all pipelines in h2_ablation_final_lhs.toml share crossover_rate=0.3."""
    toml_path = REPO_ROOT / "configs" / "h2_ablation_final_lhs.toml"
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    pipelines = cfg.get("pipelines", {})
    assert pipelines, "No pipelines found in h2_ablation_final_lhs.toml"

    for pipe_name, pipe_cfg in pipelines.items():
        if "crossover" in pipe_cfg:
            crossover_spec = pipe_cfg["crossover"]
            assert "crossover_rate=0.3" in crossover_spec, (
                f"Pipeline '{pipe_name}' has crossover='{crossover_spec}' "
                "without explicit crossover_rate=0.3"
            )

