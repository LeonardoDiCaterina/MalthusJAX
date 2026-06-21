import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from malthusjax.stats.regression_analyzer import OLSRegressionAnalyzer, RegressionSpec

def test_regression_analyzer_handles_lhs_data(tmp_path: Path):
    # Setup mock LHS dataset
    np.random.seed(42)
    n_samples = 50
    
    # We need variance in D to trigger the OLS regression block
    D_vals = np.random.choice([10, 20, 50, 100], size=n_samples)
    P_vals = np.random.choice([100, 200], size=n_samples)
    G_vals = np.random.choice([50, 100], size=n_samples)
    
    data_target = {
        "fn_name": ["sphere"] * n_samples,
        "seed": list(range(n_samples)),
        "D": D_vals,
        "P": P_vals,
        "G": G_vals,
        "pipeline": ["target"] * n_samples,
        "execution_time": np.random.uniform(1.0, 5.0, size=n_samples),
        "best_fitness": np.random.uniform(0.1, 1.0, size=n_samples),
    }
    
    data_ref = {
        "fn_name": ["sphere"] * n_samples,
        "seed": list(range(n_samples)),
        "D": D_vals,
        "P": P_vals,
        "G": G_vals,
        "pipeline": ["ref"] * n_samples,
        "execution_time": np.random.uniform(2.0, 10.0, size=n_samples),
        "best_fitness": np.random.uniform(0.5, 2.0, size=n_samples),
    }
    
    df_global = pd.concat([pd.DataFrame(data_target), pd.DataFrame(data_ref)], ignore_index=True)
    
    spec = RegressionSpec(
        dependent_vars=["execution_time", "best_fitness"],
        robust_cov_types=["HC0", "HC1", "HC3"],
        apply_multiple_testing=True,
        multiple_testing_method="holm"
    )
    
    analyzer = OLSRegressionAnalyzer(spec)
    
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    
    pivot_df = analyzer.analyze_suite(
        df_global=df_global,
        ref_pipeline="ref",
        target_pipelines=["target"],
        analysis_dir=analysis_dir
    )
    
    # Verify outputs
    assert not pivot_df.empty
    assert len(pivot_df) == 2  # one for execution_time, one for best_fitness
    
    # Check that required columns are present
    expected_cols = [
        "Target", "Benchmark", "Dependent_Var", "R2",
        "beta_1 (Treatment)", "beta_1_pval",
        "beta_3 (Interaction)", "beta_3_pval",
        "beta_3_pval_HC0", "beta_3_pval_HC3",
        "beta_3_pval_holm"
    ]
    for col in expected_cols:
        assert col in pivot_df.columns
        
    # Check that markdown summaries were generated
    assert (analysis_dir / "target_vs_ref_sphere_execution_time_ols_summary.md").exists()
    assert (analysis_dir / "target_vs_ref_sphere_best_fitness_ols_summary.md").exists()
