import numpy as np
import pandas as pd
import pytest

from malthusjax.stats.dataset_builder import synthesize_regression_dataset, build_design_matrix

def test_synthesize_regression_dataset():
    # Setup mock global dataframe
    data = {
        "fn_name": ["sphere", "sphere", "sphere", "sphere"],
        "seed": [1, 1, 2, 2],
        "D": [10, 10, 20, 20],
        "P": [100, 100, 200, 200],
        "G": [50, 50, 100, 100],
        "pipeline": ["target", "ref", "target", "ref"],
        "best_fitness": [0.1, 0.2, 0.5, 0.8],
        "execution_time": [1.0, 1.5, 2.0, 2.5]
    }
    df_global = pd.DataFrame(data)
    
    # Run function
    df_paired = synthesize_regression_dataset(df_global, "target", "ref")
    
    assert len(df_paired) == 4
    assert set(df_paired["is_treatment"].unique()) == {0, 1}
    assert (df_paired[df_paired["pipeline"] == "target"]["is_treatment"] == 1).all()
    assert (df_paired[df_paired["pipeline"] == "ref"]["is_treatment"] == 0).all()


def test_build_design_matrix_mean_centering():
    # Setup mock dataframe with varied continuous predictors
    data = {
        "fn_name": ["sphere"] * 4,
        "seed": [1, 2, 3, 4],
        "D": [10, 100, 10, 100],
        "P": [10, 100, 10, 100],
        "G": [10, 100, 10, 100],
        "pipeline": ["target", "target", "ref", "ref"],
        "is_treatment": [1, 1, 0, 0],
        "best_fitness": [1.0, 2.0, 3.0, 4.0],
        "execution_time": [1.0, 2.0, 3.0, 4.0]
    }
    df = pd.DataFrame(data)
    
    y, X = build_design_matrix(df, "execution_time")
    
    # Assert continuous variables were log-transformed and mean-centered
    expected_log_D = np.log([10, 100, 10, 100])
    mean_log_D = np.mean(expected_log_D)
    expected_centered_D = expected_log_D - mean_log_D
    
    np.testing.assert_allclose(X["log_D_centered"], expected_centered_D)
    
    # Assert mean is zero (within numerical precision)
    assert np.isclose(np.mean(X["log_D_centered"]), 0.0, atol=1e-7)
    assert np.isclose(np.mean(X["log_P_centered"]), 0.0, atol=1e-7)
    assert np.isclose(np.mean(X["log_G_centered"]), 0.0, atol=1e-7)
    
    # Assert interaction term is calculated correctly
    expected_interaction = df["is_treatment"].values * expected_centered_D
    np.testing.assert_allclose(X["interaction_treatment_log_D"], expected_interaction)

