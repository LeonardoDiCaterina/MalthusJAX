import numpy as np
import pandas as pd


def synthesize_regression_dataset(
    df_global: pd.DataFrame, target_pipeline: str, ref_pipeline: str
) -> pd.DataFrame:
    """Join target pipeline data with the reference pipeline to calculate relative effect.

    Assigns is_treatment=1 to target, is_treatment=0 to reference, and strictly pairs by fn_name, seed, D, P, G.
    """
    df_target = df_global[df_global["pipeline"] == target_pipeline].copy()
    df_ref = df_global[df_global["pipeline"] == ref_pipeline].copy()

    df_target["is_treatment"] = 1
    df_ref["is_treatment"] = 0

    common_keys = ["fn_name", "seed", "D", "P", "G"]
    merged = pd.merge(df_target[common_keys], df_ref[common_keys], on=common_keys, how="inner")

    df_target_paired = pd.merge(df_target, merged, on=common_keys, how="inner")
    df_ref_paired = pd.merge(df_ref, merged, on=common_keys, how="inner")

    return pd.concat([df_ref_paired, df_target_paired], ignore_index=True)


def build_design_matrix(
    df: pd.DataFrame, dependent_var: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply log-transforms, shifts, and mean-centering to build an OLS design matrix.

    Returns:
        y (np.ndarray): The transformed dependent variable.
        X (dict): Feature dictionary suitable for RegressionDataset.
    """
    df_clean = df.copy()

    # Target transformation
    if dependent_var == "execution_time":
        y = np.log(df_clean[dependent_var] + 1e-9)
    else:
        min_y = df_clean[dependent_var].min()
        shift = abs(min_y) + 1 if min_y <= 0 else 0
        y = np.log(df_clean[dependent_var] + shift)

    # Feature transformation
    log_D = np.log(df_clean["D"])
    log_P = np.log(df_clean["P"])
    log_G = np.log(df_clean["G"])
    is_treatment = df_clean["is_treatment"]

    # Mean-centering continuous predictors to ensure intercept and main effects
    # are interpretable at the average scale of the problem.
    log_D_centered = log_D - log_D.mean()
    log_P_centered = log_P - log_P.mean()
    log_G_centered = log_G - log_G.mean()

    # Interaction with centered D
    interaction = is_treatment * log_D_centered

    X = {
        "is_treatment": is_treatment.values,
        "log_D_centered": log_D_centered.values,
        "log_P_centered": log_P_centered.values,
        "log_G_centered": log_G_centered.values,
        "interaction_treatment_log_D": interaction.values,
    }

    return y.values, X
