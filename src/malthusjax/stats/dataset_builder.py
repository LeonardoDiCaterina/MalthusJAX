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
    if "replicate_id" in df_global.columns:
        common_keys.append("replicate_id")
    for frame in (df_target, df_ref):
        if frame[common_keys].isna().any().any() or frame.duplicated(common_keys).any():
            raise ValueError("Ambiguous or missing observation identity; use unique replicate_id values")
    matched = df_target[common_keys].merge(df_ref[common_keys], on=common_keys, validate="one_to_one")
    target = df_target.merge(matched,on=common_keys,validate="one_to_one")
    ref = df_ref.merge(matched,on=common_keys,validate="one_to_one")
    result = pd.concat([ref,target], ignore_index=True)
    result.attrs["pairing"] = {"keys":common_keys,"matched_pairs":len(matched),
        "target_excluded":len(df_target)-len(target),"reference_excluded":len(df_ref)-len(ref)}
    return result


def build_design_matrix(
    df: pd.DataFrame, dependent_var: str, *, transform: str | None = None,
    optimum: float | None = None, epsilon: float | None = None, maximize: bool = False
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply log-transforms, shifts, and mean-centering to build an OLS design matrix.

    Returns:
        y (np.ndarray): The transformed dependent variable.
        X (dict): Feature dictionary suitable for RegressionDataset.
    """
    df_clean = df.copy()

    values = np.asarray(df_clean[dependent_var], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dependent values must be finite")
    transform = transform or ("log" if dependent_var == "execution_time" else "raw")
    if transform == "log_gap":
        if optimum is None or not np.isfinite(optimum) or epsilon is None or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("log_gap requires a finite optimum and positive declared epsilon")
        gap = optimum - values if maximize else values - optimum
        if np.any(gap < 0):
            raise ValueError("Negative optimality gaps; check objective orientation/optimum")
        y = np.log(gap + epsilon)
    elif transform == "log":
        if np.any(values <= 0):
            raise ValueError("Log transform requires strictly positive values")
        y = np.log(values)
    elif transform == "raw":
        y = values
    else:
        raise ValueError("Unknown dependent-variable transform")
    if not np.isfinite(df_clean[["D","P","G","is_treatment"]].to_numpy(dtype=float)).all() or (df_clean[["D","P","G"]] <= 0).any().any():
        raise ValueError("Predictors require finite positive dimensions, populations and horizons")
    df.attrs["regression_transform"] = {"dependent_var":dependent_var,"transform":transform,
        "optimum":optimum,"epsilon":epsilon,"maximize":maximize,"units":"seconds" if dependent_var == "execution_time" else "reported_objective"}

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

    return np.asarray(y), X
