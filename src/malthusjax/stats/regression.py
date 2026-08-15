import numpy as np

from malthusjax.stats.core import OLSResult, RegressionDataset
from malthusjax.stats.diagnostics import breusch_pagan, shapiro_wilk


def fit_ols(dataset: RegressionDataset) -> OLSResult:
    """Fit an Ordinary Least Squares regression model and run diagnostics.

    Uses robust standard errors (HC0, HC1, HC3) to handle heteroskedasticity.
    """
    import statsmodels.api as sm

    if not dataset.X:
        raise ValueError("RegressionDataset.X cannot be empty")

    y = np.asarray(dataset.y, dtype=float)
    if y.size == 0:
        raise ValueError("RegressionDataset.y cannot be empty")

    keys = list(dataset.X.keys())
    # Ensure shapes match
    for k in keys:
        if dataset.X[k].shape != y.shape:
            raise ValueError(
                f"Feature '{k}' shape {dataset.X[k].shape} does not match target shape {y.shape}"
            )

    # Build design matrix
    # Statsmodels expects 2D array for exog, shape (N, k).
    X_matrix = np.column_stack([np.asarray(dataset.X[k], dtype=float) for k in keys])
    sm.add_constant(X_matrix, has_constant="add")

    # Track the feature names in the model
    # statsmodels add_constant puts 'const' as the first column usually.
    # We will build a DataFrame to pass to sm.OLS so it tracks names correctly.
    import pandas as pd

    df_X = pd.DataFrame(X_matrix, columns=keys)
    df_X = sm.add_constant(df_X)

    model = sm.OLS(y, df_X).fit()

    # Calculate robust standard errors
    model_hc0 = sm.OLS(y, df_X).fit(cov_type="HC0")
    model_hc1 = sm.OLS(y, df_X).fit(cov_type="HC1")
    model_hc3 = sm.OLS(y, df_X).fit(cov_type="HC3")

    # Extract coefficients
    coefficients = {col: float(model.params[col]) for col in df_X.columns}
    p_values = {col: float(model.pvalues[col]) for col in df_X.columns}

    robust_p_values = {}
    for col in df_X.columns:
        if col == "const":
            continue
        robust_p_values[col] = {
            "HC0": float(model_hc0.pvalues[col]),
            "HC1": float(model_hc1.pvalues[col]),
            "HC3": float(model_hc3.pvalues[col]),
        }

    # Run diagnostics
    diagnostics = [breusch_pagan(model.resid, df_X.values), shapiro_wilk(model.resid)]

    return OLSResult(
        coefficients=coefficients,
        p_values=p_values,
        robust_p_values=robust_p_values,
        r_squared=float(model.rsquared),
        diagnostics=diagnostics,
        label=dataset.label,
    )
