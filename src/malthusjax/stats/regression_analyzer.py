import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from malthusjax.stats.core import RegressionDataset
from malthusjax.stats.dataset_builder import build_design_matrix, synthesize_regression_dataset
from malthusjax.stats.io import regression_to_markdown
from malthusjax.stats.regression import fit_ols


@dataclasses.dataclass
class RegressionSpec:
    """Specification for a suite of OLS regressions."""

    dependent_vars: list[str]
    robust_cov_types: list[str] = dataclasses.field(default_factory=lambda: ["HC0", "HC1", "HC3"])
    apply_multiple_testing: bool = True
    multiple_testing_method: str = "holm"


class OLSRegressionAnalyzer:
    """High-level analyzer for managing suites of OLS interaction regressions."""

    def __init__(self, spec: RegressionSpec):
        self.spec = spec

    def analyze_suite(
        self,
        df_global: pd.DataFrame,
        ref_pipeline: str,
        target_pipelines: list[str],
        analysis_dir: Path,
    ) -> pd.DataFrame:
        """Run OLS regressions for all targets and dependent variables, returning a summary DataFrame."""
        pivot_rows = []

        for target in target_pipelines:
            df_paired = synthesize_regression_dataset(
                df_global, target_pipeline=target, ref_pipeline=ref_pipeline
            )
            if df_paired.empty:
                continue

            for fn_name in df_paired["fn_name"].unique():
                df_fn = df_paired[df_paired["fn_name"] == fn_name]
                prefix = f"{target}_vs_{ref_pipeline}_{fn_name}"

                # Check if it's LHS mode (D varies) or just a single point
                if len(df_fn["D"].unique()) <= 1:
                    continue  # OLS scaling requires variance in D

                for var in self.spec.dependent_vars:
                    y_vals, X_dict = build_design_matrix(df_fn, dependent_var=var)

                    dataset = RegressionDataset(y=y_vals, X=X_dict, label=f"{prefix}_{var}")

                    try:
                        ols_res = fit_ols(dataset)

                        # Generate markdown diagnostic output
                        ols_res.target_name = dataset.label  # type: ignore[attr-defined]
                        ols_res.n_observations = len(y_vals)  # type: ignore[attr-defined]
                        ols_res.adjusted_r_squared = ols_res.r_squared  # type: ignore[attr-defined]
                        ols_res.features = list(dataset.X.keys())  # type: ignore[attr-defined]
                        ols_res.standard_errors = {}  # type: ignore[attr-defined]
                        ols_res.t_values = {}  # type: ignore[attr-defined]

                        md_path = analysis_dir / f"{prefix}_{var}_ols_summary.md"
                        md_path.write_text(regression_to_markdown(ols_res))

                        # Extract p-values and coefficients
                        bp_pval = next(
                            (d.p_value for d in ols_res.diagnostics if d.name == "Breusch-Pagan"),
                            np.nan,
                        )
                        sw_pval = next(
                            (d.p_value for d in ols_res.diagnostics if d.name == "Shapiro-Wilk"),
                            np.nan,
                        )

                        row = {
                            "Target": target,
                            "Benchmark": fn_name,
                            "Dependent_Var": var,
                            "R2": ols_res.r_squared,
                            "beta_1 (Treatment)": ols_res.coefficients.get("is_treatment", np.nan),
                            "beta_1_pval": ols_res.p_values.get("is_treatment", np.nan),
                            "beta_3 (Interaction)": ols_res.coefficients.get(
                                "interaction_treatment_log_D", np.nan
                            ),
                            "beta_3_pval": ols_res.p_values.get(
                                "interaction_treatment_log_D", np.nan
                            ),
                        }

                        # Add robust p-values for the interaction term
                        for hc in self.spec.robust_cov_types:
                            robust_pval = ols_res.robust_p_values.get(
                                "interaction_treatment_log_D", {}
                            ).get(hc, np.nan)
                            row[f"beta_3_pval_{hc}"] = robust_pval

                        row["BP_pval"] = bp_pval
                        row["SW_pval"] = sw_pval

                        pivot_rows.append(row)

                    except Exception as e:
                        print(f"Warning: Failed to run OLS for {prefix}_{var}: {e}")

        df_pivot = pd.DataFrame(pivot_rows)

        if not df_pivot.empty and self.spec.apply_multiple_testing:
            p_cols = ["beta_3_pval"] + [f"beta_3_pval_{hc}" for hc in self.spec.robust_cov_types]
            for pval_col in p_cols:
                if pval_col in df_pivot.columns:
                    valid_idx = df_pivot[pval_col].notna()
                    if valid_idx.any():
                        _, corrected, _, _ = multipletests(
                            df_pivot.loc[valid_idx, pval_col],
                            method=self.spec.multiple_testing_method,
                        )
                        df_pivot.loc[valid_idx, f"{pval_col}_holm"] = corrected

            # Reorder columns to ensure Target, Benchmark, Dependent_Var are first
            cols = ["Target", "Benchmark", "Dependent_Var"] + [
                c for c in df_pivot.columns if c not in ["Target", "Benchmark", "Dependent_Var"]
            ]
            df_pivot = df_pivot[cols]

        return df_pivot
