import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from malthusjax.core.logger import get_logger
from malthusjax.stats.core import RegressionDataset
from malthusjax.stats.dataset_builder import build_design_matrix, synthesize_regression_dataset
from malthusjax.stats.io import regression_to_markdown
from malthusjax.stats.regression import fit_ols

logger = get_logger("stats.regression")


@dataclasses.dataclass
class RegressionSpec:
    """Specification for a suite of OLS regressions."""

    dependent_vars: list[str]
    robust_cov_types: list[str] = dataclasses.field(default_factory=lambda: ["HC0", "HC1", "HC3"])
    apply_multiple_testing: bool = True
    multiple_testing_method: str = "holm"
    transforms: dict = dataclasses.field(default_factory=dict)
    covariance: str = "HC3"
    cluster_by: list[str] | None = None


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
                    y_vals, X_dict = build_design_matrix(df_fn, dependent_var=var, **self.spec.transforms.get(var, {}))

                    dataset = RegressionDataset(y=y_vals, X=X_dict, label=f"{prefix}_{var}",
                        covariance=self.spec.covariance,
                        groups=(pd.factorize(pd.MultiIndex.from_frame(df_fn[self.spec.cluster_by]))[0] if self.spec.cluster_by else None),
                        metadata={**df_fn.attrs,"observation_ids": df_fn[["fn_name","seed","D","P","G","is_treatment"]].to_dict("records")})

                    try:
                        ols_res = fit_ols(dataset)

                        # Generate markdown diagnostic output
                        (analysis_dir / f"{prefix}_{var}_ols_result.json").write_text(json.dumps(dataclasses.asdict(ols_res), indent=2))
                        md_path = analysis_dir / f"{prefix}_{var}_ols_summary.md"
                        md_path.write_text(regression_to_markdown(ols_res))

                        # Extract p-values and coefficients (support both snake_case and Title-Case diagnostic names)
                        bp_pval = next(
                            (
                                d.p_value
                                for d in ols_res.diagnostics
                                if d.name.lower().replace("-", "_") == "breusch_pagan"
                                and d.p_value is not None
                            ),
                            np.nan,
                        )
                        sw_pval = next(
                            (
                                d.p_value
                                for d in ols_res.diagnostics
                                if d.name.lower().replace("-", "_") == "shapiro_wilk"
                                and d.p_value is not None
                            ),
                            np.nan,
                        )

                        row = {
                            "Target": target,
                            "Benchmark": fn_name,
                            "Dependent_Var": var,
                            "R2": ols_res.r_squared,
                            "Adjusted_R2": ols_res.adjusted_r_squared,
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
                        logger.warning("Failed to run OLS for %s_%s: %s", prefix, var, e)

        df_pivot = pd.DataFrame(pivot_rows)

        if not df_pivot.empty and self.spec.apply_multiple_testing:
            p_cols = ["beta_3_pval"] + [f"beta_3_pval_{hc}" for hc in self.spec.robust_cov_types]
            for pval_col in p_cols:
                if pval_col in df_pivot.columns:
                    df_pivot[f"{pval_col}_holm"] = np.nan
                    for _, group_df in df_pivot.groupby("Dependent_Var"):
                        valid_idx = group_df[pval_col].dropna().index
                        if len(valid_idx) > 0:
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
