#!/usr/bin/env python3
"""Unified TOML-Driven Benchmark Analyzer.

Parses the raw JSON outputs of the `benchmark_runner.py` and performs
advanced statistical analysis using the `malthusjax.stats` package.
- Cartesian Mode: TOST Equivalence, Wilcoxon Signed-Rank, Cohen's d_z via `StatisticalComparator`.
- LHS Mode: OLS Log-Log Interaction Regression with Diagnostics via `fit_ols`.

Generates publication-ready LaTeX tables, Markdown summaries, and automated
plots (Convergence, Scaling Laws, Boxplots).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.stats.multitest import multipletests

from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.stats.core import (
    HypothesisKind,
    Sidedness,
    ExpectedDirection,
    MultipleTestingPolicy,
    RegressionDataset,
    StatisticalComparisonSpec,
    PairedMetricDataset,
)
from malthusjax.stats.regression import fit_ols
from malthusjax.stats.comparator import StatisticalComparator
from malthusjax.stats.io import suite_to_markdown, suite_to_dict, regression_to_markdown


def parse_global_data(results_dir: Path) -> pd.DataFrame:
    """Parse all JSON artifacts into a single unpivoted dataframe."""
    records = []
    
    # Support both new TOML engine and legacy hardcoded script outputs
    search_patterns = ["benchmark_results.json", "parity_results.json", "ablation_results.json", "representation_results.json"]
    
    for pattern in search_patterns:
        for json_path in results_dir.rglob(pattern):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
                
            exp_name = data.get("experiment", "unknown")
            config = data.get("config", data)
            
            fn_name = config.get("fn_name", config.get("function", "unknown"))
            D = config.get("D", config.get("dimensions", np.nan))
            P = config.get("P", config.get("population_size", np.nan))
            G = config.get("G", config.get("generations", np.nan))
            
            # Legacy Fallback
            if pd.isna(D) or pd.isna(P):
                import re
                if pd.isna(D):
                    m = re.search(r'_d(\d+)_', exp_name)
                    if m: D = float(m.group(1))
                if pd.isna(P):
                    m = re.search(r'_p(\d+)_', exp_name)
                    if m: P = float(m.group(1))
                if pd.isna(G):
                    m = re.search(r'_g(\d+)', exp_name)
                    if m: G = float(m.group(1))
            
            pipelines = data.get("pipelines", {})
            for p_name, p_data in pipelines.items():
                runs = p_data.get("per_seed", p_data)
                if not isinstance(runs, list):
                    runs = [runs]
                    
                for run in runs:
                    seed = run.get("seed", -1)
                    best_fit = run.get("best_fitness", np.nan)
                    exec_time = run.get("duration_seconds", np.nan)
                    
                    if "timings" in run and "total" in run["timings"]:
                        exec_time = run["timings"]["total"]
                    
                    try:
                        best_fit = float(best_fit) if best_fit is not None else np.nan
                        exec_time = float(exec_time) if exec_time is not None else np.nan
                    except (ValueError, TypeError):
                        best_fit, exec_time = np.nan, np.nan
                    
                    if np.isnan(best_fit) or np.isnan(exec_time):
                        continue
                        
                    records.append({
                        "experiment": exp_name,
                        "fn_name": fn_name,
                        "D": D,
                        "P": P,
                        "G": G,
                        "pipeline": p_name,
                        "seed": seed,
                        "best_fitness": best_fit,
                        "execution_time": exec_time,
                    })
                
    return pd.DataFrame(records)


def synthesize_regression_dataset(df_global: pd.DataFrame, target_pipeline: str, ref_pipeline: str) -> pd.DataFrame:
    """Join target pipeline data with the reference pipeline to calculate relative effect."""
    df_target = df_global[df_global["pipeline"] == target_pipeline].copy()
    df_ref = df_global[df_global["pipeline"] == ref_pipeline].copy()
    
    df_target["is_treatment"] = 1
    df_ref["is_treatment"] = 0
    
    common_keys = ["fn_name", "seed", "D", "P", "G"]
    merged = pd.merge(df_target[common_keys], df_ref[common_keys], on=common_keys, how="inner")
    
    df_target_paired = pd.merge(df_target, merged, on=common_keys, how="inner")
    df_ref_paired = pd.merge(df_ref, merged, on=common_keys, how="inner")
    
    return pd.concat([df_ref_paired, df_target_paired], ignore_index=True)


def generate_boxplots(df: pd.DataFrame, output_dir: Path, prefix: str):
    """Generate side-by-side boxplots for speed and fitness."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="pipeline", y="execution_time")
    plt.yscale("log")
    plt.title("Execution Time (Seconds)")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x="pipeline", y="best_fitness")
    plt.title("Best Fitness (Lower is Better)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_boxplots.png", dpi=300)
    plt.close()


def generate_scaling_plots(df: pd.DataFrame, dependent_var: str, output_dir: Path, prefix: str):
    """Generate Log-Log scaling scatter plots."""
    plt.figure(figsize=(8, 6))
    
    df_clean = df.copy()
    if dependent_var == "execution_time":
        df_clean["log_Y"] = np.log(df_clean[dependent_var] + 1e-9)
        title_y = "Log Execution Time"
    else:
        min_y = df_clean[dependent_var].min()
        shift = abs(min_y) + 1 if min_y <= 0 else 0
        df_clean["log_Y"] = np.log(df_clean[dependent_var] + shift)
        title_y = "Log Fitness"
        
    df_clean["log_D"] = np.log(df_clean["D"])
    
    sns.lmplot(data=df_clean, x="log_D", y="log_Y", hue="pipeline", height=6, aspect=1.2, scatter_kws={'alpha':0.5})
    plt.title(f"Scaling Law: {dependent_var.replace('_', ' ').title()}")
    plt.xlabel("Log Dimensionality (ln D)")
    plt.ylabel(title_y)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_{dependent_var}_scaling.png", dpi=300)
    plt.close("all")


def export_latex_safe(df: pd.DataFrame, filepath: Path):
    """Safely export a DataFrame to LaTeX, falling back to manual string building if jinja2 is missing."""
    try:
        df.to_latex(filepath, index=False, float_format="%.4e")
    except ImportError:
        with open(filepath, "w") as f:
            f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n")
            f.write("\\toprule\n")
            f.write(" & ".join([str(c).replace('_', '\\_') for c in df.columns]) + " \\\\\n")
            f.write("\\midrule\n")
            for _, row in df.iterrows():
                formatted = []
                for val in row:
                    if isinstance(val, float):
                        formatted.append(f"{val:.4e}")
                    else:
                        formatted.append(str(val).replace('_', '\\_'))
                f.write(" & ".join(formatted) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")


def analyze_suite(toml_path: str, data_dir_override: str = None):
    config = BenchmarkConfig.from_toml(toml_path)
    
    if data_dir_override:
        results_dir = Path(data_dir_override)
    else:
        results_dir = Path(config.suite.output_dir)
        
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing raw results from {results_dir}...")
    df_global = parse_global_data(results_dir)
    
    if df_global.empty:
        print("ERROR: No valid JSON result artifacts found!")
        sys.exit(1)
        
    print(f"Loaded {len(df_global)} individual traces.")
    
    df_global = df_global.drop_duplicates(subset=["fn_name", "D", "P", "G", "pipeline", "seed"], keep="last")
    print(f"Deduplicated to {len(df_global)} unique traces.")
    df_global.to_csv(analysis_dir / "unpivoted_raw_data.csv", index=False)
    
    ref_pipeline = config.analysis.reference_pipeline
    target_pipelines = [p for p in config.pipelines.keys() if p != ref_pipeline]
    
    if not ref_pipeline or ref_pipeline not in config.pipelines:
        print(f"WARNING: Reference pipeline '{ref_pipeline}' not found in TOML.")
        return

    pivot_rows = []
    parity_rows = []
    comparator = StatisticalComparator()
    
    for target in target_pipelines:
        print(f"\nAnalyzing: {target} vs {ref_pipeline}")
        df_paired = synthesize_regression_dataset(df_global, target_pipeline=target, ref_pipeline=ref_pipeline)
        
        if df_paired.empty:
            print("  -> No strictly paired traces found. Skipping.")
            continue
            
        print(f"  -> Synthesized {len(df_paired)//2} strictly paired coordinate traces.")
        
        for fn_name in df_paired["fn_name"].unique():
            df_fn = df_paired[df_paired["fn_name"] == fn_name]
            prefix = f"{target}_vs_{ref_pipeline}_{fn_name}"
            
            generate_boxplots(df_fn, analysis_dir, prefix)
            
            if config.suite.mode == "lhs" or len(df_fn["D"].unique()) > 1 and config.suite.mode != "cartesian":
                # OLS Scaling Regressions using malthusjax.stats
                generate_scaling_plots(df_fn, "execution_time", analysis_dir, prefix)
                generate_scaling_plots(df_fn, "best_fitness", analysis_dir, prefix)
                
                for var in ["execution_time", "best_fitness"]:
                    df_clean = df_fn.copy()
                    if var == "execution_time":
                        y = np.log(df_clean[var] + 1e-9)
                    else:
                        min_y = df_clean[var].min()
                        shift = abs(min_y) + 1 if min_y <= 0 else 0
                        y = np.log(df_clean[var] + shift)
                    
                    log_D = np.log(df_clean["D"])
                    log_P = np.log(df_clean["P"])
                    log_G = np.log(df_clean["G"])
                    is_treatment = df_clean["is_treatment"]
                    interaction = is_treatment * log_D
                    
                    dataset = RegressionDataset(
                        y=y.values,
                        X={
                            "is_treatment": is_treatment.values,
                            "log_D": log_D.values,
                            "log_P": log_P.values,
                            "log_G": log_G.values,
                            "interaction_term": interaction.values,
                        },
                        label=f"{prefix}_{var}"
                    )
                    
                    # Convert to OLSResult wrapper via new package
                    try:
                        ols_res = fit_ols(dataset)
                        
                        # Generate markdown diagnostic output
                        ols_res.target_name = dataset.label
                        ols_res.n_observations = len(y)
                        ols_res.adjusted_r_squared = ols_res.r_squared
                        ols_res.features = list(dataset.X.keys())
                        ols_res.standard_errors = {}
                        ols_res.t_values = {}
                        
                        with open(analysis_dir / f"{prefix}_{var}_ols_summary.md", "w") as f:
                            f.write(regression_to_markdown(ols_res))
                        
                        bp_pval = next((d.p_value for d in ols_res.diagnostics if d.name == "Breusch-Pagan"), np.nan)
                        sw_pval = next((d.p_value for d in ols_res.diagnostics if d.name == "Shapiro-Wilk"), np.nan)
                        
                        pivot_rows.append({
                            "Target": target,
                            "Benchmark": fn_name,
                            "Dependent_Var": var,
                            "R2": ols_res.r_squared,
                            "beta_1 (Treatment)": ols_res.coefficients.get("is_treatment", np.nan),
                            "beta_1_pval": ols_res.p_values.get("is_treatment", np.nan),
                            "beta_3 (Interaction)": ols_res.coefficients.get("interaction_term", np.nan),
                            "beta_3_pval": ols_res.p_values.get("interaction_term", np.nan),
                            "beta_3_pval_HC0": ols_res.robust_p_values.get("interaction_term", {}).get("HC0", np.nan),
                            "beta_3_pval_HC1": ols_res.robust_p_values.get("interaction_term", {}).get("HC1", np.nan),
                            "beta_3_pval_HC3": ols_res.robust_p_values.get("interaction_term", {}).get("HC3", np.nan),
                            "BP_pval": bp_pval,
                            "SW_pval": sw_pval,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to run OLS for {prefix}_{var}: {e}")
            else:
                # Cartesian Parity using malthusjax.stats
                print(f"  -> {fn_name}: Cartesian run detected. Using malthusjax.stats.comparator for TOST and Wilcoxon.")
                
                # Split and test per Dimensionality
                for D_val in df_fn["D"].unique():
                    df_D = df_fn[df_fn["D"] == D_val]
                    df_target = df_D[df_D["is_treatment"] == 1].sort_values("seed")
                    df_ref = df_D[df_D["is_treatment"] == 0].sort_values("seed")
                    
                    prefix_D = f"{prefix}_d{int(D_val)}"
                    
                    datasets = []
                    for var in ["execution_time", "best_fitness"]:
                        target_vals = df_target[var].values
                        ref_vals = df_ref[var].values
                        
                        if len(target_vals) > 0 and len(target_vals) == len(ref_vals):
                            datasets.append(PairedMetricDataset(
                                label=f"{prefix_D}_{var}",
                                left_name=target,
                                right_name=ref_pipeline,
                                seeds=df_target["seed"].tolist(),
                                left_values=target_vals,
                                right_values=ref_vals,
                                metric_name=var,
                                metric_source="summary",
                                metadata={"D": float(D_val)}
                            ))
                    
                    if datasets:
                        for ds in datasets:
                            # Configure spec dynamically based on metric
                            if ds.metric_name == "best_fitness":
                                ref_std = np.std(ds.right_values, ddof=1)
                                delta = max(1e-9, 0.2 * ref_std)
                                spec = StatisticalComparisonSpec(
                                    metric_name="best_fitness",
                                    hypothesis_kind=HypothesisKind.EQUIVALENCE,
                                    equivalence_margin=delta,
                                    multiple_testing=MultipleTestingPolicy.NONE,
                                    include_tests=("wilcoxon", "paired_t"),
                                    include_mean_summary=True
                                )
                            else:
                                spec = StatisticalComparisonSpec(
                                    metric_name="execution_time",
                                    hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
                                    sidedness=Sidedness.TWO_SIDED,
                                    multiple_testing=MultipleTestingPolicy.NONE,
                                    include_tests=("wilcoxon", "paired_t"),
                                    include_mean_summary=True
                                )
                                
                            # Compare single dataset directly to allow per-metric specs
                            suite_res = comparator.compare_suite([ds], spec)
                            
                            # Export JSON and Markdown for the metric suite
                            with open(analysis_dir / f"{prefix_D}_{ds.metric_name}_suite.json", "w") as f:
                                json.dump(suite_to_dict(suite_res), f, indent=2)
                            with open(analysis_dir / f"{prefix_D}_{ds.metric_name}_suite.md", "w") as f:
                                f.write(suite_to_markdown(suite_res))
                                
                            # Extract metrics for the legacy aggregated CSV table
                            res = suite_res.results[0]
                            wilcox_pval = res.tests["wilcoxon"].p_value if "wilcoxon" in res.tests else np.nan
                            tost_pval = res.tost.p_value_max if res.tost else np.nan
                            tost_delta = res.tost.margin if res.tost else np.nan
                            
                            parity_rows.append({
                                "Target": target,
                                "Benchmark": fn_name,
                                "Metric": ds.metric_name,
                                "D": ds.metadata.get("D", np.nan),
                                "Target_Mean": res.left_mean,
                                "Target_Std": np.std(ds.left_values, ddof=1),
                                "Ref_Mean": res.right_mean,
                                "Ref_Std": np.std(ds.right_values, ddof=1),
                                "Wilcoxon_pval": wilcox_pval,
                                "TOST_pval": tost_pval,
                                "TOST_Delta": tost_delta,
                                "Cohen_dz": res.effects.cohen_dz
                            })

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        for pval_col in ["beta_3_pval", "beta_3_pval_HC0", "beta_3_pval_HC1", "beta_3_pval_HC3"]:
            if pval_col in pivot_df.columns:
                valid_idx = pivot_df[pval_col].notna()
                if valid_idx.any():
                    _, corrected, _, _ = multipletests(pivot_df.loc[valid_idx, pval_col], method='holm')
                    pivot_df.loc[valid_idx, f"{pval_col}_holm"] = corrected

        cols = ["Target", "Benchmark", "Dependent_Var"] + [c for c in pivot_df.columns if c not in ["Target", "Benchmark", "Dependent_Var"]]
        pivot_df = pivot_df[cols]
        
        pivot_file = analysis_dir / "ols_regression_table.csv"
        pivot_df.to_csv(pivot_file, index=False)
        export_latex_safe(pivot_df, analysis_dir / "ols_regression_table.tex")
        print(f"\nOLS Regression Pivot Table saved to {pivot_file}")

    if parity_rows:
        parity_df = pd.DataFrame(parity_rows)
        for pval_col in ["Wilcoxon_pval", "TOST_pval"]:
            if pval_col in parity_df.columns:
                valid_idx = parity_df[pval_col].notna()
                if valid_idx.any():
                    _, corrected, _, _ = multipletests(parity_df.loc[valid_idx, pval_col], method='holm')
                    parity_df.loc[valid_idx, f"{pval_col}_holm"] = corrected

        cols = ["Target", "Benchmark", "Metric", "D"] + [c for c in parity_df.columns if c not in ["Target", "Benchmark", "Metric", "D"]]
        parity_df = parity_df[cols]
        
        parity_file = analysis_dir / "parity_wilcoxon_table.csv"
        parity_df.to_csv(parity_file, index=False)
        export_latex_safe(parity_df, analysis_dir / "parity_wilcoxon_table.tex")
        print(f"\nNon-Parametric Parity Table saved to {parity_file}")

    print(f"\nAnalysis complete! Artifacts dumped to {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified TOML-Driven Benchmark Analyzer")
    parser.add_argument("--toml", type=str, required=True, help="Path to the TOML configuration file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override the results directory")
    args = parser.parse_args()
    
    analyze_suite(args.toml, args.data_dir)


if __name__ == "__main__":
    main()
