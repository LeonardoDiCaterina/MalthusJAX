#!/usr/bin/env python3
"""Unified TOML-Driven Benchmark Analyzer.

Parses the raw JSON outputs of the `benchmark_runner.py` and performs
advanced statistical analysis:
- Cartesian Mode: TOST, Wilcoxon Signed-Rank, Cohen's d_z.
- LHS Mode: OLS Log-Log Interaction Regression with Diagnostics (Breusch-Pagan, Shapiro-Wilk).

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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests

from malthusjax.benchmarking.config import BenchmarkConfig


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
            
            # Support both new TOML format (in `config`) and legacy format (top-level)
            config = data.get("config", data)
            
            fn_name = config.get("fn_name", config.get("function", "unknown"))
            D = config.get("D", config.get("dimensions", np.nan))
            P = config.get("P", config.get("population_size", np.nan))
            G = config.get("G", config.get("generations", np.nan))
            
            # Legacy Fallback: Extract from experiment string if missing
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
                
                # Support old legacy format where runs were directly in a list instead of nested under "per_seed"
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


def run_ols_diagnostics(df: pd.DataFrame, dependent_var: str, output_dir: Path, prefix: str) -> dict:
    """Run OLS and diagnostic tests on synthesized paired datasets."""
    if len(df) < 2:
        return {}
        
    df_clean = df.copy()
    if dependent_var == "execution_time":
        df_clean["log_Y"] = np.log(df_clean[dependent_var] + 1e-9)
    else:
        # Avoid log of negative fitness by shifting
        min_y = df_clean[dependent_var].min()
        shift = abs(min_y) + 1 if min_y <= 0 else 0
        df_clean["log_Y"] = np.log(df_clean[dependent_var] + shift)
        
    df_clean["log_D"] = np.log(df_clean["D"])
    df_clean["log_P"] = np.log(df_clean["P"])
    df_clean["log_G"] = np.log(df_clean["G"])
    df_clean["interaction_term"] = df_clean["is_treatment"] * df_clean["log_D"]
    
    X = df_clean[["is_treatment", "log_D", "log_P", "log_G", "interaction_term"]]
    X = sm.add_constant(X)
    y = df_clean["log_Y"]
    
    model = sm.OLS(y, X).fit()
    model_hc0 = sm.OLS(y, X).fit(cov_type='HC0')
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')
    model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
    
    bp_pval = np.nan
    try:
        _, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    except Exception: pass
    
    sw_pval = np.nan
    resid = model.resid.values
    if len(resid) > 5000:
        resid = np.random.choice(resid, 5000, replace=False)
    try:
        _, sw_pval = stats.shapiro(resid)
    except Exception: pass
    
    # Save full text summary
    with open(output_dir / f"{prefix}_{dependent_var}_ols_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f"\n\n--- Diagnostics ---\n")
        f.write(f"Breusch-Pagan p-value (Heteroskedasticity): {bp_pval:.4e}\n")
        f.write(f"Shapiro-Wilk p-value (Normality): {sw_pval:.4e}\n")
        f.write(f"\n--- Robust Standard Errors (Interaction Term) ---\n")
        f.write(f"Standard p-value: {model.pvalues.get('interaction_term', np.nan):.4e}\n")
        f.write(f"HC0 p-value:      {model_hc0.pvalues.get('interaction_term', np.nan):.4e}\n")
        f.write(f"HC1 p-value:      {model_hc1.pvalues.get('interaction_term', np.nan):.4e}\n")
        f.write(f"HC3 p-value:      {model_hc3.pvalues.get('interaction_term', np.nan):.4e}\n")
        
    return {
        "Dependent_Var": dependent_var,
        "R2": model.rsquared,
        "beta_1 (Treatment)": model.params.get("is_treatment", np.nan),
        "beta_1_pval": model.pvalues.get("is_treatment", np.nan),
        "beta_3 (Interaction)": model.params.get("interaction_term", np.nan),
        "beta_3_pval": model.pvalues.get("interaction_term", np.nan),
        "beta_3_pval_HC0": model_hc0.pvalues.get("interaction_term", np.nan),
        "beta_3_pval_HC1": model_hc1.pvalues.get("interaction_term", np.nan),
        "beta_3_pval_HC3": model_hc3.pvalues.get("interaction_term", np.nan),
        "BP_pval": bp_pval,
        "SW_pval": sw_pval,
    }


def generate_boxplots(df: pd.DataFrame, output_dir: Path, prefix: str):
    """Generate side-by-side boxplots for speed and fitness."""
    plt.figure(figsize=(12, 5))
    
    # Plot Execution Time
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="pipeline", y="execution_time")
    plt.yscale("log")
    plt.title("Execution Time (Seconds)")
    plt.xticks(rotation=45)
    
    # Plot Fitness
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
        # Fallback manual LaTeX generator for clusters without jinja2
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
    
    # CRITICAL: Deduplicate to prevent pd.merge Cartesian explosion if the user ran the same config multiple times
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
            
            # Generate Visuals
            generate_boxplots(df_fn, analysis_dir, prefix)
            
            # Run OLS Scaling if we have multidimensional data
            if len(df_fn["D"].unique()) > 1:
                generate_scaling_plots(df_fn, "execution_time", analysis_dir, prefix)
                generate_scaling_plots(df_fn, "best_fitness", analysis_dir, prefix)
                
                res_time = run_ols_diagnostics(df_fn, "execution_time", analysis_dir, prefix)
                res_fit = run_ols_diagnostics(df_fn, "best_fitness", analysis_dir, prefix)
                
                if res_time:
                    res_time.update({"Target": target, "Benchmark": fn_name})
                    pivot_rows.append(res_time)
                if res_fit:
                    res_fit.update({"Target": target, "Benchmark": fn_name})
                    pivot_rows.append(res_fit)
            else:
                # If we only have 1 dimension (Cartesian Parity Control), run Non-Parametric Parity Tests
                print(f"  -> {fn_name}: Only 1 dimension detected. Skipping OLS Scaling, running Wilcoxon Parity Tests.")
                df_target = df_fn[df_fn["is_treatment"] == 1].sort_values("seed")
                df_ref = df_fn[df_fn["is_treatment"] == 0].sort_values("seed")
                
                for var in ["execution_time", "best_fitness"]:
                    target_vals = df_target[var].values
                    ref_vals = df_ref[var].values
                    
                    if len(target_vals) > 0 and len(target_vals) == len(ref_vals):
                        try:
                            # Wilcoxon Signed-Rank Test (Non-Parametric Location Shift)
                            _, wilcox_pval = stats.wilcoxon(target_vals, ref_vals)
                            # Cohen's dz (Effect Size)
                            diffs = target_vals - ref_vals
                            ref_std = np.std(ref_vals, ddof=1)
                            dz = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-9)
                            
                            # TOST Equivalence Test (Delta = 0.2 * SD_ref)
                            delta = 0.2 * ref_std
                            _, p_upper = stats.ttest_1samp(diffs, popmean=delta, alternative='less')
                            _, p_lower = stats.ttest_1samp(diffs, popmean=-delta, alternative='greater')
                            tost_pval = max(p_upper, p_lower)
                        except Exception:
                            wilcox_pval, dz, delta, tost_pval = np.nan, np.nan, np.nan, np.nan
                            
                        parity_rows.append({
                            "Target": target,
                            "Benchmark": fn_name,
                            "Metric": var,
                            "D": df_fn["D"].iloc[0],
                            "Target_Mean": np.mean(target_vals),
                            "Target_Std": np.std(target_vals),
                            "Ref_Mean": np.mean(ref_vals),
                            "Ref_Std": np.std(ref_vals),
                            "Wilcoxon_pval": wilcox_pval,
                            "TOST_pval": tost_pval,
                            "TOST_Delta": delta,
                            "Cohen_dz": dz
                        })

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        
        # Apply Holm-Bonferroni correction
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
        print(f"\nOLS Regression Pivot Table saved to {pivot_file}")
        
        tex_file = analysis_dir / "ols_regression_table.tex"
        export_latex_safe(pivot_df, tex_file)
        print(f"LaTeX Table exported to {tex_file}")

    if parity_rows:
        parity_df = pd.DataFrame(parity_rows)
        
        # Apply Holm-Bonferroni correction
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
        print(f"\nNon-Parametric Parity Table saved to {parity_file}")
        
        tex_file = analysis_dir / "parity_wilcoxon_table.tex"
        export_latex_safe(parity_df, tex_file)
        print(f"LaTeX Table exported to {tex_file}")

    print(f"\nAnalysis complete! Artifacts dumped to {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified TOML-Driven Benchmark Analyzer")
    parser.add_argument("--toml", type=str, required=True, help="Path to the TOML configuration file")
    parser.add_argument("--data_dir", type=str, default=None, help="Override the results directory")
    args = parser.parse_args()
    
    analyze_suite(args.toml, args.data_dir)


if __name__ == "__main__":
    main()
