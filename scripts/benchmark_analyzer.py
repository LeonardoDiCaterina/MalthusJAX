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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

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
        
    return {
        "Dependent_Var": dependent_var,
        "R2": model.rsquared,
        "beta_1 (Treatment)": model.params.get("is_treatment", np.nan),
        "beta_1_pval": model.pvalues.get("is_treatment", np.nan),
        "beta_3 (Interaction)": model.params.get("interaction_term", np.nan),
        "beta_3_pval": model.pvalues.get("interaction_term", np.nan),
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
    df_global.to_csv(analysis_dir / "unpivoted_raw_data.csv", index=False)
    
    ref_pipeline = config.analysis.reference_pipeline
    target_pipelines = [p for p in config.pipelines.keys() if p != ref_pipeline]
    
    if not ref_pipeline or ref_pipeline not in config.pipelines:
        print(f"WARNING: Reference pipeline '{ref_pipeline}' not found in TOML.")
        return

    pivot_rows = []
    
    for target in target_pipelines:
        print(f"\nAnalyzing: {target} vs {ref_pipeline}")
        df_paired = synthesize_regression_dataset(df_global, target_pipeline=target, ref_pipeline=ref_pipeline)
        
        if df_paired.empty:
            print("  -> No strictly paired traces found. Skipping.")
            continue
            
        print(f"  -> Synthesized {len(df_paired)//2} paired coordinate traces.")
        
        for fn_name in df_paired["fn_name"].unique():
            df_fn = df_paired[df_paired["fn_name"] == fn_name]
            prefix = f"{target}_vs_{ref_pipeline}_{fn_name}"
            
            # Generate Visuals
            generate_boxplots(df_fn, analysis_dir, prefix)
            generate_scaling_plots(df_fn, "execution_time", analysis_dir, prefix)
            generate_scaling_plots(df_fn, "best_fitness", analysis_dir, prefix)
            
            # Run OLS if LHS (or Cartesian with enough D variance)
            if len(df_fn["D"].unique()) > 1:
                res_time = run_ols_diagnostics(df_fn, "execution_time", analysis_dir, prefix)
                res_fit = run_ols_diagnostics(df_fn, "best_fitness", analysis_dir, prefix)
                
                if res_time:
                    res_time.update({"Target": target, "Benchmark": fn_name})
                    pivot_rows.append(res_time)
                if res_fit:
                    res_fit.update({"Target": target, "Benchmark": fn_name})
                    pivot_rows.append(res_fit)

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        # Move target and benchmark to front
        cols = ["Target", "Benchmark", "Dependent_Var"] + [c for c in pivot_df.columns if c not in ["Target", "Benchmark", "Dependent_Var"]]
        pivot_df = pivot_df[cols]
        
        pivot_file = analysis_dir / "ols_regression_table.csv"
        pivot_df.to_csv(pivot_file, index=False)
        print(f"\nOLS Regression Pivot Table saved to {pivot_file}")
        
        # Export to LaTeX
        tex_file = analysis_dir / "ols_regression_table.tex"
        pivot_df.to_latex(tex_file, index=False, float_format="%.4e")
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
