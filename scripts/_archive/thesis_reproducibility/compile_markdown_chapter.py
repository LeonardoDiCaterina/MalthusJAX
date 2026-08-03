"""Compile the final Thesis Chapter Results in Markdown format.

This script parses the LHS raw data and Grand Pivot Table from the
diagnostics engine, generates targeted seaborn.lmplot interaction plots,
and weaves everything into a structured Markdown document.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pval_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def format_coef(coef: float, pval: float) -> str:
    if pd.isna(coef): return "-"
    return f"{coef:.3f}{pval_stars(pval)}"

def format_diagnostics(row: pd.Series) -> str:
    warnings = []
    if row["BP_pval"] < 0.05: warnings.append("Het")
    if row["SW_pval"] < 0.05: warnings.append("NonNorm")
    if pd.notna(row["Mardia_hz_pval"]) and row["Mardia_hz_pval"] < 0.05:
        warnings.append("MultiNorm")
    return ", ".join(warnings) if warnings else "Pass"

def build_pivot_table(pivot_df: pd.DataFrame) -> str:
    md = "| Hypothesis | Target Pipeline | Benchmark | Metric | $R^2$ | Base Effect ($\\beta_1$) | Interaction $\\beta_3$ | Diagnostics |\n"
    md += "|------------|-----------------|-----------|--------|-------|------------------------|------------------------|-------------|\n"
    
    for _, row in pivot_df.iterrows():
        treatment_effect = format_coef(row["is_treatment_coef"], row["is_treatment_pval"])
        interaction = format_coef(row["interaction_coef"], row["interaction_pval"])
        r2 = f"{row['R2']:.3f}" if pd.notna(row['R2']) else "-"
        # Mardia might not exist anymore
        mardia_pval = row.get("Mardia_hz_pval", np.nan)
        warnings = []
        if row.get("BP_pval", 1.0) < 0.05: warnings.append("Het")
        if row.get("SW_pval", 1.0) < 0.05: warnings.append("NonNorm")
        if pd.notna(mardia_pval) and mardia_pval < 0.05:
            warnings.append("MultiNorm")
        diag = ", ".join(warnings) if warnings else "Pass"
        
        md += f"| {row['Hypothesis']} | {row.get('Target_Pipeline', 'target')} | {row['Benchmark']} | {row['Dependent_Var']} | {r2} | {treatment_effect} | {interaction} | {diag} |\n"
        
    return md

def generate_interaction_plots(raw_df: pd.DataFrame, out_dir: Path):
    plot_paths = {}
    
    # Precompute log mappings
    raw_df["log_D"] = np.log(raw_df["D"])
    raw_df["log_execution_time"] = np.log(raw_df["execution_time"] + 1e-9)
    # Fitness might have negatives, but we plot raw fitness
    
    sns.set_theme(style="whitegrid")
    
    # We want to plot pipelines for each function
    for fn_name in raw_df["fn_name"].unique():
        df_fn = raw_df[raw_df["fn_name"] == fn_name]
        
        # 1. Execution Time Plot
        plt.figure()
        g = sns.lmplot(
            data=df_fn, 
            x="log_D", 
            y="log_execution_time", 
            hue="pipeline",
            scatter_kws={'alpha':0.5},
            height=5, 
            aspect=1.2
        )
        g.set_axis_labels("log(Genome Size D)", "log(Execution Time)")
        plt.title(f"{fn_name.capitalize()}: Execution Time Scaling")
        time_path = out_dir / f"{fn_name}_time_lmplot.png"
        plt.savefig(time_path, bbox_inches='tight', dpi=150)
        plt.close('all')
        
        if "all" not in plot_paths: plot_paths["all"] = []
        plot_paths["all"].append(time_path)
        
    return plot_paths

def main():
    diagnostics_dir = Path("results/diagnostics")
    plots_dir = Path("results/thesis_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    pivot_csv = diagnostics_dir / "regression_pivot_table.csv"
    raw_csv = diagnostics_dir / "lhs_global_raw_data.csv"
    
    if not pivot_csv.exists() or not raw_csv.exists():
        print("Required CSV files not found. Please run thesis-lhs-regression first.")
        return
        
    pivot_df = pd.read_csv(pivot_csv)
    raw_df = pd.read_csv(raw_csv)
    
    # Generate Plots
    print("Generating seaborn lmplots...")
    plot_paths = generate_interaction_plots(raw_df, plots_dir)
    
    # Build Table
    pivot_md = build_pivot_table(pivot_df)
    
    # Assemble Markdown
    chapter_content = f"""# Chapter 4: Results and LHS Experimental Analysis

This chapter presents the statistical findings from the Latin Hypercube Sampling (LHS) experiments across the parameter space $D \\in [2, 100]$, $P \\in [10, 1000]$, and $G \\in [10, 1000]$.

To robustly validate the thesis hypotheses, we fit Multiple Linear Regression models:
`log(Y) ~ is_mjx + log(D) + log(P) + log(G) + is_mjx:log(D)`

## 4.1 Grand Regression Summary

The table below consolidates the coefficients and diagnostic p-values for all models. Significance codes: `***` $p<0.001$, `**` $p<0.01$, `*` $p<0.05$.

{pivot_md}

*Note on Diagnostics: 'Het' indicates Heteroscedasticity (Breusch-Pagan p<0.05). 'NonNorm' indicates non-normal residuals (Shapiro-Wilk p<0.05). 'MultiNorm' indicates failure of Mardia's multivariate normality test on the LHS design matrix.*

## 4.2 Interaction Scaling Visualizations

The following plots visualize the core `is_mjx:log(D)` interaction effect. The scatter points represent the individual LHS samples marginalized over population and generation variance, while the regression line demonstrates the dimension-scaling slope.

"""
    # Insert images
    for hyp, paths in plot_paths.items():
        chapter_content += f"### {hyp.upper()} Scalability\n\n"
        for p in paths:
            rel_path = os.path.relpath(p, ".")
            chapter_content += f"![{p.stem}](../../{rel_path})\n\n"

    out_file = Path("chapter_results.md")
    with open(out_file, "w") as f:
        f.write(chapter_content)
        
    print(f"Compilation complete! Rendered to {out_file}")

if __name__ == "__main__":
    main()
