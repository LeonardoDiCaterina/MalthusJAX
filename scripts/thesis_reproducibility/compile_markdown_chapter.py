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
    md = "| Hypothesis | Benchmark | Metric | $R^2$ | `is_mjx` Effect | Interaction `is_mjx:log(D)` | Diagnostics |\n"
    md += "|------------|-----------|--------|-------|-----------------|-----------------------------|-------------|\n"
    
    for _, row in pivot_df.iterrows():
        mjx_effect = format_coef(row["is_mjx_coef"], row["is_mjx_pval"])
        interaction = format_coef(row["interaction_coef"], row["interaction_pval"])
        r2 = f"{row['R2']:.3f}" if pd.notna(row['R2']) else "-"
        diag = format_diagnostics(row)
        
        md += f"| {row['Hypothesis']} | {row['Benchmark']} | {row['Dependent_Var']} | {r2} | {mjx_effect} | {interaction} | {diag} |\n"
        
    return md

def generate_interaction_plots(raw_df: pd.DataFrame, out_dir: Path):
    plot_paths = {}
    
    # Precompute log mappings
    raw_df["log_D"] = np.log(raw_df["D"])
    # Clean zeros for log
    raw_df["log_execution_time"] = np.log(raw_df["execution_time"] + 1e-9)
    # Fitness might have negatives if maximize=True wasn't handled right, 
    # but we assume objective natively here (>=0)
    raw_df["log_best_fitness"] = np.log(raw_df["best_fitness"] + 1e-9)
    
    sns.set_theme(style="whitegrid")
    
    for hyp in ["hyp1", "hyp2", "hyp3"]:
        df_hyp = raw_df[raw_df["hypothesis"] == hyp]
        if df_hyp.empty:
            continue
            
        plot_paths[hyp] = []
        for fn_name in df_hyp["fn_name"].unique():
            df_plot = df_hyp[df_hyp["fn_name"] == fn_name]
            
            # 1. Execution Time Plot
            plt.figure()
            g = sns.lmplot(
                data=df_plot, 
                x="log_D", 
                y="log_execution_time", 
                hue="is_mjx",
                scatter_kws={'alpha':0.5},
                height=5, 
                aspect=1.2
            )
            g.set_axis_labels("log(Genome Size D)", "log(Execution Time)")
            plt.title(f"{hyp.upper()} - {fn_name.capitalize()}: Execution Time Scaling")
            time_path = out_dir / f"{hyp}_{fn_name}_time_lmplot.png"
            plt.savefig(time_path, bbox_inches='tight', dpi=150)
            plt.close('all')
            plot_paths[hyp].append(time_path)
            
            # 2. Fitness Plot
            plt.figure()
            g = sns.lmplot(
                data=df_plot, 
                x="log_D", 
                y="log_best_fitness", 
                hue="is_mjx",
                scatter_kws={'alpha':0.5},
                height=5, 
                aspect=1.2
            )
            g.set_axis_labels("log(Genome Size D)", "log(Best Fitness)")
            plt.title(f"{hyp.upper()} - {fn_name.capitalize()}: Fitness Scaling")
            fit_path = out_dir / f"{hyp}_{fn_name}_fitness_lmplot.png"
            plt.savefig(fit_path, bbox_inches='tight', dpi=150)
            plt.close('all')
            plot_paths[hyp].append(fit_path)
            
    return plot_paths

def main():
    diagnostics_dir = Path("results/diagnostics")
    plots_dir = Path("results/thesis_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    pivot_csv = diagnostics_dir / "grand_pivot_table.csv"
    raw_csv = diagnostics_dir / "lhs_raw_data.csv"
    
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
