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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.stats.comparator import StatisticalComparator
from malthusjax.stats.core import (
    HypothesisKind,
    MultipleTestingPolicy,
    PairedMetricDataset,
    Sidedness,
    StatisticalComparisonSpec,
)
from malthusjax.stats.dataset_builder import synthesize_regression_dataset
from malthusjax.stats.io import suite_to_dict, suite_to_markdown
from malthusjax.stats.regression_analyzer import OLSRegressionAnalyzer, RegressionSpec


def parse_global_data(results_dir: Path, target_metrics: list = None) -> pd.DataFrame:
    """Parse all JSON artifacts into a single unpivoted dataframe."""
    if target_metrics is None:
        target_metrics = ["best_fitness", "execution_time"]

    records = []

    # Support both new TOML engine and legacy hardcoded script outputs
    search_patterns = [
        "benchmark_results.json",
        "parity_results.json",
        "ablation_results.json",
        "representation_results.json",
        "summary.json",
        "suite_summary.json",
    ]

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
                    m = re.search(r"_d(\d+)_", exp_name)
                    if m:
                        D = float(m.group(1))
                if pd.isna(P):
                    m = re.search(r"_p(\d+)_", exp_name)
                    if m:
                        P = float(m.group(1))
                if pd.isna(G):
                    m = re.search(r"_g(\d+)", exp_name)
                    if m:
                        G = float(m.group(1))

            pipelines = data.get("pipelines", {})
            for p_name, p_data in pipelines.items():
                runs = p_data.get("per_seed", p_data)
                if not isinstance(runs, list):
                    runs = [runs]

                for run in runs:
                    seed = run.get("seed", -1)

                    record = {
                        "experiment": exp_name,
                        "fn_name": fn_name,
                        "D": D,
                        "P": P,
                        "G": G,
                        "pipeline": p_name,
                        "seed": seed,
                    }

                    for metric in target_metrics:
                        val = run.get(metric, np.nan)

                        # Fallbacks for execution time aliases
                        if metric == "execution_time" and pd.isna(val):
                            if "timings" in run and "total" in run["timings"]:
                                val = run["timings"]["total"]
                            else:
                                val = run.get("duration_seconds", np.nan)

                        try:
                            val = float(val) if val is not None else np.nan
                        except (ValueError, TypeError):
                            val = np.nan

                        record[metric] = val

                    # Skip if ALL target metrics are nan
                    if all(pd.isna(record[m]) for m in target_metrics):
                        continue

                    records.append(record)

    return pd.DataFrame(records)


# synthesize_regression_dataset has been moved to malthusjax.stats.dataset_builder


def generate_boxplots(df: pd.DataFrame, output_dir: Path, prefix: str, target_metrics: list):
    """Generate side-by-side boxplots for dynamically specified metrics."""
    n_metrics = len(target_metrics)
    plt.figure(figsize=(6 * n_metrics, 5))

    for i, metric in enumerate(target_metrics):
        if metric not in df.columns:
            continue

        plt.subplot(1, n_metrics, i + 1)
        sns.boxplot(data=df.dropna(subset=[metric]), x="pipeline", y=metric)

        if metric == "execution_time":
            plt.yscale("log")

        plt.title(f"{metric.replace('_', ' ').title()}")
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

    sns.lmplot(
        data=df_clean,
        x="log_D",
        y="log_Y",
        hue="pipeline",
        height=6,
        aspect=1.2,
        scatter_kws={"alpha": 0.5},
    )
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
            f.write(" & ".join([str(c).replace("_", "\\_") for c in df.columns]) + " \\\\\n")
            f.write("\\midrule\n")
            for _, row in df.iterrows():
                formatted = []
                for val in row:
                    if isinstance(val, float):
                        formatted.append(f"{val:.4e}")
                    else:
                        formatted.append(str(val).replace("_", "\\_"))
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
    df_global = parse_global_data(results_dir, target_metrics=config.analysis.target_metrics)

    if df_global.empty:
        print("ERROR: No valid JSON result artifacts found!")
        sys.exit(1)

    print(f"Loaded {len(df_global)} individual traces.")

    df_global = df_global.drop_duplicates(
        subset=["fn_name", "D", "P", "G", "pipeline", "seed"], keep="last"
    )
    print(f"Deduplicated to {len(df_global)} unique traces.")
    df_global.to_csv(analysis_dir / "unpivoted_raw_data.csv", index=False)

    ref_pipeline = config.analysis.reference_pipeline
    target_pipelines = [p for p in config.pipelines.keys() if p != ref_pipeline]

    if not ref_pipeline or ref_pipeline not in config.pipelines:
        print(f"WARNING: Reference pipeline '{ref_pipeline}' not found in TOML.")
        return

    parity_rows = []
    comparator = StatisticalComparator()

    for target in target_pipelines:
        print(f"\nAnalyzing: {target} vs {ref_pipeline}")
        df_paired = synthesize_regression_dataset(
            df_global, target_pipeline=target, ref_pipeline=ref_pipeline
        )

        if df_paired.empty:
            print("  -> No strictly paired traces found. Skipping.")
            continue

        print(f"  -> Synthesized {len(df_paired) // 2} strictly paired coordinate traces.")

        for fn_name in df_paired["fn_name"].unique():
            df_fn = df_paired[df_paired["fn_name"] == fn_name]
            prefix = f"{target}_vs_{ref_pipeline}_{fn_name}"

            generate_boxplots(df_fn, analysis_dir, prefix, config.analysis.target_metrics)

            if (
                config.suite.mode == "lhs"
                or len(df_fn["D"].unique()) > 1
                and config.suite.mode != "cartesian"
            ):
                # OLS Scaling Regressions using malthusjax.stats
                for var in config.analysis.target_metrics:
                    generate_scaling_plots(df_fn, var, analysis_dir, prefix)

                # The OLS Regressions are now handled globally via the OLSRegressionAnalyzer
                # Scaling plots are still generated per benchmark fn_name.
                pass
            else:
                # Cartesian Parity using malthusjax.stats
                print(
                    f"  -> {fn_name}: Cartesian run detected. Using malthusjax.stats.comparator for TOST and Wilcoxon."
                )

                # Split and test per Dimensionality
                for D_val in df_fn["D"].unique():
                    df_D = df_fn[df_fn["D"] == D_val]
                    df_target = df_D[df_D["is_treatment"] == 1].sort_values("seed")
                    df_ref = df_D[df_D["is_treatment"] == 0].sort_values("seed")

                    prefix_D = f"{prefix}_d{int(D_val)}"

                    datasets = []
                    for var in config.analysis.target_metrics:
                        if var not in df_target.columns or var not in df_ref.columns:
                            continue

                        target_vals = df_target[var].dropna().values
                        ref_vals = df_ref[var].dropna().values

                        if len(target_vals) > 0 and len(target_vals) == len(ref_vals):
                            datasets.append(
                                PairedMetricDataset(
                                    label=f"{prefix_D}_{var}",
                                    left_name=target,
                                    right_name=ref_pipeline,
                                    seeds=df_target["seed"].tolist(),
                                    left_values=target_vals,
                                    right_values=ref_vals,
                                    metric_name=var,
                                    metric_source="summary",
                                    metadata={"D": float(D_val)},
                                )
                            )

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
                                    include_mean_summary=True,
                                )
                            else:
                                spec = StatisticalComparisonSpec(
                                    metric_name="execution_time",
                                    hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
                                    sidedness=Sidedness.TWO_SIDED,
                                    multiple_testing=MultipleTestingPolicy.NONE,
                                    include_tests=("wilcoxon", "paired_t"),
                                    include_mean_summary=True,
                                )

                            # Compare single dataset directly to allow per-metric specs
                            suite_res = comparator.compare_suite([ds], spec)

                            # Export JSON and Markdown for the metric suite
                            with open(
                                analysis_dir / f"{prefix_D}_{ds.metric_name}_suite.json", "w"
                            ) as f:
                                json.dump(suite_to_dict(suite_res), f, indent=2)
                            with open(
                                analysis_dir / f"{prefix_D}_{ds.metric_name}_suite.md", "w"
                            ) as f:
                                f.write(suite_to_markdown(suite_res))

                            # Extract metrics for the legacy aggregated CSV table
                            res = suite_res.results[0]
                            wilcox_pval = (
                                res.tests["wilcoxon"].p_value if "wilcoxon" in res.tests else np.nan
                            )
                            tost_pval = res.tost.p_value_max if res.tost else np.nan
                            tost_delta = res.tost.margin if res.tost else np.nan

                            parity_rows.append(
                                {
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
                                    "Cohen_dz": res.effects.cohen_dz,
                                }
                            )

    if (
        config.suite.mode == "lhs"
        or len(df_global["D"].unique()) > 1
        and config.suite.mode != "cartesian"
    ):
        print("\nRunning Global OLS Regression Analysis...")
        ols_spec = RegressionSpec(
            dependent_vars=config.analysis.target_metrics, apply_multiple_testing=True
        )
        ols_analyzer = OLSRegressionAnalyzer(spec=ols_spec)
        pivot_df = ols_analyzer.analyze_suite(
            df_global=df_global,
            ref_pipeline=ref_pipeline,
            target_pipelines=target_pipelines,
            analysis_dir=analysis_dir,
        )

        if not pivot_df.empty:
            pivot_file = analysis_dir / "ols_regression_table.csv"
            pivot_df.to_csv(pivot_file, index=False)
            export_latex_safe(pivot_df, analysis_dir / "ols_regression_table.tex")
            print(f"OLS Regression Pivot Table saved to {pivot_file}")

    if parity_rows:
        parity_df = pd.DataFrame(parity_rows)
        for pval_col in ["Wilcoxon_pval", "TOST_pval"]:
            if pval_col in parity_df.columns:
                valid_idx = parity_df[pval_col].notna()
                if valid_idx.any():
                    _, corrected, _, _ = multipletests(
                        parity_df.loc[valid_idx, pval_col], method="holm"
                    )
                    parity_df.loc[valid_idx, f"{pval_col}_holm"] = corrected

        cols = ["Target", "Benchmark", "Metric", "D"] + [
            c for c in parity_df.columns if c not in ["Target", "Benchmark", "Metric", "D"]
        ]
        parity_df = parity_df[cols]

        parity_file = analysis_dir / "parity_wilcoxon_table.csv"
        parity_df.to_csv(parity_file, index=False)
        export_latex_safe(parity_df, analysis_dir / "parity_wilcoxon_table.tex")
        print(f"\nNon-Parametric Parity Table saved to {parity_file}")

    print(f"\nAnalysis complete! Artifacts dumped to {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified TOML-Driven Benchmark Analyzer")
    parser.add_argument(
        "--toml", type=str, required=True, help="Path to the TOML configuration file"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Override the results directory")
    args = parser.parse_args()

    analyze_suite(args.toml, args.data_dir)


if __name__ == "__main__":
    main()
