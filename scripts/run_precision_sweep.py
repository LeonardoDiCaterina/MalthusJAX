#!/usr/bin/env python3
"""Automation script for comparing MalthusJAX float32 vs float16 precision.

Runs the 10 BBOB functions under combo2 hyperparameters using native operators,
computes Wilcoxon signed-rank tests, win-loss splits, and timing averages.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from malthusjax.benchmarking.statistics import (
    HypothesisKind,
    StatisticalComparator,
    StatisticalComparisonSpec,
    Sidedness,
    paired_dataset_from_comparison,
)
from malthusjax.composer import Composer

# 1. Directory Setup
BASE_DIR = Path("/Users/leonardodicaterina/Documents/GitHub/MalthusJAX")
SWEEP_DIR = BASE_DIR / "results" / "sweeps" / "precision_sweep"
CONFIG_DIR = SWEEP_DIR / "configs"
RAW_DIR = SWEEP_DIR / "raw"
POST_DIR = SWEEP_DIR / "postprocessed"
IMAGE_DIR = BASE_DIR / "docs" / "thesis" / "images"

for d in [CONFIG_DIR, RAW_DIR, POST_DIR, IMAGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Configurations & Functions
BBOB_FUNCTIONS = [
    "sphere",
    "linear_slope",
    "rosenbrock",
    "step_ellipsoidal",
    "ellipsoidal_rotated",
    "different_powers",
    "rastrigin_rotated",
    "weierstrass",
    "schwefel",
    "katsuura",
]

SEEDS_50 = list(range(50))

COMBO2 = {"pop_size": 20, "elite_k": 5, "cr": 0.5, "ms": 0.10, "generations": 20}
COMBO3 = {"pop_size": 50, "elite_k": 10, "cr": 0.8, "ms": 0.20, "generations": 20}
EXTREME = {"pop_size": 100, "elite_k": 20, "cr": 0.8, "ms": 0.20, "generations": 100}


def run_precision_experiment(toml_path: Path, spec: StatisticalComparisonSpec, out_subdir: Path) -> dict:
    """Run Composer on TOML, compute statistics, and return summary row."""
    cached_json = out_subdir / "precision_summary.json"
    if cached_json.exists() and os.environ.get("FORCE_RERUN") != "1":
        print(f"[{toml_path.stem}] Found cached results at {cached_json}. Loading...", flush=True)
        with open(cached_json, "r") as f:
            cached_data = json.load(f)
        return cached_data["summary_row"]

    print(f"[{toml_path.stem}] Running optimization runs...", flush=True)
    comparison = Composer.from_toml(toml_path, pop_seed=42, shared_initial_population=False)
    
    print(f"[{toml_path.stem}] Extracting aligned paired datasets...", flush=True)
    dataset = paired_dataset_from_comparison(
        comparison=comparison,
        left_pipeline="mjx_float32",
        right_pipeline="mjx_float16",
        spec=spec,
    )
    
    print(f"[{toml_path.stem}] Executing statistical hypothesis tests...", flush=True)
    comparator = StatisticalComparator()
    suite = comparator.compare_suite([dataset], spec)
    result = suite.results[0]
    
    # Calculate Shapiro-Wilk for normality of differences
    diffs = dataset.left_values - dataset.right_values
    if diffs.size >= 3:
        _, shapiro_p = stats.shapiro(diffs)
    else:
        shapiro_p = np.nan
        
    # Extract timing details (excluding seed 0 warmup)
    f32_runs = comparison.pipelines["mjx_float32"].runs
    f16_runs = comparison.pipelines["mjx_float16"].runs
    
    # Exclude seed 0 from execution timing
    f32_exec_times = [run.timings.get("execution", 0.0) for run in f32_runs[1:]]
    f16_exec_times = [run.timings.get("execution", 0.0) for run in f16_runs[1:]]
    
    left_evo_mean = np.mean(f32_exec_times)
    right_evo_mean = np.mean(f16_exec_times)
    evo_speedup = left_evo_mean / right_evo_mean if right_evo_mean > 0 else 1.0

    primary_p = result.tests.get("wilcoxon", {}).p_value if "wilcoxon" in result.tests else None

    row = {
        "label": result.label,
        "n_paired": result.n_paired,
        "shapiro_p": shapiro_p,
        "primary_p": primary_p,
        "wins_left": result.wins_left,
        "wins_right": result.wins_right,
        "ties": result.ties,
        "cohen_dz": result.effects.cohen_dz,
        "decision": "pass" if result.decision_pass else "fail",
        "basis": result.decision_basis,
        "left_evo_mean": float(left_evo_mean),
        "right_evo_mean": float(right_evo_mean),
        "evo_speedup": float(evo_speedup),
    }

    # Save Markdown and JSON reports
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "precision_summary.md").write_text(suite.to_markdown())
    
    save_data = {
        "suite": suite.to_dict(),
        "summary_row": row
    }
    (out_subdir / "precision_summary.json").write_text(json.dumps(save_data, indent=2))
    
    # Save a boxplot comparing the final fitness distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([dataset.left_values, dataset.right_values], labels=["float32", "float16"])
    ax.set_ylabel("End Fitness")
    ax.set_title(f"Precision Impact on {toml_path.stem.split('_')[0]}")
    fig.tight_layout()
    fig.savefig(out_subdir / "plot_precision_comparison.png", dpi=150)
    plt.close(fig)

    print(f"[{toml_path.stem}] Wilcoxon p={row['primary_p']:.4g}, Wins: F32={row['wins_left']} vs F16={row['wins_right']}", flush=True)
    print(f"[{toml_path.stem}] Mean Execution: F32={left_evo_mean*1000:.2f}ms, F16={right_evo_mean*1000:.2f}ms", flush=True)
    return row


def main():
    print("===================================================", flush=True)
    print("=== STARTING PRECISION IMPACT SWEEP (F32 vs F16) ===", flush=True)
    print("===================================================", flush=True)
    
    spec_precision = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
        include_tests=("wilcoxon", "paired_t", "sign"),
    )

    configs_to_run = {
        "combo2": COMBO2,
        "combo3": COMBO3,
        "extreme": EXTREME,
    }

    for name, params in configs_to_run.items():
        print(f"\n===================================================", flush=True)
        print(f"=== RUNNING CONFIGURATION: {name.upper()} ===", flush=True)
        print(f"===================================================", flush=True)
        
        master_rows = []
        total_fns = len(BBOB_FUNCTIONS)
        
        for idx, fn in enumerate(BBOB_FUNCTIONS, 1):
            print(f"\n[PROGRESS] Config={name} | {idx}/{total_fns}: Function={fn}...", flush=True)
            toml_content = f"""# Auto-generated precision sweep TOML
[experiment]
name = "precision_{fn}_{name}"
output_dir = "results/sweeps/precision_sweep/raw/{fn}_{name}"

[experiment.shared]
genome_type = "real"
genome_length = 5
bounds = [-5.0, 5.0]
pop_size = {params['pop_size']}
generations = {params['generations']}
seeds = {SEEDS_50}
track_best = "NONE"
use_history_for_final = true

[pipelines.mjx_float32]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections={params['pop_size']},elite_k={params['elite_k']}"
crossover = "uniform_real"
mutation = "gaussian:mutation_strength={params['ms']}"
dtype = "float32"

[pipelines.mjx_float16]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections={params['pop_size']},elite_k={params['elite_k']}"
crossover = "uniform_real"
mutation = "gaussian:mutation_strength={params['ms']}"
dtype = "float16"
"""
            toml_path = CONFIG_DIR / f"{fn}_precision_{name}.toml"
            toml_path.write_text(toml_content)
            
            out_subdir = POST_DIR / f"{fn}_precision_{name}"
            row = run_precision_experiment(toml_path, spec_precision, out_subdir)
            row["function"] = fn
            row["config"] = name
            master_rows.append(row)

        # Save Config-Specific CSV Report
        master_df = pd.DataFrame(master_rows)
        master_csv = SWEEP_DIR / f"precision_sweep_report_{name}.csv"
        master_df.to_csv(master_csv, index=False)
        print(f"\n[{name.upper()}] Report saved to: {master_csv}", flush=True)

        print(f"\n[{name.upper()}] Summary Table:")
        summary_print = master_df[["function", "wins_left", "wins_right", "ties", "primary_p", "cohen_dz", "left_evo_mean", "right_evo_mean"]]
        summary_print.columns = ["Function", "Wins F32", "Wins F16", "Ties", "Wilcoxon p", "Cohen dz", "F32 Time (s)", "F16 Time (s)"]
        print(summary_print.to_string(index=False))

        # Generate visual summary of p-values
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(master_df["function"], master_df["primary_p"], edgecolor="black", color="skyblue")
        ax.axhline(0.05, color="red", linestyle="--", label="alpha=0.05")
        ax.set_ylabel("Wilcoxon p-value")
        ax.set_title(f"Wilcoxon Signed-Rank P-Values: F32 vs F16 ({name.upper()}, 50 Seeds)")
        ax.set_xticklabels(master_df["function"], rotation=30, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(SWEEP_DIR / f"plot_precision_pvalues_{name}.png", dpi=150)
        plt.close(fig)

    print("===================================================", flush=True)
    print("=== ALL PRECISION SWEEPS COMPLETED SUCCESSFULLY ===", flush=True)
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
