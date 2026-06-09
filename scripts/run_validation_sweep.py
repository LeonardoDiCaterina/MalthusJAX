#!/usr/bin/env python3
"""Automation script for selection operator validation sweep.

Compares MalthusJAX (with evosax_mimic_selection) against EvoSAX under combo3,
computing Wilcoxon signed-rank tests, win-loss splits, and printing the results.
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
SWEEP_DIR = BASE_DIR / "results" / "sweeps" / "validation_sweep"
CONFIG_DIR = SWEEP_DIR / "configs"
RAW_DIR = SWEEP_DIR / "raw"
POST_DIR = SWEEP_DIR / "postprocessed"

for d in [CONFIG_DIR, RAW_DIR, POST_DIR]:
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
COMBO3 = {"pop_size": 50, "elite_k": 10, "elite_ratio": 0.20, "cr": 0.8, "ms": 0.20, "generations": 20}


def run_validation_experiment(toml_path: Path, spec: StatisticalComparisonSpec, out_subdir: Path) -> dict:
    """Run Composer on TOML, compute statistics, and return summary row."""
    cached_json = out_subdir / "validation_summary.json"
    if cached_json.exists() and os.environ.get("FORCE_RERUN") != "1":
        print(f"[{toml_path.stem}] Found cached results at {cached_json}. Loading...", flush=True)
        with open(cached_json, "r") as f:
            cached_data = json.load(f)
        return cached_data["summary_row"]

    print(f"[{toml_path.stem}] Running optimization runs...", flush=True)
    comparison = Composer.from_toml(toml_path, pop_seed=42, shared_initial_population=True)
    
    print(f"[{toml_path.stem}] Extracting aligned paired datasets...", flush=True)
    dataset = paired_dataset_from_comparison(
        comparison=comparison,
        left_pipeline="malthusjax_mimic",
        right_pipeline="evosax",
        spec=spec,
    )
    
    print(f"[{toml_path.stem}] Executing statistical hypothesis tests...", flush=True)
    comparator = StatisticalComparator()
    suite = comparator.compare_suite([dataset], spec)
    result = suite.results[0]
    
    primary_p = result.tests.get("wilcoxon", {}).p_value if "wilcoxon" in result.tests else None

    row = {
        "label": result.label,
        "n_paired": result.n_paired,
        "primary_p": primary_p,
        "wins_left": result.wins_left,
        "wins_right": result.wins_right,
        "ties": result.ties,
        "cohen_dz": result.effects.cohen_dz,
        "decision": "pass" if result.decision_pass else "fail",
        "basis": result.decision_basis,
    }

    # Save Markdown and JSON reports
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "validation_summary.md").write_text(suite.to_markdown())
    
    save_data = {
        "suite": suite.to_dict(),
        "summary_row": row
    }
    (out_subdir / "validation_summary.json").write_text(json.dumps(save_data, indent=2))
    
    print(f"[{toml_path.stem}] Wilcoxon p={row['primary_p']:.4g}, Wins: MJX_Mimic={row['wins_left']} vs EvoSAX={row['wins_right']}", flush=True)
    return row


def main():
    print("===================================================", flush=True)
    print("=== STARTING SELECTION VALIDATION SWEEP (COMBO3) ===", flush=True)
    print("===================================================", flush=True)
    
    spec_validation = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
        include_tests=("wilcoxon", "paired_t", "sign"),
    )

    master_rows = []
    total_fns = len(BBOB_FUNCTIONS)
    
    for idx, fn in enumerate(BBOB_FUNCTIONS, 1):
        print(f"\n[PROGRESS] {idx}/{total_fns}: Function={fn}...", flush=True)
        toml_content = f"""# Auto-generated validation sweep TOML
[experiment]
name = "validation_{fn}"
output_dir = "results/sweeps/validation_sweep/raw/{fn}"

[experiment.shared]
genome_type = "real"
genome_length = 5
bounds = [-5.0, 5.0]
pop_size = {COMBO3['pop_size']}
generations = {COMBO3['generations']}
seeds = {SEEDS_50}
track_best = "NONE"
use_history_for_final = true

[pipelines.malthusjax_mimic]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "evosax_mimic_selection:num_selections={COMBO3['pop_size']},elite_k={COMBO3['elite_k']}"
crossover = "evosax_uniform_crossover:crossover_rate={COMBO3['cr']}"
mutation = "evosax_gaussian:mutation_strength={COMBO3['ms']}"

[pipelines.evosax]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "evosax"
evosax_strategy = "SimpleGA"
strategy_params = {{ crossover_rate = {COMBO3['cr']}, elite_ratio = {COMBO3['elite_ratio']}, mutation_std = {COMBO3['ms']} }}
"""
        toml_path = CONFIG_DIR / f"{fn}_validation.toml"
        toml_path.write_text(toml_content)
        
        out_subdir = POST_DIR / f"{fn}_validation"
        row = run_validation_experiment(toml_path, spec_validation, out_subdir)
        row["function"] = fn
        master_rows.append(row)

    # Save Master CSV Report
    master_df = pd.DataFrame(master_rows)
    master_csv = SWEEP_DIR / "validation_sweep_report.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\nValidation Sweep Report saved to: {master_csv}", flush=True)

    print("\nSummary Table:")
    summary_print = master_df[["function", "wins_left", "wins_right", "ties", "primary_p", "cohen_dz", "decision"]]
    summary_print.columns = ["Function", "Wins MJX_Mimic", "Wins EvoSAX", "Ties", "Wilcoxon p", "Cohen dz", "Verdict"]
    print(summary_print.to_string(index=False))

    # Generate visual summary of p-values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(master_df["function"], master_df["primary_p"], edgecolor="black", color="lightgreen")
    ax.axhline(0.05, color="red", linestyle="--", label="alpha=0.05")
    ax.set_ylabel("Wilcoxon p-value")
    ax.set_title("Wilcoxon Signed-Rank P-Values comparing MJX (Mimic Selection) vs EvoSAX (COMBO3)")
    ax.set_xticklabels(master_df["function"], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SWEEP_DIR / "plot_validation_pvalues.png", dpi=150)
    plt.close(fig)

    print("===================================================", flush=True)
    print("=== VALIDATION SWEEP COMPLETED SUCCESSFULLY ===", flush=True)
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
