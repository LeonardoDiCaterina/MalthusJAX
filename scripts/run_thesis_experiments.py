#!/usr/bin/env python3
"""Master automation script for the complete thesis experiments suite.

Runs Part 1 (Parity Sweeps and Ablations) and Part 2 (Advanced Operators and Diverse Genomes),
saves raw outputs and postprocessed reports/plots, and compiles a master report.
"""

import json
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from malthusjax.benchmarking.statistics import (
    ExpectedDirection,
    HypothesisKind,
    MultipleTestingPolicy,
    Sidedness,
    StatisticalComparator,
    StatisticalComparisonSpec,
    paired_dataset_from_comparison,
)
from malthusjax.composer import Composer

# 1. Directory Setup
BASE_DIR = Path("/Users/leonardodicaterina/Documents/GitHub/MalthusJAX")
SWEEP_DIR = BASE_DIR / "results" / "sweeps" / "bbob_parity_sweep"
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
SEEDS_20 = list(range(20))

COMBOS = {
    "combo1": {"pop_size": 12, "elite_k": 4, "elite_ratio": 0.33333333, "cr": 0.3, "ms": 0.05},
    "combo2": {"pop_size": 20, "elite_k": 5, "elite_ratio": 0.25, "cr": 0.5, "ms": 0.10},
    "combo3": {"pop_size": 50, "elite_k": 10, "elite_ratio": 0.20, "cr": 0.8, "ms": 0.20},
}


def run_experiment(toml_path: Path, left_name: str, right_name: str, spec: StatisticalComparisonSpec, out_subdir: Path) -> dict:
    """Run Composer on TOML, compute statistics, generate plots, and return master row."""
    # Check if we have cached results to bypass optimization runs
    cached_json = out_subdir / "parity_summary.json"
    import os
    if cached_json.exists() and os.environ.get("FORCE_RERUN") != "1":
        print(f"[{toml_path.stem}] Found cached results at {cached_json}. Loading cached row...", flush=True)
        try:
            with open(cached_json, "r") as f:
                cached_data = json.load(f)
            result_entry = cached_data["results"][0]
            
            # Extract timing details
            timing_summary = result_entry.get("metadata", {}).get("timing_summary", {})
            duration = timing_summary.get("duration_seconds", {})
            left_time_mean = duration.get("left_mean", 0.0)
            right_time_mean = duration.get("right_mean", 0.0)
            total_speedup = right_time_mean / left_time_mean if left_time_mean > 0 else 1.0
            
            _comps = timing_summary.get("components", {})
            evolution_timing = _comps.get("execution", _comps.get("evolution", {}))
            left_evo_mean = evolution_timing.get("left_mean", 0.0)
            right_evo_mean = evolution_timing.get("right_mean", 0.0)
            evo_speedup = right_evo_mean / left_evo_mean if left_evo_mean > 0 else 1.0
            
            # Primary p-value and basis
            primary_p = None
            decision_basis = result_entry.get("decision_basis", "")
            if decision_basis == "tost" and result_entry.get("tost") is not None:
                primary_p = result_entry["tost"]["p_value_max"]
            elif decision_basis.startswith("wilcoxon") and "wilcoxon" in result_entry.get("tests", {}):
                primary_p = result_entry["tests"]["wilcoxon"]["p_value"]
            elif decision_basis.startswith("paired_t") and "paired_t" in result_entry.get("tests", {}):
                primary_p = result_entry["tests"]["paired_t"]["p_value"]
            elif "wilcoxon" in result_entry.get("tests", {}):
                primary_p = result_entry["tests"]["wilcoxon"]["p_value"]
                
            shapiro_p = np.nan
            
            row = {
                "label": result_entry["label"],
                "n_paired": result_entry["n_paired"],
                "shapiro_p": shapiro_p,
                "primary_p": primary_p,
                "wins_left": result_entry.get("wins_left", 0),
                "wins_right": result_entry.get("wins_right", 0),
                "ties": result_entry.get("ties", 0),
                "cohen_dz": result_entry.get("effects", {}).get("cohen_dz", 0.0),
                "decision": "pass" if result_entry.get("decision_pass", False) else "fail",
                "basis": decision_basis,
                "left_time_mean": left_time_mean,
                "right_time_mean": right_time_mean,
                "total_speedup": total_speedup,
                "left_evo_mean": left_evo_mean,
                "right_evo_mean": right_evo_mean,
                "evo_speedup": evo_speedup
            }
            
            # Print brief summary matching the regular printout
            p_str = f"{row['primary_p']:.4g}" if row['primary_p'] is not None else "NaN"
            shap_str = f"{row['shapiro_p']:.4g}" if not np.isnan(row['shapiro_p']) else "NaN"
            print(f"[{toml_path.stem}] Parity Verdict: {row['decision'].upper()} (Wilcoxon p={p_str}, Shapiro p={shap_str})", flush=True)
            print(f"[{toml_path.stem}] Evolution Timing: MJX Mean = {row['left_evo_mean']*1000:.2f}ms, EvoSAX Mean = {row['right_evo_mean']*1000:.2f}ms (Speedup: {row['evo_speedup']:.2f}x)", flush=True)
            print(f"[{toml_path.stem}] Total Duration: MJX Mean = {row['left_time_mean']:.4f}s, EvoSAX Mean = {row['right_time_mean']:.4f}s (Speedup: {row['total_speedup']:.2f}x)", flush=True)
            return row
        except Exception as e:
            print(f"[{toml_path.stem}] Error loading cached results: {e}. Running optimization runs...", flush=True)

    print(f"[{toml_path.stem}] Starting optimization runs for {left_name} and {right_name}...", flush=True)
    
    # Run optimization runs
    comparison = Composer.from_toml(toml_path, pop_seed=42)
    print(f"[{toml_path.stem}] Runs completed. Extracting aligned paired datasets...", flush=True)
    
    # Extract paired dataset
    dataset = paired_dataset_from_comparison(
        comparison=comparison,
        left_pipeline=left_name,
        right_pipeline=right_name,
        spec=spec,
    )
    
    # Run statistical comparison suite
    print(f"[{toml_path.stem}] Executing statistical hypothesis tests...", flush=True)
    comparator = StatisticalComparator()
    suite = comparator.compare_suite([dataset], spec)
    result = suite.results[0]
    
    # Check normality of paired differences
    diffs = dataset.left_values - dataset.right_values
    if diffs.size >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diffs)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Save Markdown and JSON reports
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "parity_summary.md").write_text(suite.to_markdown())
    (out_subdir / "parity_summary.json").write_text(json.dumps(suite.to_dict(), indent=2))
    
    # Generate Plots
    # 1) Paired Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dataset.left_values, dataset.right_values, alpha=0.7)
    min_v = float(min(np.min(dataset.left_values), np.min(dataset.right_values)))
    max_v = float(max(np.max(dataset.left_values), np.max(dataset.right_values)))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
    ax.set_xlabel(f"{left_name} end fitness")
    ax.set_ylabel(f"{right_name} end fitness")
    ax.set_title("Paired End-Fitness Scatter")
    fig.tight_layout()
    fig.savefig(out_subdir / "plot_end_scatter.png", dpi=150)
    plt.close(fig)

    # 2) Paired Differences Hist
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diffs, bins=15, alpha=0.8, color="skyblue", edgecolor="black")
    ax.axvline(0.0, linestyle="--", color="red")
    ax.set_xlabel("Paired difference (left - right)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Paired Differences")
    fig.tight_layout()
    fig.savefig(out_subdir / "plot_diff_hist.png", dpi=150)
    plt.close(fig)

    # 3) ECDF comparison
    lx = np.sort(dataset.left_values)
    ly = np.arange(1, lx.size + 1, dtype=float) / lx.size
    rx = np.sort(dataset.right_values)
    ry = np.arange(1, rx.size + 1, dtype=float) / rx.size
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(lx, ly, where="post", label=left_name)
    ax.step(rx, ry, where="post", label=right_name)
    ax.set_xlabel("End fitness")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of End Fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_subdir / "plot_end_ecdf.png", dpi=150)
    plt.close(fig)
    
    # Extract timing details
    timing_summary = result.metadata.get("timing_summary", {})
    duration = timing_summary.get("duration_seconds", {})
    left_time_mean = duration.get("left_mean", 0.0)
    right_time_mean = duration.get("right_mean", 0.0)
    total_speedup = right_time_mean / left_time_mean if left_time_mean > 0 else 1.0
    
    _comps = timing_summary.get("components", {})
    evolution_timing = _comps.get("execution", _comps.get("evolution", {}))
    left_evo_mean = evolution_timing.get("left_mean", 0.0)
    right_evo_mean = evolution_timing.get("right_mean", 0.0)
    evo_speedup = right_evo_mean / left_evo_mean if left_evo_mean > 0 else 1.0

    # Primary p-value and basis
    primary_p = None
    if result.decision_basis == "tost" and result.tost is not None:
        primary_p = result.tost.p_value_max
    elif result.decision_basis.startswith("wilcoxon") and "wilcoxon" in result.tests:
        primary_p = result.tests["wilcoxon"].p_value
    elif result.decision_basis.startswith("paired_t") and "paired_t" in result.tests:
        primary_p = result.tests["paired_t"].p_value

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
        "left_time_mean": left_time_mean,
        "right_time_mean": right_time_mean,
        "total_speedup": total_speedup,
        "left_evo_mean": left_evo_mean,
        "right_evo_mean": right_evo_mean,
        "evo_speedup": evo_speedup
    }

    # Verbose Output Print
    print(f"[{toml_path.stem}] Parity Verdict: {row['decision'].upper()} (Wilcoxon p={row['primary_p']:.4g}, Shapiro p={row['shapiro_p']:.4g})", flush=True)
    print(f"[{toml_path.stem}] Evolution Timing: MJX Mean = {row['left_evo_mean']*1000:.2f}ms, EvoSAX Mean = {row['right_evo_mean']*1000:.2f}ms (Speedup: {row['evo_speedup']:.2f}x)", flush=True)
    print(f"[{toml_path.stem}] Total Duration: MJX Mean = {row['left_time_mean']:.4f}s, EvoSAX Mean = {row['right_time_mean']:.4f}s (Speedup: {row['total_speedup']:.2f}x)", flush=True)
    return row


def main():
    print("===================================================", flush=True)
    print("=== STARTING MASTER THESIS EXPERIMENTS SUITE ===", flush=True)
    print("===================================================", flush=True)
    
    master_rows = []

    # ==========================================
    # PART 1.1: BBOB Parity Sweep
    # ==========================================
    print("\n--- Starting Part 1.1: BBOB Parity Sweep (10 functions x 3 configurations) ---", flush=True)
    spec_parity = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
        include_tests=("wilcoxon", "paired_t", "sign"),
    )

    idx = 1
    total_parity = len(BBOB_FUNCTIONS) * len(COMBOS)
    for fn in BBOB_FUNCTIONS:
        for name, combo in COMBOS.items():
            print(f"\n[PROGRESS] Sweeping Parity {idx}/{total_parity}: Function={fn}, Configuration={name}...", flush=True)
            toml_content = f"""# Auto-generated parity check TOML
[experiment]
name = "{fn}_d5_{name}"
output_dir = "results/sweeps/bbob_parity_sweep/raw/{fn}_{name}"

[experiment.shared]
genome_type = "real"
genome_length = 5
bounds = [-5.0, 5.0]
pop_size = {combo['pop_size']}
generations = 20
seeds = {SEEDS_50}
track_best = "NONE"
use_history_for_final = true

[pipelines.malthusjax]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections={combo['pop_size']},elite_k={combo['elite_k']}"
crossover = "evosax_uniform_crossover:crossover_rate={combo['cr']}"
mutation = "evosax_gaussian:mutation_strength={combo['ms']}"

[pipelines.evosax]
fitness = "bbob:fn_name={fn},num_dims=5,seed=0,maximize=false"
backend = "evosax"
evosax_strategy = "SimpleGA"
strategy_params = {{ crossover_rate = {combo['cr']}, elite_ratio = {combo['elite_ratio']}, mutation_std = {combo['ms']} }}
"""
            toml_path = CONFIG_DIR / f"{fn}_{name}.toml"
            toml_path.write_text(toml_content)
            
            out_subdir = POST_DIR / f"{fn}_{name}"
            row = run_experiment(toml_path, "malthusjax", "evosax", spec_parity, out_subdir)
            row["scope"] = "BBOB Parity"
            row["function"] = fn
            row["configuration"] = name
            master_rows.append(row)
            idx += 1

    # ==========================================
    # PART 1.2: Ablation Study
    # ==========================================
    print("\n--- Starting Part 1.2: Ablation Study (Sphere) ---", flush=True)
    ablation_toml = """# Auto-generated ablation check TOML
[experiment]
name = "sphere_ablation"
output_dir = "results/sweeps/bbob_parity_sweep/raw/sphere_ablation"

[experiment.shared]
genome_type = "real"
genome_length = 5
bounds = [-5.0, 5.0]
pop_size = 20
generations = 30
seeds = {seeds}
track_best = "NONE"
use_history_for_final = true

[pipelines.full_ga]
fitness = "bbob:fn_name=sphere,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections=20,elite_k=5"
crossover = "evosax_uniform_crossover:crossover_rate=0.5"
mutation = "evosax_gaussian:mutation_strength=0.1"

[pipelines.selection_only]
fitness = "bbob:fn_name=sphere,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections=20,elite_k=5"
crossover = "evosax_uniform_crossover:crossover_rate=0.0"
mutation = "evosax_gaussian:mutation_strength=0.0"

[pipelines.crossover_only]
fitness = "bbob:fn_name=sphere,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections=20,elite_k=5"
crossover = "evosax_uniform_crossover:crossover_rate=0.5"
mutation = "evosax_gaussian:mutation_strength=0.0"

[pipelines.mutation_only]
fitness = "bbob:fn_name=sphere,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 0
selection = "elite_pool:num_selections=20,elite_k=5"
crossover = "evosax_uniform_crossover:crossover_rate=0.0"
mutation = "evosax_gaussian:mutation_strength=0.1"
""".format(seeds=SEEDS_50)

    toml_path = CONFIG_DIR / "sphere_ablation.toml"
    toml_path.write_text(ablation_toml)
    
    print("[Ablation] Executing pipeline runs for all four variants...", flush=True)
    comp_ablation = Composer.from_toml(toml_path, pop_seed=42)
    
    # Plotting convergence for the 4 pipelines
    print("[Ablation] Processing history and generating convergence plots...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["full_ga", "selection_only", "crossover_only", "mutation_only"]:
        run_res = comp_ablation.pipelines[name]
        hist = run_res.combined_history()
        df = pd.DataFrame(hist)
        mean_curve = df.groupby('generation')['best_fitness'].mean()
        ax.plot(list(mean_curve.index), list(mean_curve.values), label=name.replace("_", " ").title(), linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Best Fitness")
    ax.set_yscale("log")
    ax.set_title("Ablation Study: Sphere 5D Convergence Trajectories")
    ax.legend()
    fig.tight_layout()
    
    ablation_out = POST_DIR / "ablation"
    ablation_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(ablation_out / "plot_ablation_convergence.png", dpi=150)
    plt.close(fig)

    # Save copies of plots to docs/thesis/images
    shutil.copy(ablation_out / "plot_ablation_convergence.png", IMAGE_DIR / "plot_ablation_convergence.png")
    print("[Ablation] Plots generated and copied successfully.", flush=True)

    # ==========================================
    # PART 2.1: Advanced Real Operators
    # ==========================================
    print("\n--- Starting Part 2.1: Advanced Real Operators (Rastrigin Rotated) ---", flush=True)
    operators_toml = """# Auto-generated advanced operators TOML
[experiment]
name = "rastrigin_operators"
output_dir = "results/sweeps/bbob_parity_sweep/raw/rastrigin_operators"

[experiment.shared]
genome_type = "real"
genome_length = 5
bounds = [-5.0, 5.0]
pop_size = 50
generations = 40
seeds = {seeds}
track_best = "NONE"
use_history_for_final = true

[pipelines.standard_ga]
fitness = "bbob:fn_name=rastrigin_rotated,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 2
selection = "tournament:num_selections=50,tournament_size=3"
crossover = "uniform_real"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"

[pipelines.advanced_ga]
fitness = "bbob:fn_name=rastrigin_rotated,num_dims=5,seed=0,maximize=false"
backend = "malthusjax"
elitism = 2
selection = "elite_pool:num_selections=50,elite_k=10"
crossover = "simulated_binary:eta=20.0"
mutation = "polynomial:mutation_rate=0.1,eta=20.0"
""".format(seeds=SEEDS_50)

    toml_path = CONFIG_DIR / "rastrigin_operators.toml"
    toml_path.write_text(operators_toml)
    comp_ops = Composer.from_toml(toml_path, pop_seed=42)
    
    # Plot convergence comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["standard_ga", "advanced_ga"]:
        hist = comp_ops.pipelines[name].combined_history()
        df = pd.DataFrame(hist)
        mean_curve = df.groupby('generation')['best_fitness'].mean()
        ax.plot(list(mean_curve.index), list(mean_curve.values), label=name.replace("_", " ").title(), linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Best Fitness")
    ax.set_yscale("log")
    ax.set_title("Operator Comparison on Rastrigin Rotated 5D")
    ax.legend()
    fig.tight_layout()
    
    ops_out = POST_DIR / "rastrigin_operators"
    ops_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(ops_out / "plot_operators_convergence.png", dpi=150)
    plt.close(fig)
    shutil.copy(ops_out / "plot_operators_convergence.png", IMAGE_DIR / "plot_operators_convergence.png")

    # Run paired stats between standard and advanced
    spec_ops = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
    )
    row_ops = run_experiment(toml_path, "advanced_ga", "standard_ga", spec_ops, ops_out)
    row_ops["scope"] = "Advanced Operators"
    row_ops["function"] = "rastrigin_rotated"
    row_ops["configuration"] = "advanced_vs_standard"
    master_rows.append(row_ops)

    # ==========================================
    # PART 2.2: Scope 2 - Combinatorial Knapsack
    # ==========================================
    print("\n--- Starting Part 2.2: Combinatorial Knapsack (Binary Genomes) ---", flush=True)
    knapsack_toml = """# Auto-generated Knapsack check TOML
[experiment]
name = "knapsack_experiment"
output_dir = "results/sweeps/bbob_parity_sweep/raw/knapsack"

[experiment.shared]
genome_type = "binary"
genome_length = 20
pop_size = 32
generations = 30
seeds = {seeds}
track_best = "NONE"
use_history_for_final = true
maximize = true

[data.benchmark_knapsack]
source = "synthetic"
num_items = 20
capacity_ratio = 0.5
random_seed = 42

[pipelines.knapsack_single_point]
fitness = "knapsack:data_id=benchmark_knapsack"
backend = "malthusjax"
elitism = 2
selection = "tournament:num_selections=32,tournament_size=3"
crossover = "single_point"
mutation = "bitflip:mutation_rate=0.05"

[pipelines.knapsack_uniform]
fitness = "knapsack:data_id=benchmark_knapsack"
backend = "malthusjax"
elitism = 2
selection = "tournament:num_selections=32,tournament_size=3"
crossover = "uniform_binary"
mutation = "swap:mutation_rate=0.05"
""".format(seeds=SEEDS_20)

    toml_path = CONFIG_DIR / "knapsack.toml"
    toml_path.write_text(knapsack_toml)
    comp_ks = Composer.from_toml(toml_path, pop_seed=42)
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["knapsack_single_point", "knapsack_uniform"]:
        hist = comp_ks.pipelines[name].combined_history()
        df = pd.DataFrame(hist)
        mean_curve = df.groupby('generation')['best_fitness'].mean()
        ax.plot(list(mean_curve.index), list(mean_curve.values), label=name.replace("_", " ").title(), linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Best Value (Maximize)")
    ax.set_title("Combinatorial 0/1 Knapsack Convergence")
    ax.legend()
    fig.tight_layout()
    
    ks_out = POST_DIR / "knapsack"
    ks_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(ks_out / "plot_knapsack_convergence.png", dpi=150)
    plt.close(fig)
    shutil.copy(ks_out / "plot_knapsack_convergence.png", IMAGE_DIR / "plot_knapsack_convergence.png")

    spec_ks = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
    )
    row_ks = run_experiment(toml_path, "knapsack_uniform", "knapsack_single_point", spec_ks, ks_out)
    row_ks["scope"] = "Diverse Genomes"
    row_ks["function"] = "knapsack"
    row_ks["configuration"] = "uniform_vs_single_point"
    master_rows.append(row_ks)

    # ==========================================
    # PART 2.2: Scope 2 - Traveling Salesman Problem (TSP)
    # ==========================================
    print("\n--- Starting Part 2.2: Traveling Salesman Problem (TSP) ---", flush=True)
    tsp_toml = """# Auto-generated TSP check TOML
[experiment]
name = "tsp_experiment"
output_dir = "results/sweeps/bbob_parity_sweep/raw/tsp"

[experiment.shared]
genome_type = "real"
genome_length = 50
bounds = [0.0, 1.0]
pop_size = 32
generations = 30
seeds = {seeds}
track_best = "NONE"
use_history_for_final = true

[data.berlin52]
source = "synthetic"
num_cities = 50
random_seed = 42

[pipelines.tsp_malthusjax]
fitness = "tsp:data_id=berlin52"
backend = "malthusjax"
elitism = 2
selection = "tournament:num_selections=32,tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"

[pipelines.tsp_random]
fitness = "tsp:data_id=berlin52"
backend = "malthusjax"
elitism = 2
selection = "roulette:num_selections=32"
crossover = "uniform_real"
mutation = "gaussian:mutation_rate=0.2,mutation_strength=0.2"
""".format(seeds=SEEDS_20)

    toml_path = CONFIG_DIR / "tsp.toml"
    toml_path.write_text(tsp_toml)
    comp_tsp = Composer.from_toml(toml_path, pop_seed=42)
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["tsp_malthusjax", "tsp_random"]:
        hist = comp_tsp.pipelines[name].combined_history()
        df = pd.DataFrame(hist)
        mean_curve = df.groupby('generation')['best_fitness'].mean()
        ax.plot(list(mean_curve.index), list(mean_curve.values), label=name.replace("_", " ").title(), linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Best Distance (Min)")
    ax.set_yscale("log")
    ax.set_title("TSP Berlin52 (50 Cities) Convergence")
    ax.legend()
    fig.tight_layout()
    
    tsp_out = POST_DIR / "tsp"
    tsp_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(tsp_out / "plot_tsp_convergence.png", dpi=150)
    plt.close(fig)
    shutil.copy(tsp_out / "plot_tsp_convergence.png", IMAGE_DIR / "plot_tsp_convergence.png")

    spec_tsp = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.TWO_SIDED,
        alpha=0.05,
    )
    row_tsp = run_experiment(toml_path, "tsp_malthusjax", "tsp_random", spec_tsp, tsp_out)
    row_tsp["scope"] = "Diverse Genomes"
    row_tsp["function"] = "tsp"
    row_tsp["configuration"] = "tuned_vs_random"
    master_rows.append(row_tsp)

    # ==========================================
    # SAVE MASTER REPORT
    # ==========================================
    print("\n--- Compiling and saving Master Report ---", flush=True)
    master_df = pd.DataFrame(master_rows)
    master_csv = SWEEP_DIR / "master_sweep_report.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"Master Report saved to: {master_csv}", flush=True)

    # Generate master plot (Wilcoxon p-values across functions)
    print("[Plots] Generating BBOB suite Wilcoxon p-values bar chart...", flush=True)
    parity_only = master_df[master_df["scope"] == "BBOB Parity"]
    if not parity_only.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivoted = parity_only.pivot(index="function", columns="configuration", values="primary_p")
        pivoted.plot(kind="bar", ax=ax, edgecolor="black")
        ax.axhline(0.05, color="red", linestyle="--", label="alpha=0.05")
        ax.set_ylabel("Wilcoxon p-value")
        ax.set_title("Wilcoxon Signed-Rank P-Values across BBOB Functions (50 Seeds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(SWEEP_DIR / "plot_bbob_pvalues.png", dpi=150)
        plt.close(fig)
        shutil.copy(SWEEP_DIR / "plot_bbob_pvalues.png", IMAGE_DIR / "plot_bbob_pvalues.png")

    # Copy raw plots of sphere_combo2 to docs/thesis/images as standard parity figures
    print("[Plots] Copying standard sphere_combo2 parity figures...", flush=True)
    sphere_combo2_dir = POST_DIR / "sphere_combo2"
    if sphere_combo2_dir.exists():
        for filename in ["plot_end_scatter.png", "plot_end_ecdf.png", "plot_diff_hist.png"]:
            shutil.copy(sphere_combo2_dir / filename, IMAGE_DIR / filename)
            
    print("===================================================", flush=True)
    print("=== THESIS SUITE COMPLETED SUCCESSFULLY ===", flush=True)
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
