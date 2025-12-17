import argparse
import os
import time
import pandas as pd
import jax
import sys
from datetime import datetime
from typing import Any

# Handle TOML parsing (Python 3.11+ native, else tomli)
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# --- Internal Imports ---
# Ensure we can import from the local project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks.framework.registry import ComparisonRegistry
from benchmarks.framework.runner import run_adapter_benchmark

# Malthus BBOB Factory (Serves as the Single Source of Truth for Problem Def)
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def setup_bbob_instances(task_name: str, dim: int, seed: int):
    """
    Creates a synchronized pair of problem evaluators.
    We create the Malthus evaluator first, then extract its internal
    Evosax problem to ensure 100% alignment.
    """
    # 1. Create Malthus Config
    config = BBOBConfig(
        fn_name=task_name,
        num_dims=dim,
        seed=seed,
        maximize=True,  # Malthus assumes Maximization
    )

    # 2. Instantiate Malthus Evaluator
    malthus_evaluator = BBOBEvaluator.create(config)

    # 3. Extract the underlying Evosax Problem
    evosax_problem = malthus_evaluator.evosax_problem

    return malthus_evaluator, evosax_problem


def run_single_benchmark(
    algo_name: str,
    spec: Any,
    hypers: dict,
    task: str,
    dim: int,
    pop_size: int,
    seed: int,
    unroll: int,
    generations: int,
    repeats: int,
    run_num: int,
    total_runs: int,
) -> tuple[dict, dict]:
    """
    Run a single benchmark comparison between MalthusJAX and Evosax.
    
    Returns:
        Tuple of (malthus_record, evosax_record) dictionaries for CSV logging.
    """
    print(
        f"\n[{run_num}/{total_runs}] Comparing {algo_name} on {task} "
        f"(D={dim}, N={pop_size}, Unroll={unroll})..."
    )

    # A. Instantiate Problems
    m_eval, e_prob = setup_bbob_instances(task, dim, seed)

    # B. Build Malthus Adapter
    m_adapter = spec.malthus_factory(
        pop_size=pop_size,
        dims=dim,
        seed=seed,
        hypers=hypers,
        problem_evaluator=m_eval,
    )

    # C. Build Evosax Adapter
    e_adapter = spec.evosax_factory(
        pop_size=pop_size,
        dims=dim,
        seed=seed,
        hypers=hypers,
        problem_object=e_prob,
    )

    # D. Run Benchmarks
    res_m = run_adapter_benchmark(
        m_adapter,
        generations,
        seed,
        "MalthusJAX",
        unroll=unroll,
        repeats=repeats,
    )

    res_e = run_adapter_benchmark(
        e_adapter,
        generations,
        seed,
        "Evosax",
        unroll=unroll,
        repeats=repeats,
    )

    # E. Build Result Records
    base_record = {
        "Algorithm": algo_name,
        "Task": task,
        "Dim": dim,
        "Pop_Size": pop_size,
        "Generations": generations,
        "Seed": seed,
        "Unroll": unroll,
        "Device": res_m.device,
    }

    rec_m = {
        **base_record,
        "Framework": "MalthusJAX",
        "Compile_Time": res_m.compile_time,
        "Exec_Time": res_m.execution_time,
        "GPS": res_m.generations_per_sec,
        "Best_Fitness": res_m.best_fitness,
        "Fitness_Std": res_m.fitness_std,
    }

    rec_e = {
        **base_record,
        "Framework": "Evosax",
        "Compile_Time": res_e.compile_time,
        "Exec_Time": res_e.execution_time,
        "GPS": res_e.generations_per_sec,
        "Best_Fitness": res_e.best_fitness,
        "Fitness_Std": res_e.fitness_std,
    }

    # F. Print Quick Stat
    speedup = res_m.generations_per_sec / res_e.generations_per_sec
    fitness_diff = res_m.best_fitness - res_e.best_fitness
    print(
        f"   >>> Speedup: {speedup:.2f}x "
        f"(Malthus={res_m.generations_per_sec:.0f}, "
        f"Evosax={res_e.generations_per_sec:.0f} GPS)"
    )
    print(
        f"   >>> Fitness: MalthusJAX={res_m.best_fitness:.4e}±{res_m.fitness_std:.4e}, "
        f"Evosax={res_e.best_fitness:.4e}±{res_e.fitness_std:.4e}, "
        f"Δ={fitness_diff:+.4e}"
    )

    # G. Clean up memory to prevent accumulation across runs
    del m_adapter, e_adapter, m_eval, e_prob, res_m, res_e
    jax.clear_caches()

    return rec_m, rec_e


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX vs Evosax Benchmark Runner")
    parser.add_argument("config", help="Path to .toml configuration file")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Read repeats from the experiment config (used to average execution timings)
    repeats = cfg.get('experiment', {}).get('repeats', 1)
    # Get unroll factors to sweep (defaults to single factor of 1)
    grid = cfg['grid']
    unroll_factors = grid.get('unroll_factors', [1])

    print(f"🚀 Starting Benchmark Suite: {exp_name}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"⚙️  Device: {jax.devices()[0].device_kind}")

    results = []

    # 3. Generate all parameter combinations and run benchmarks
    total_runs = (
        len(grid['algorithms'])
        * len(grid['tasks'])
        * len(grid['dimensions'])
        * len(grid['pop_sizes'])
        * len(grid['seeds'])
        * len(unroll_factors)
    )

    current_run = 0

    for algo_name in grid['algorithms']:
        spec = ComparisonRegistry.get(algo_name)
        hypers = {**spec.default_hypers, **grid.get('hyperparams', {})}

        for task in grid['tasks']:
            for dim in grid['dimensions']:
                for pop_size in grid['pop_sizes']:
                    for seed in grid['seeds']:
                        for unroll in unroll_factors:
                            current_run += 1
                            
                            rec_m, rec_e = run_single_benchmark(
                                algo_name=algo_name,
                                spec=spec,
                                hypers=hypers,
                                task=task,
                                dim=dim,
                                pop_size=pop_size,
                                seed=seed,
                                unroll=unroll,
                                generations=grid['generations'],
                                repeats=repeats,
                                run_num=current_run,
                                total_runs=total_runs,
                            )
                            
                            results.append(rec_m)
                            results.append(rec_e)

    # 4. Save Final Data
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    df.to_csv(filename, index=False)

    print(f"\n✅ Benchmark Complete. Results saved to: {filename}")

    # 5. Print Summary Table (Pop Size Scaling)
    print("\n=== Scaling Summary (Mean GPS) ===")
    summary = df.groupby(["Framework", "Pop_Size"])["GPS"].mean().unstack().T
    print(summary)
    
    # 6. Print Fitness Comparison Summary
    print("\n=== Fitness Quality Comparison ===")
    fitness_summary = df.groupby(["Framework", "Task"])[["Best_Fitness", "Fitness_Std"]].mean()
    print(fitness_summary)
    
    # Calculate statistical significance of fitness differences
    print("\n=== Fitness Difference by Task ===")
    for task in df["Task"].unique():
        task_data = df[df["Task"] == task]
        m_fitness = task_data[task_data["Framework"] == "MalthusJAX"]["Best_Fitness"].values
        e_fitness = task_data[task_data["Framework"] == "Evosax"]["Best_Fitness"].values
        
        if len(m_fitness) > 0 and len(e_fitness) > 0:
            diff = m_fitness.mean() - e_fitness.mean()
            rel_diff = (diff / abs(e_fitness.mean())) * 100 if e_fitness.mean() != 0 else 0
            print(f"{task:15s}: Δ={diff:+.4e} ({rel_diff:+.2f}%)")


if __name__ == "__main__":
    main()
