import time
import os
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd
import jax
from datetime import datetime

# Import your internal modules
from benchmarks.framework.registry import ComparisonRegistry
from benchmarks.framework.runner import run_adapter_benchmark
from benchmarks.framework.adapters import setup_bbob_instances

# TOML Loader
try:
    import tomllib
except ImportError:
    import tomli as tomllib 

def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)

def main():
    parser = argparse.ArgumentParser(description="MalthusJAX vs Evosax Benchmark Runner")
    parser.add_argument("config", help="Path to .toml configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid Parameters
    grid = cfg['grid']
    
    # 1. Define the Parameter Space
    algorithms = grid['algorithms']
    tasks = grid['tasks']
    dimensions = grid['dimensions']
    pop_sizes = grid['pop_sizes']
    unroll_factors = grid.get('unroll_factors', [1])
    repeats = grid.get('repeats', 30)
    master_seed = grid['seeds'][0]

    # 2. Create the Cartesian Product (Flattened Job List)
    # This creates a single list of tuples: (algo, task, dim, pop, unroll)
    job_queue = list(itertools.product(algorithms, tasks, dimensions, pop_sizes, unroll_factors))
    total_jobs = len(job_queue)
    
    print(f"🚀 Starting Benchmark: {exp_name}")
    print(f"📋 Total Configurations: {total_jobs}")
    print(f"📊 Repeats per Config:   {repeats}")
    print(f"⚙️  Hardware:             {jax.devices()[0].device_kind}")
    print("=" * 60)

    results = []
    
    # 3. Iterate linearly
    for i, (algo_name, task, dim, pop_size, unroll) in enumerate(job_queue, 1):
        print(f"\n[{i}/{total_jobs}] {task} | D={dim} | N={pop_size} | Unroll={unroll}")

        # --- A. Setup ---
        spec = ComparisonRegistry.get(algo_name)
        hypers = {**spec.default_hypers, **grid.get('hyperparams', {})}
        
        m_eval, e_prob = setup_bbob_instances(task, dim, master_seed)
        m_adapter = spec.malthus_factory(pop_size, dim, master_seed, hypers, m_eval)
        e_adapter = spec.evosax_factory(pop_size, dim, master_seed, hypers, e_prob)

        # --- B. Run Benchmark (Compile Once, Run N times) ---
        # MalthusJAX
        res_m = run_adapter_benchmark(
            m_adapter, grid['generations'], master_seed, "MalthusJAX",
            pop_size=pop_size, unroll_factor=unroll, repeats=repeats
        )
        
        # Evosax
        res_e = run_adapter_benchmark(
            e_adapter, grid['generations'], master_seed, "Evosax",
            pop_size=pop_size, unroll_factor=unroll, repeats=repeats
        )

        # --- C. Log & Report ---
        speedup = res_m.mean_gps / res_e.mean_gps
        print(f"   >>> Result: Malthus={res_m.mean_gps:.0f} GPS | Evosax={res_e.mean_gps:.0f} GPS")
        print(f"   >>> Speedup: {speedup:.2f}x")
        
        # Helper to package result row
        base_rec = {
            "Algorithm": algo_name, "Task": task, "Dim": dim,
            "Pop_Size": pop_size, "Unroll": unroll, "Gens": grid['generations']
        }
        
        def package(res):
            return {
                **base_rec,
                "Framework": res.framework,
                "Mean_GPS": res.mean_gps,
                "Mean_Time": res.mean_exec_time,
                "Std_Time": res.std_exec_time,
                "Compile_Time": res.compile_time,
                "Device": res.device
            }

        results.append(package(res_m))
        results.append(package(res_e))

    # 4. Save Final
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"final_benchmark_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"\n✅ Benchmark Complete. Saved to: {filename}")

if __name__ == "__main__":
    main()