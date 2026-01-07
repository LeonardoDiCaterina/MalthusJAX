import time
import os
import argparse
import itertools
import numpy as np
import pandas as pd
import jax
from datetime import datetime

from benchmarks.framework.registry import ComparisonRegistry
from benchmarks.framework.runner import run_adapter_benchmark
from benchmarks.framework.adapters import setup_bbob_instances

try:
    import tomllib
except ImportError:
    import tomli as tomllib 

def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)
    
    
def run_single_benchmark(
    algo_name,
    spec,
    hypers,
    task,
    dim,
    pop_size,
    seed,
    unroll,
    generations,
    repeats,
    run_num,
    total_runs,
):
    """Run benchmark for a single algorithm spec (may be Malthus-only or dual)."""
    jax.clear_caches()
    m_eval, e_prob = setup_bbob_instances(task, dim, seed)
    m_adapter = spec.malthus_factory(pop_size, dim, seed, hypers, m_eval)

    res_m = run_adapter_benchmark(m_adapter, generations, seed, "MalthusJAX", pop_size, unroll, repeats)
    
    # Only run Evosax if the spec includes it
    res_e = None
    if spec.evosax_factory is not None:
        e_adapter = spec.evosax_factory(pop_size, dim, seed, hypers, e_prob)
        res_e = run_adapter_benchmark(e_adapter, generations, seed, "Evosax", pop_size, unroll, repeats)

    base = {
        "Algorithm": algo_name,
        "Task": task,
        "Dim": dim,
        "Pop_Size": pop_size,
        "Seed": seed,
        "Unroll": unroll,
        "Generations": generations,
    }

    def package(res, framework_name):
        return {
            **base,
            "Framework": framework_name,
            "Device": getattr(res, "device", "CPU"),
            "Compile_Time": getattr(res, "compile_time", getattr(res, "compile_time", None)),
            "Exec_Time": getattr(res, "mean_exec_time", getattr(res, "execution_time", None)),
            "GPS": getattr(res, "mean_gps", getattr(res, "generations_per_sec", None)),
            "Best_Fitness": getattr(res, "best_fitness_final", getattr(res, "best_fitness", None)),
            "Fitness_Std": getattr(res, "std_exec_time", None),
        }

    results = [package(res_m, "MalthusJAX")]
    if res_e is not None:
        results.append(package(res_e, "Evosax"))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Runner")
    parser.add_argument("config", help="Path to .toml configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    grid = cfg['grid']
    job_queue = list(itertools.product(
        grid['algorithms'], grid['tasks'], grid['dimensions'], 
        grid['pop_sizes'], grid.get('unroll_factors', [1])
    ))
    repeats = grid.get('repeats', 30)
    master_seed = grid['seeds'][0]
    
    print(f"🚀 Starting Benchmark: {exp_name}")
    print(f"📋 Total Configurations: {len(job_queue)}")
    print(f"📊 Repeats per Config:   {repeats}")
    print(f"⚙️  Hardware:             {jax.devices()[0].device_kind}")
    print(f"Available Algorithms:  {list(ComparisonRegistry._registry.keys())}")
    print("=" * 60)

    results = []
    
    for i, (algo, task, dim, pop, unroll) in enumerate(job_queue, 1):
        print(f"\n[{i}/{len(job_queue)}] {algo} | {task} | D={dim} | N={pop} | Unroll={unroll}")

        spec = ComparisonRegistry.get(algo)
        hypers = {**spec.default_hypers, **grid.get('hyperparams', {})}
        
        m_eval, e_prob = setup_bbob_instances(task, dim, master_seed)
        m_adapter = spec.malthus_factory(pop, dim, master_seed, hypers, m_eval)

        res_m = run_adapter_benchmark(m_adapter, grid['generations'], master_seed, "MalthusJAX", pop, unroll, repeats)
        
        base = {"Algorithm": algo, "Task": task, "Dim": dim, "Pop_Size": pop, "Unroll": unroll, "Gens": grid['generations']}
        
        def package(res):
            # Use absolute value for normalized fitness comparison
            # MalthusJAX uses maximization (positive), Evosax uses minimization (negative)
            normalized_fitness = abs(res.best_fitness_final) if res.best_fitness_final is not None else None
            return {
                **base,
                "Framework": res.framework,
                "Mean_GPS": res.mean_gps,
                "Mean_Time": res.mean_exec_time,
                "Std_Time": res.std_exec_time,
                "Compile_Time": res.compile_time,
                "Best_Fitness": res.best_fitness_final,
                "Fitness_Std": res.fitness_std,
                "Normalized_Fitness": normalized_fitness,
            }

        results.append(package(res_m))
        
        # Only run Evosax if spec has it
        if spec.evosax_factory is not None:
            e_adapter = spec.evosax_factory(pop, dim, master_seed, hypers, e_prob)
            res_e = run_adapter_benchmark(e_adapter, grid['generations'], master_seed, "Evosax", pop, unroll, repeats)
            
            # Log & Report (dual framework) - use absolute values for comparison
            speedup = res_m.mean_gps / res_e.mean_gps
            mjx_fit_norm = abs(res_m.best_fitness_final)
            evosax_fit_norm = abs(res_e.best_fitness_final)
            fit_diff = abs(mjx_fit_norm - evosax_fit_norm)
            fit_match = "✓" if fit_diff < 1.0 else "✗"
            print(f"   >>> MalthusJAX Mean GPS: {res_m.mean_gps:.2f}")
            print(f"   >>> Evosax Mean GPS:     {res_e.mean_gps:.2f}")
            print(f"   >>> Speedup: {speedup:.2f}x")
            print(f"   >>> Fitness (|val|): MJX={mjx_fit_norm:.2e}±{res_m.fitness_std:.2e} | Evosax={evosax_fit_norm:.2e}±{res_e.fitness_std:.2e} {fit_match}")
            results.append(package(res_e))
        else:
            # Log & Report (Malthus-only)
            print(f"   >>> [Malthus-Only] Mean GPS: {res_m.mean_gps:.2f}")
            print(f"   >>> Fitness: {abs(res_m.best_fitness_final):.2e}±{res_m.fitness_std:.2e}")

    # Save
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"final_benchmark_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"\n✅ Benchmark Complete. CSV: {filename}")

if __name__ == "__main__":
    main()