import time
import os
import argparse
import itertools
import numpy as np
import pandas as pd
import jax
from datetime import datetime

# Import internal modules
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
    print("=" * 60)

    results = []
    
    for i, (algo, task, dim, pop, unroll) in enumerate(job_queue, 1):
        print(f"\n[{i}/{len(job_queue)}] {task} | D={dim} | N={pop} | Unroll={unroll}")

        spec = ComparisonRegistry.get(algo)
        hypers = {**spec.default_hypers, **grid.get('hyperparams', {})}
        
        m_eval, e_prob = setup_bbob_instances(task, dim, master_seed)
        m_adapter = spec.malthus_factory(pop, dim, master_seed, hypers, m_eval)
        e_adapter = spec.evosax_factory(pop, dim, master_seed, hypers, e_prob)

        res_m = run_adapter_benchmark(m_adapter, grid['generations'], master_seed, "MalthusJAX", pop, unroll, repeats)
        res_e = run_adapter_benchmark(e_adapter, grid['generations'], master_seed, "Evosax", pop, unroll, repeats)

        # Log & Report
        speedup = res_m.mean_gps / res_e.mean_gps
        print(f"   >>> MalthusJAX Mean GPS: {res_m.mean_gps:.2f}")
        print(f"   >>> Evosax Mean GPS:     {res_e.mean_gps:.2f}")
        print(f"   >>> Speedup: {speedup:.2f}x")
        print(f"   >>> Final Fit (average): Malthus={res_m.best_fitness_final:.2e} | Evosax={res_e.best_fitness_final:.2e}")
        
        base = {"Algorithm": algo, "Task": task, "Dim": dim, "Pop_Size": pop, "Unroll": unroll, "Gens": grid['generations']}
        
        def package(res):
            return {
                **base,
                "Framework": res.framework,
                "Mean_GPS": res.mean_gps,
                "Mean_Time": res.mean_exec_time,
                "Compile_Time": res.compile_time,
                "Best_Fitness": res.best_fitness_final # SAVED HERE
            }

        results.append(package(res_m))
        results.append(package(res_e))

    # Save
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"final_benchmark_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"\n✅ Benchmark Complete. CSV: {filename}")

if __name__ == "__main__":
    main()