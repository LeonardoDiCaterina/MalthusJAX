"""
Hyperparameter tuning CLI for MalthusJAX fitness optimization.

This script performs a grid search over hyperparameters to find the best
configuration for solution quality (fitness), independent of speed.
"""
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


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Fitness Tuning")
    parser.add_argument("config", help="Path to .toml configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    grid = cfg['grid']
    
    # Build hyperparameter sweep grid
    hyperparam_grid = cfg.get('hyperparam_sweep', {})
    
    # Create all combinations of hyperparameters
    hyperparam_keys = list(hyperparam_grid.keys())
    hyperparam_values = [hyperparam_grid[k] for k in hyperparam_keys]
    hyperparam_combinations = list(itertools.product(*hyperparam_values))
    
    # Build job queue: (algo, task, dim, pop, hyperparam_combo)
    job_queue = list(itertools.product(
        grid['algorithms'],
        grid['tasks'],
        grid['dimensions'],
        grid['pop_sizes'],
        hyperparam_combinations
    ))
    
    repeats = grid.get('repeats', 10)  # Fewer repeats for tuning
    master_seed = grid['seeds'][0]
    generations = grid.get('generations', 500)  # More generations for convergence
    
    print(f"🎯 Starting Fitness Tuning: {exp_name}")
    print(f"📋 Total Configurations: {len(job_queue)}")
    print(f"🔧 Hyperparameter combinations: {len(hyperparam_combinations)}")
    print(f"📊 Repeats per Config: {repeats}")
    print(f"⚙️  Hardware: {jax.devices()[0].device_kind}")
    print("=" * 80)

    results = []
    best_fitness = float('inf')
    best_config = None
    
    for i, (algo, task, dim, pop, hyperparam_tuple) in enumerate(job_queue, 1):
        # Build hyperparameter dict from tuple
        hypers = dict(zip(hyperparam_keys, hyperparam_tuple))
        
        print(f"\n[{i}/{len(job_queue)}] {algo} | {task} | D={dim} | N={pop}")
        print(f"   Hyperparams: {hypers}")

        spec = ComparisonRegistry.get(algo)
        # Merge with algorithm defaults
        full_hypers = {**spec.default_hypers, **hypers}
        
        # Setup problem
        m_eval, _ = setup_bbob_instances(task, dim, master_seed)
        m_adapter = spec.malthus_factory(pop, dim, master_seed, full_hypers, m_eval)

        # Run benchmark
        res = run_adapter_benchmark(
            m_adapter, generations, master_seed, "MalthusJAX", pop, 
            unroll=1, repeats=repeats
        )
        
        # Extract fitness (use absolute value for BBOB)
        mean_fitness = abs(res.best_fitness_final)
        fitness_std = res.fitness_std
        
        print(f"   >>> Mean Fitness: {mean_fitness:.4f} ± {fitness_std:.4f}")
        print(f"   >>> Mean GPS: {res.mean_gps:.2f}")
        
        # Track best configuration
        if mean_fitness < best_fitness:
            best_fitness = mean_fitness
            best_config = {
                'algorithm': algo,
                'task': task,
                'dim': dim,
                'pop_size': pop,
                'hyperparams': hypers,
                'fitness': mean_fitness,
                'fitness_std': fitness_std,
                'gps': res.mean_gps
            }
            print(f"   🌟 NEW BEST FITNESS: {best_fitness:.4f}")
        
        # Store results
        result_row = {
            "Algorithm": algo,
            "Task": task,
            "Dim": dim,
            "Pop_Size": pop,
            "Generations": generations,
            **{f"hyperparam_{k}": v for k, v in hypers.items()},
            "Mean_Fitness": mean_fitness,
            "Fitness_Std": fitness_std,
            "Mean_GPS": res.mean_gps,
            "Compile_Time": res.compile_time,
            "Mean_Exec_Time": res.mean_exec_time,
        }
        results.append(result_row)
        
        # Save incrementally (in case of interruption)
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"fitness_tuning_{timestamp}.csv")
        df.to_csv(filename, index=False)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 TUNING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Best Configuration Found:")
    print(f"   Task: {best_config['task']}")
    print(f"   Dimension: {best_config['dim']}")
    print(f"   Population: {best_config['pop_size']}")
    print(f"   Hyperparameters:")
    for k, v in best_config['hyperparams'].items():
        print(f"      {k}: {v}")
    print(f"\n   Best Fitness: {best_config['fitness']:.6f} ± {best_config['fitness_std']:.6f}")
    print(f"   GPS: {best_config['gps']:.2f}")
    
    # Save final results
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Mean_Fitness')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"fitness_tuning_final_{timestamp}.csv")
    df_sorted.to_csv(filename, index=False)
    
    # Save top 10 configurations
    top10_file = os.path.join(output_dir, f"top10_configs_{timestamp}.csv")
    df_sorted.head(10).to_csv(top10_file, index=False)
    
    print(f"\n✅ Results saved:")
    print(f"   Full results: {filename}")
    print(f"   Top 10 configs: {top10_file}")


if __name__ == "__main__":
    main()
