"""
Hyperparameter tuning CLI for MalthusJAX fitness optimization.

This script performs a grid search over hyperparameters to find the best
configuration for solution quality (fitness), independent of speed.
"""
import argparse
import itertools
import os
from pathlib import Path

try:
    import pandas as pd
except Exception as e:
    pd = None
    print(f"⚠️  Warning: pandas import failed: {e}. Result saving will use a simple CSV writer.")
from datetime import datetime

import jax

from benchmarks.framework.adapters import setup_bbob_instances
from benchmarks.framework.registry import ComparisonRegistry
from benchmarks.framework.runner import run_adapter_benchmark

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

    # Build job queue: include seeds so we can repeat each config across multiple seeds
    job_queue = list(itertools.product(
        grid['algorithms'],
        grid['tasks'],
        grid['dimensions'],
        grid['pop_sizes'],
        grid.get('seeds', [42]),
        hyperparam_combinations
    ))

    repeats = grid.get('repeats', 10)  # Fewer repeats for tuning
    seeds_list = grid.get('seeds', [42])
    generations = grid.get('generations', 500)  # More generations for convergence

    print(f"🎯 Starting Fitness Tuning: {exp_name}")
    print(f"📋 Total Configurations: {len(job_queue)}")
    print(f"🔧 Hyperparameter combinations: {len(hyperparam_combinations)}")
    print(f"🔢 Seeds: {seeds_list}")
    print(f"📊 Repeats per Config: {repeats}")
    print(f"⚙️  Hardware: {jax.devices()[0].device_kind}")
    print("=" * 80)

    results = []
    best_fitness = float('inf')
    best_config = None

    for i, (algo, task, dim, pop, seed, hyperparam_tuple) in enumerate(job_queue, 1):
        # Build hyperparameter dict from tuple
        hypers = dict(zip(hyperparam_keys, hyperparam_tuple))

        print(f"\n[{i}/{len(job_queue)}] {algo} | {task} | D={dim} | N={pop}")
        print(f"   Hyperparams: {hypers}")

        spec = ComparisonRegistry.get(algo)
        # DEBUG: show which factory will be used to build the adapter
        print(f"   >>> Using factory: {spec.malthus_factory.__name__}")
        # Merge with algorithm defaults
        full_hypers = {**spec.default_hypers, **hypers}

        # Setup problem (use job-specific seed derived from provided seed)
        job_seed = seed + i
        m_eval, es_problem = setup_bbob_instances(task, dim, job_seed)
        m_adapter = spec.malthus_factory(pop, dim, job_seed, full_hypers, m_eval)

        # DEBUG: show that hyperparams were applied to the adapter/engine
        print(f"   >>> job_seed: {job_seed}")
        print(f"   >>> full_hypers passed: {full_hypers}")
        try:
            print(f"   >>> Engine mutation: {m_adapter.engine.mutation}")
            print(f"   >>> Engine crossover: {m_adapter.engine.crossover}")
            print(f"   >>> Engine selection: {m_adapter.engine.selection}")
        except Exception as e:
            print(f"   >>> Debug print failed: {e}")

        # Run MalthusJAX benchmark
        res_m = run_adapter_benchmark(
            m_adapter, generations, job_seed, "MalthusJAX", pop,
            unroll_factor=1, repeats=repeats
        )

        # Extract fitness (use absolute value for BBOB parity)
        mean_fitness_m = abs(res_m.best_fitness_final)
        fitness_std_m = res_m.fitness_std

        print(f"   >>> MJX Mean Fitness: {mean_fitness_m:.4f} ± {fitness_std_m:.4f}")
        print(f"   >>> MJX Mean GPS: {res_m.mean_gps:.2f}")

        # Track best configuration (based on MJX results)
        if mean_fitness_m < best_fitness:
            best_fitness = mean_fitness_m
            best_config = {
                'algorithm': algo,
                'task': task,
                'dim': dim,
                'pop_size': pop,
                'hyperparams': hypers,
                'fitness': mean_fitness_m,
                'fitness_std': fitness_std_m,
                'gps': res_m.mean_gps
            }
            print(f"   🌟 NEW BEST FITNESS: {best_fitness:.4f}")

        # Store MJX results
        result_row_m = {
            "Framework": "MalthusJAX",
            "Algorithm": algo,
            "Task": task,
            "Dim": dim,
            "Pop_Size": pop,
            "Seed": seed,
            "Job_Seed": job_seed,
            "Generations": generations,
            **{f"hyperparam_{k}": v for k, v in hypers.items()},
            "Mean_Fitness": mean_fitness_m,
            "Fitness_Std": fitness_std_m,
            "Mean_GPS": res_m.mean_gps,
            "Compile_Time": res_m.compile_time,
            "Mean_Exec_Time": res_m.mean_exec_time,
        }
        results.append(result_row_m)

        # Run Evosax (same job seed for comparability)
        if spec.evosax_factory is not None:
            try:
                e_adapter = spec.evosax_factory(pop, dim, job_seed, full_hypers, es_problem)
                print(f"   >>> Evosax strategy: {e_adapter.strategy}")
                print(f"   >>> Evosax params: {e_adapter.params}")

                res_e = run_adapter_benchmark(
                    e_adapter, generations, job_seed, "Evosax", pop,
                    unroll_factor=1, repeats=repeats
                )
                mean_fitness_e = abs(res_e.best_fitness_final)
                fitness_std_e = res_e.fitness_std

                print(f"   >>> Evosax Mean Fitness: {mean_fitness_e:.4f} ± {fitness_std_e:.4f}")
                print(f"   >>> Evosax Mean GPS: {res_e.mean_gps:.2f}")

                result_row_e = {
                    "Framework": "Evosax",
                    "Algorithm": algo,
                    "Task": task,
                    "Dim": dim,
                    "Pop_Size": pop,
                    "Seed": seed,
                    "Job_Seed": job_seed,
                    "Generations": generations,
                    **{f"hyperparam_{k}": v for k, v in hypers.items()},
                    "Mean_Fitness": mean_fitness_e,
                    "Fitness_Std": fitness_std_e,
                    "Mean_GPS": res_e.mean_gps,
                    "Compile_Time": res_e.compile_time,
                    "Mean_Exec_Time": res_e.mean_exec_time,
                }
                results.append(result_row_e)

                # Update best_config based on Evosax only if better than MJX best
                if mean_fitness_e < best_fitness:
                    best_fitness = mean_fitness_e
                    best_config = {
                        'algorithm': algo,
                        'task': task,
                        'dim': dim,
                        'pop_size': pop,
                        'hyperparams': hypers,
                        'fitness': mean_fitness_e,
                        'fitness_std': fitness_std_e,
                        'gps': res_e.mean_gps
                    }
                    print(f"   🌟 NEW BEST FITNESS: {best_fitness:.4f} (Evosax)")

            except Exception as e:
                print(f"   >>> Evosax run failed: {e}")

        # Save incrementally (in case of interruption)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"fitness_tuning_{timestamp}.csv")
        if pd is not None:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
        else:
            import csv
            if results:
                keys = list(results[0].keys())
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, keys)
                    writer.writeheader()
                    writer.writerows(results)

    # Final summary
    print("\n" + "=" * 80)
    print("🏆 TUNING COMPLETE")
    print("=" * 80)
    print("\n📊 Best Configuration Found:")
    print(f"   Task: {best_config['task']}")
    print(f"   Dimension: {best_config['dim']}")
    print(f"   Population: {best_config['pop_size']}")
    print("   Hyperparameters:")
    for k, v in best_config['hyperparams'].items():
        print(f"      {k}: {v}")
    print(f"\n   Best Fitness: {best_config['fitness']:.6f} ± {best_config['fitness_std']:.6f}")
    print(f"   GPS: {best_config['gps']:.2f}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"fitness_tuning_final_{timestamp}.csv")
    if pd is not None:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('Mean_Fitness')
        df_sorted.to_csv(filename, index=False)
        # Save top 10 configurations
        top10_file = os.path.join(output_dir, f"top10_configs_{timestamp}.csv")
        df_sorted.head(10).to_csv(top10_file, index=False)
    else:
        import csv
        if results:
            keys = list(results[0].keys())
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # full results
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(results)
            # top10
            top10_file = os.path.join(output_dir, f"top10_configs_{timestamp}.csv")
            with open(top10_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(results[:10])

    print("\n✅ Results saved:")
    print(f"   Full results: {filename}")
    print(f"   Top 10 configs: {top10_file}")


if __name__ == "__main__":
    main()
