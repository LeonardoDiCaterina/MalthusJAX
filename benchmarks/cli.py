import argparse
import os
import time
import pandas as pd
import jax
import sys
from datetime import datetime

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
    # Map common names to IDs if necessary, or pass string
    config = BBOBConfig(
        fn_name=task_name,
        num_dims=dim,
        seed=seed,
        maximize=True # Malthus assumes Maximization
    )
    
    # 2. Instantiate Malthus Evaluator
    malthus_evaluator = BBOBEvaluator.create(config)
    
    # 3. Extract the underlying Evosax Problem
    # (The Registry expects this to pass to the Evosax Adapter)
    evosax_problem = malthus_evaluator.evosax_problem
    
    return malthus_evaluator, evosax_problem

def main():
    parser = argparse.ArgumentParser(description="MalthusJAX vs Evosax Benchmark Runner")
    parser.add_argument("config", help="Path to .toml configuration file")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    exp_name = cfg['experiment']['name']
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Starting Benchmark Suite: {exp_name}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"⚙️  Device: {jax.devices()[0].device_kind}")

    # 2. Extract Grid
    grid = cfg['grid']
    results = []
    
    # 3. The Grand Loop
    total_runs = (
        len(grid['algorithms']) * len(grid['tasks']) * len(grid['dimensions']) * len(grid['pop_sizes']) * len(grid['seeds'])
    )
    current_run = 0

    for algo_name in grid['algorithms']:
        # Get the Fight Card from Registry
        spec = ComparisonRegistry.get(algo_name)
        
        # Merge TOML hyperparams with Spec defaults
        hypers = {**spec.default_hypers, **grid.get('hyperparams', {})}

        for task in grid['tasks']:
            for dim in grid['dimensions']:
                for pop_size in grid['pop_sizes']:
                    for seed in grid['seeds']:
                        current_run += 1
                        print(f"\n[{current_run}/{total_runs}] Comparing {algo_name} on {task} (D={dim}, N={pop_size})...")

                        # A. Instantiate Problems
                        m_eval, e_prob = setup_bbob_instances(task, dim, seed)

                        # B. Build Malthus Adapter
                        m_adapter = spec.malthus_factory(
                            pop_size=pop_size, 
                            dims=dim, 
                            seed=seed, 
                            hypers=hypers, 
                            problem_evaluator=m_eval
                        )

                        # C. Build Evosax Adapter
                        e_adapter = spec.evosax_factory(
                            pop_size=pop_size, 
                            dims=dim, 
                            seed=seed, 
                            hypers=hypers, 
                            problem_object=e_prob
                        )

                        # D. Run Benchmarks
                        # 1. MalthusJAX
                        res_m = run_adapter_benchmark(
                            m_adapter, 
                            grid['generations'], 
                            seed, 
                            "MalthusJAX"
                        )
                        
                        # 2. Evosax
                        res_e = run_adapter_benchmark(
                            e_adapter, 
                            grid['generations'], 
                            seed, 
                            "Evosax"
                        )

                        # E. Log Results
                        base_record = {
                            "Algorithm": algo_name,
                            "Task": task,
                            "Dim": dim,
                            "Pop_Size": pop_size,
                            "Generations": grid['generations'],
                            "Seed": seed,
                            "Device": res_m.device
                        }
                        
                        # Flatten Malthus Metrics
                        rec_m = {**base_record, "Framework": "MalthusJAX", 
                                 "Compile_Time": res_m.compile_time,
                                 "Exec_Time": res_m.execution_time,
                                 "GPS": res_m.generations_per_sec,
                                 "Best_Fitness": res_m.best_fitness}
                        
                        # Flatten Evosax Metrics
                        rec_e = {**base_record, "Framework": "Evosax",
                                 "Compile_Time": res_e.compile_time,
                                 "Exec_Time": res_e.execution_time,
                                 "GPS": res_e.generations_per_sec,
                                 "Best_Fitness": res_e.best_fitness}

                        results.append(rec_m)
                        results.append(rec_e)
                        
                        # Print Quick Stat
                        speedup = res_m.generations_per_sec / res_e.generations_per_sec
                        print(f"   >>> Speedup: {speedup:.2f}x (Malthus={res_m.generations_per_sec:.0f}, Evosax={res_e.generations_per_sec:.0f} GPS)")

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

if __name__ == "__main__":
    main()