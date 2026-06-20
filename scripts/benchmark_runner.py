#!/usr/bin/env python3
"""Unified TOML-Driven Benchmark Runner.

Parses a `.toml` suite definition, generates the experimental grid (Cartesian or LHS),
and securely executes the benchmarking pipelines via the MalthusJAX Composer API.
Includes heavy memory protections (JAX cache clearing) and artifact pruning
to support massive multi-day cluster executions without Out-Of-Memory errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Prevent JAX NCCL multi-device rendezvous deadlocks on cluster by restricting to 1 GPU
if "CUDA_VISIBLE_DEVICES" in os.environ:
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Prevent cuSolver OOM errors by disabling aggressive memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.benchmarking.sampling import generate_grid
from malthusjax.composer import Composer


def run_suite(toml_path: str, force_smoke: bool = False) -> None:
    # 1. Parse configuration
    config = BenchmarkConfig.from_toml(toml_path)
    
    output_dir = Path(config.suite.output_dir)
    if force_smoke:
        output_dir = Path(str(output_dir) + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_seeds = 3 if force_smoke else config.suite.num_seeds
    seeds = list(range(1, num_seeds + 1))
    
    # Copy the TOML to the output directory for reproducibility
    import shutil
    shutil.copy2(toml_path, output_dir / "suite_config.toml")
    
    # 2. Generate Coordinate Grid
    grid_coords = generate_grid(config)
    
    if force_smoke:
        # Just take the first 2 coordinates for a smoke test
        grid_coords = grid_coords[:2]
        
    total_experiments = len(grid_coords)
    
    print(f"\n{'#'*70}")
    print(f"  BENCHMARK SUITE: {config.suite.name}")
    print(f"  Mode: {config.suite.mode.upper()}")
    print(f"  Total Coordinates: {total_experiments}")
    print(f"  Seeds per Coordinate: {len(seeds)} (Total Runs: {total_experiments * len(seeds)})")
    print(f"  Aggregation Level: {config.suite.aggregation_level}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    composer = Composer.create_default()
    all_results = []
    suite_start = time.time()
    
    # 3. Execute Loop
    for i, coord in enumerate(grid_coords, 1):
        fn_name = coord["fn_name"]
        D = coord["D"]
        P = coord["P"]
        G = coord["G"]
        
        if config.suite.mode == "lhs":
            lhs_id = coord["lhs_id"]
            experiment_name = f"exp_{fn_name}_{lhs_id}_d{D}_p{P}_g{G}"
        else:
            experiment_name = f"exp_{fn_name}_d{D}_p{P}_g{G}"
            
        exp_output = output_dir / experiment_name
        exp_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n>>> Experiment {i}/{total_experiments} | {experiment_name}")
        
        # Build pipelines dynamically from TOML definitions
        # The Composer handles substituting dynamic variables if we provided a formatter
        # Wait, the TOML has `{pop_size}` in strings! We need to format them.
        formatted_pipelines = {}
        for p_name, p_kwargs in config.pipelines.items():
            formatted_kwargs = {}
            for k, v in p_kwargs.items():
                if isinstance(v, str):
                    formatted_val = v.format(
                        pop_size=P, 
                        genome_length=D, 
                        generations=G, 
                        elite_k=max(2, int(P / 6))
                    )
                    formatted_kwargs[k] = int(formatted_val) if formatted_val.isdigit() else formatted_val
                else:
                    formatted_kwargs[k] = v
            formatted_pipelines[p_name] = formatted_kwargs
            
        shared_kwargs = {
            "fitness": f"bbob:fn_name={fn_name},num_dims={D},maximize=false",
            "pop_size": P,
            "generations": G,
            "genome_length": D,
            "bounds": (-5.0, 5.0),
            "maximize": False,
        }
        
        t0 = time.time()
        try:
            comparison = composer.compare(
                pipelines=formatted_pipelines,
                seeds=seeds,
                shared_initial_population=True,
                pop_seed=42,
                **shared_kwargs,
            )
            wall_time = time.time() - t0
            
            # Prune and Format Results safely into python primitives
            result_summary = {
                "experiment": experiment_name,
                "config": coord,
                "wall_time_seconds": wall_time,
                "pipelines": {},
            }
            
            for p_name, exp_result in comparison.pipelines.items():
                pipeline_data = {
                    "num_runs": len(exp_result.runs),
                    "successful_runs": sum(1 for r in exp_result.runs if r.status == "success"),
                    "per_seed": [],
                }
                
                for run in exp_result.runs:
                    best_fit = run.metrics.get("best_fitness")
                    final_gen = run.metrics.get("final_generation")
                    tot_evals = run.metrics.get("total_evaluations")
                    
                    seed_data = {
                        "seed": run.seed,
                        "status": run.status,
                        "duration_seconds": run.duration_seconds,
                        "best_fitness": float(best_fit) if best_fit is not None else None,
                        "final_generation": int(final_gen) if final_gen is not None else None,
                        "total_evaluations": int(tot_evals) if tot_evals is not None else None,
                    }
                    if run.timings:
                        seed_data["timings"] = run.timings
                        
                    # Handle Aggregation Level
                    if config.suite.aggregation_level in ["full_trace", "final_only"]:
                        if run.history:
                            history_to_save = run.history
                            if config.suite.aggregation_level == "final_only":
                                history_to_save = [run.history[-1]] if len(run.history) > 0 else []
                                
                            seed_data["convergence"] = [
                                {
                                    "generation": int(h.get("generation")) if h.get("generation") is not None else None,
                                    "best_fitness": float(h.get("best_fitness")) if h.get("best_fitness") is not None else None,
                                }
                                for h in history_to_save
                            ]
                    
                    pipeline_data["per_seed"].append(seed_data)
                
                result_summary["pipelines"][p_name] = pipeline_data
            
            # Clear GPU memory caches to prevent sequential out-of-memory errors
            import jax
            import gc
            jax.clear_caches()
            gc.collect()
            
            # Save artifact
            result_file = exp_output / "benchmark_results.json"
            with open(result_file, "w") as f:
                json.dump(result_summary, f, indent=2, default=str)
                
            all_results.append({
                "experiment": experiment_name,
                "status": "success",
                "wall_time": wall_time
            })
            
        except Exception as e:
            print(f"  ERROR executing {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "experiment": experiment_name,
                "status": "error",
                "error": str(e)
            })

    suite_time = time.time() - suite_start
    
    suite_summary = {
        "suite": config.suite.name,
        "mode": config.suite.mode,
        "total_experiments": total_experiments,
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "error"),
        "total_wall_time_seconds": suite_time,
        "experiments": all_results,
    }

    suite_file = output_dir / "suite_summary.json"
    with open(suite_file, "w") as f:
        json.dump(suite_summary, f, indent=2, default=str)

    print(f"\n{'#'*70}")
    print(f"  SUITE COMPLETE")
    print(f"  Successful: {suite_summary['successful']}/{total_experiments}")
    print(f"  Failed: {suite_summary['failed']}/{total_experiments}")
    print(f"  Total wall time: {suite_time:.1f}s")
    print(f"  Output Directory: {output_dir}")
    print(f"{'#'*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified TOML-Driven Benchmark Runner")
    parser.add_argument("--toml", type=str, required=True, help="Path to the TOML configuration file")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test (2 coords, 3 seeds)")
    args = parser.parse_args()
    
    run_suite(args.toml, force_smoke=args.smoke)


if __name__ == "__main__":
    main()
