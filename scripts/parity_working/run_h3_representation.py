#!/usr/bin/env python3
"""H3 Representation: Domain Generality across Numerical Precisions.

This script tests whether MalthusJAX's architecture can seamlessly handle
different numerical precisions (bfloat16, float16) without degrading
solution quality compared to the native default (float32).

Pipelines:
  1. mjx_f32    — MalthusJAX baseline (float32)
  2. mjx_bf16   — MalthusJAX bfloat16 variant
  3. mjx_f16    — MalthusJAX float16 variant

Usage:
    # Smoke test (local, ~30 seconds)
    python scripts/parity_working/run_h3_representation.py --smoke

    # Full run (cluster)
    python scripts/parity_working/run_h3_representation.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Prevent JAX NCCL multi-device rendezvous deadlocks on cluster by restricting to 1 GPU
if "CUDA_VISIBLE_DEVICES" in os.environ:
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_DEFAULTS = {
    "functions": ["sphere"],
    "dims": [5],
    "pops": [32],
    "gens": [10],
    "num_seeds": 3,
    "output_dir": "results/h3_representation_smoke",
}

FULL_DEFAULTS = {
    "functions": ["sphere", "rosenbrock", "rastrigin"],
    "dims": [10, 50, 100],
    "pops": [64, 256, 1024],
    "gens": [50, 200],
    "num_seeds": 100,
    "output_dir": "results/h3_representation",
}


# ---------------------------------------------------------------------------
# Core: run a single representation experiment
# ---------------------------------------------------------------------------


def run_single_representation(
    fn_name: str,
    num_dims: int,
    pop_size: int,
    generations: int,
    seeds: list[int],
    output_dir: Path,
    elite_ratio: float = 1 / 6,
) -> dict[str, Any]:
    """Run one H3 representation comparison.

    Returns a summary dict with convergence + timing data for all 3 pipelines.
    """
    from malthusjax.composer import Composer

    elite_k = max(2, int(pop_size * elite_ratio))
    experiment_name = f"h3_{fn_name}_d{num_dims}_p{pop_size}_g{generations}"

    exp_output = output_dir / experiment_name
    exp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  H3 REPRESENTATION: {fn_name} | D={num_dims} P={pop_size} G={generations}")
    print(f"  Seeds: {len(seeds)} | Output: {exp_output}")
    print(f"{'=' * 70}")

    composer = Composer.create_default()

    # Shared default operators for all precision pipelines
    baseline_selection = f"evosax_mimic_selection:num_selections={pop_size},elite_k={elite_k}"
    baseline_crossover = "evosax_uniform_crossover:crossover_rate=0.3"
    baseline_mutation = "evosax_gaussian:mutation_strength=1.0"

    # ------------------------------------------------------------------
    # Define the 3 pipelines
    # ------------------------------------------------------------------
    pipelines = {
        # 1. Baseline: float32
        "mjx_f32": {
            "backend": "malthusjax",
            "selection": baseline_selection,
            "crossover": baseline_crossover,
            "mutation": baseline_mutation,
            "elitism": 0,
            "dtype": "float32",
        },
        # 2. bfloat16
        "mjx_bf16": {
            "backend": "malthusjax",
            "selection": baseline_selection,
            "crossover": baseline_crossover,
            "mutation": baseline_mutation,
            "elitism": 0,
            "dtype": "bfloat16",
        },
        # 3. float16
        "mjx_f16": {
            "backend": "malthusjax",
            "selection": baseline_selection,
            "crossover": baseline_crossover,
            "mutation": baseline_mutation,
            "elitism": 0,
            "dtype": "float16",
        },
    }

    # ------------------------------------------------------------------
    # Shared config (identical for all pipelines)
    # ------------------------------------------------------------------
    shared = {
        "fitness": f"bbob:fn_name={fn_name},num_dims={num_dims},maximize=false",
        "pop_size": pop_size,
        "generations": generations,
        "genome_length": num_dims,
        "bounds": (-5.0, 5.0),
        "maximize": False,
    }

    # ------------------------------------------------------------------
    # Execute via Composer.compare()
    # ------------------------------------------------------------------
    t0 = time.time()
    comparison = composer.compare(
        pipelines=pipelines,
        seeds=seeds,
        shared_initial_population=True,
        pop_seed=42,
        output_dir=str(exp_output),
        **shared,
    )
    wall_time = time.time() - t0

    # ------------------------------------------------------------------
    # Extract and save results
    # ------------------------------------------------------------------
    result_summary = {
        "experiment": experiment_name,
        "config": {
            "fn_name": fn_name,
            "num_dims": num_dims,
            "pop_size": pop_size,
            "generations": generations,
            "num_seeds": len(seeds),
            "elite_ratio": elite_ratio,
            "elite_k": elite_k,
        },
        "wall_time_seconds": wall_time,
        "pipelines": {},
    }

    for name, exp_result in comparison.pipelines.items():
        pipeline_data = {
            "num_runs": len(exp_result.runs),
            "successful_runs": sum(1 for r in exp_result.runs if r.status == "success"),
            "per_seed": [],
        }

        for run in exp_result.runs:
            # Cast JAX arrays to native Python types to free GPU buffers
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
            if run.history:
                seed_data["convergence"] = [
                    {
                        "generation": int(h.get("generation"))
                        if h.get("generation") is not None
                        else None,
                        "best_fitness": float(h.get("best_fitness"))
                        if h.get("best_fitness") is not None
                        else None,
                    }
                    for h in run.history
                ]
            pipeline_data["per_seed"].append(seed_data)

        result_summary["pipelines"][name] = pipeline_data

    # Clear JAX caches and force garbage collection to prevent OOM across many experiments
    import gc

    import jax

    jax.clear_caches()
    gc.collect()

    # Save JSON
    result_file = exp_output / "representation_results.json"
    with open(result_file, "w") as f:
        json.dump(result_summary, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Print quick summary
    # ------------------------------------------------------------------
    for name, pdata in result_summary["pipelines"].items():
        successes = pdata["successful_runs"]
        total = pdata["num_runs"]
        fitnesses = [s["best_fitness"] for s in pdata["per_seed"] if s["best_fitness"] is not None]
        if fitnesses:
            import statistics

            mean_fit = statistics.mean(fitnesses)
            std_fit = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0
            print(f"  {name:30s} | {successes}/{total} ok | fitness={mean_fit:.6f} ± {std_fit:.6f}")
        else:
            print(f"  {name:30s} | {successes}/{total} ok | NO FITNESS DATA")

    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Results saved: {result_file}")

    return result_summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_representation_suite(args: argparse.Namespace) -> None:
    """Run the full H3 representation suite across the parameter grid."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.num_seeds + 1))

    all_results = []
    total_experiments = len(args.functions) * len(args.dims) * len(args.pops) * len(args.gens)
    current = 0

    print(f"\n{'#' * 70}")
    print("  H3 REPRESENTATION SUITE")
    print(f"  Functions: {args.functions}")
    print(f"  Dims: {args.dims}")
    print(f"  Pops: {args.pops}")
    print(f"  Gens: {args.gens}")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Output: {output_dir}")
    print(f"{'#' * 70}")

    suite_start = time.time()

    for fn_name in args.functions:
        for D in args.dims:
            for P in args.pops:
                for G in args.gens:
                    current += 1
                    print(f"\n>>> Experiment {current}/{total_experiments}")

                    try:
                        result = run_single_representation(
                            fn_name=fn_name,
                            num_dims=D,
                            pop_size=P,
                            generations=G,
                            seeds=seeds,
                            output_dir=output_dir,
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback

                        traceback.print_exc()
                        all_results.append(
                            {
                                "experiment": f"h3_{fn_name}_d{D}_p{P}_g{G}",
                                "error": str(e),
                            }
                        )

    suite_time = time.time() - suite_start

    # Save suite summary
    suite_summary = {
        "suite": "h3_representation",
        "mode": "smoke" if args.smoke else "full",
        "total_experiments": total_experiments,
        "successful": sum(1 for r in all_results if "error" not in r),
        "failed": sum(1 for r in all_results if "error" in r),
        "total_wall_time_seconds": suite_time,
        "experiments": all_results,
    }

    suite_file = output_dir / "suite_summary.json"
    with open(suite_file, "w") as f:
        json.dump(suite_summary, f, indent=2, default=str)

    print(f"\n{'#' * 70}")
    print("  SUITE COMPLETE")
    print(f"  Successful: {suite_summary['successful']}/{total_experiments}")
    print(f"  Failed: {suite_summary['failed']}/{total_experiments}")
    print(f"  Total wall time: {suite_time:.1f}s")
    print(f"  Suite summary: {suite_file}")
    print(f"{'#' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> list[str]:
    """Parse comma-separated strings."""
    return [x.strip() for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H3 Representation: Numerical Precision Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal smoke test (1 function, small grid, few seeds)",
    )
    parser.add_argument(
        "--functions",
        type=parse_str_list,
        default=None,
        help="Comma-separated BBOB functions (default: smoke=sphere, full=sphere,rosenbrock,rastrigin)",
    )
    parser.add_argument(
        "--dims",
        type=parse_int_list,
        default=None,
        help="Comma-separated dimensionalities (default: smoke=5, full=10,50,100)",
    )
    parser.add_argument(
        "--pops",
        type=parse_int_list,
        default=None,
        help="Comma-separated population sizes (default: smoke=32, full=64,256,1024)",
    )
    parser.add_argument(
        "--gens",
        type=parse_int_list,
        default=None,
        help="Comma-separated generation counts (default: smoke=10, full=50,200)",
    )
    parser.add_argument(
        "--seeds",
        "--num-seeds",
        type=int,
        default=None,
        dest="num_seeds",
        help="Number of independent seeds (default: smoke=3, full=100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: smoke=results/h3_representation_smoke, full=results/h3_representation)",
    )

    args = parser.parse_args()

    # Apply defaults based on mode
    defaults = SMOKE_DEFAULTS if args.smoke else FULL_DEFAULTS
    if args.functions is None:
        args.functions = defaults["functions"]
    if args.dims is None:
        args.dims = defaults["dims"]
    if args.pops is None:
        args.pops = defaults["pops"]
    if args.gens is None:
        args.gens = defaults["gens"]
    if args.num_seeds is None:
        args.num_seeds = defaults["num_seeds"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]

    run_representation_suite(args)


if __name__ == "__main__":
    main()
