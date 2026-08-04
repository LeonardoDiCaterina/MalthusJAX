#!/usr/bin/env python3
"""H1 Parity: MalthusJAX Native MAP-Elites vs QDAX MAP-Elites.

This script runs the ground-truth parity experiment for Quality-Diversity.
It runs two pipelines:
  1. qdax_baseline     — QDAX MAP-Elites natively through its standard API.
  2. malthusjax_native — MalthusJAX engine natively running the MapElitesStrategy.

Both pipelines share identical seeds. Since they are Quality-Diversity algorithms,
we compare their QD metrics (e.g. qd_score, max_fitness) over generations.

Usage:
    # Smoke test (local, ~30 seconds)
    python scripts/parity_working/run_h1_parity_qdax.py --smoke

    # Full run (cluster)
    python scripts/parity_working/run_h1_parity_qdax.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# Prevent JAX NCCL multi-device rendezvous deadlocks on cluster by restricting to 1 GPU
if "CUDA_VISIBLE_DEVICES" in os.environ:
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from malthusjax.composer import Composer
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.composer.catalog import OperatorCatalog

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_DEFAULTS = {
    "functions": ["rosenbrock"],
    "dims": [10],
    "pops": [100],
    "gens": [50],
    "num_seeds": 3,
    "output_dir": "results/h1_parity_qdax_smoke",
}

FULL_DEFAULTS = {
    "functions": ["rosenbrock", "rastrigin"],
    "dims": [10, 50, 100],
    "pops": [100, 500],
    "gens": [500, 1000],
    "num_seeds": 30,
    "output_dir": "results/h1_parity_qdax",
}

def run_single_parity(
    fn_name: str,
    num_dims: int,
    pop_size: int,
    generations: int,
    seeds: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Run one H1 parity comparison: QDAX vs MalthusJAX Native."""

    experiment_name = f"h1_qdax_{fn_name}_d{num_dims}_p{pop_size}_g{generations}"
    exp_output = output_dir / experiment_name
    exp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  H1 PARITY QDAX: {fn_name} | D={num_dims} P={pop_size} G={generations}")
    print(f"  Seeds: {len(seeds)} | Output: {exp_output}")
    print(f"{'=' * 70}")

    composer = Composer.create_default()
    cat = OperatorCatalog()
    
    # We use a pure gaussian mutation emitter for native to match QDAX default MapElites
    mutation = cat.get("gaussian:mutation_rate=1.0,mutation_strength=0.1")
    native_emitter = GeneticMutationEmitter(
        _batch_size=pop_size,
        mutation=mutation,
        genome_config=RealGenomeConfig(shape=(num_dims,), bounds=(-5.0, 5.0))
    )
    
    # Exact replica of QDAX's default MixingEmitter configuration
    import jax
    replica_emitter = QDAXReplicaMixingEmitter(
        mutation_fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1,
        variation_fn=lambda x1, x2, key: x1,  # NO-OP crossover
        variation_percentage=0.0,
        _batch_size=pop_size,
        genome_config=RealGenomeConfig(shape=(num_dims,), bounds=(-5.0, 5.0)),
    )

    pipelines = {
        "qdax_baseline": dict(
            backend="qdax",
            qdax_strategy="MAPElites",
            qdax_num_centroids=pop_size,
            qdax_mutation_sigma=0.1,
            qdax_variation_percentage=0.0
        ),
        "malthusjax_native": dict(
            strategy=MapElitesStrategy(
                emitter=native_emitter,
                num_centroids=pop_size
            ),
            track_metrics=False
        ),
        "malthusjax_replica": dict(
            strategy=MapElitesStrategy(
                emitter=replica_emitter,
                num_centroids=pop_size,
                key_derivation="qdax_replica"
            ),
            track_metrics=False
        )
    }

    shared = {
        "fitness": f"{fn_name}:dim={num_dims}",
        "pop_size": pop_size,
        "generations": generations,
        "genome_length": num_dims,
        "bounds": (-5.0, 5.0),
        "maximize": False, # QDAX parity relies on minimizing BBOB benchmarks
    }

    t0 = time.time()
    comparison = composer.compare(
        pipelines=pipelines,
        seeds=seeds,
        shared_initial_population=True,
        output_dir=str(exp_output),
        **shared,
    )
    wall_time = time.time() - t0

    result_summary = {
        "experiment": experiment_name,
        "config": {
            "fn_name": fn_name,
            "num_dims": num_dims,
            "pop_size": pop_size,
            "generations": generations,
            "num_seeds": len(seeds),
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
            best_fit = run.metrics.get("best_fitness")
            final_gen = run.metrics.get("final_generation")
            qd_score = run.metrics.get("qd_score")
            coverage = run.metrics.get("coverage")

            seed_data = {
                "seed": run.seed,
                "status": run.status,
                "duration_seconds": run.duration_seconds,
                "best_fitness": float(best_fit) if best_fit is not None else None,
                "qd_score": float(qd_score) if qd_score is not None else None,
                "coverage": float(coverage) if coverage is not None else None,
            }
            if run.timings:
                seed_data["timings"] = run.timings
            if run.history:
                seed_data["convergence"] = [
                    {
                        "generation": int(h.get("generation")) if h.get("generation") is not None else None,
                        "best_fitness": float(h.get("best_fitness")) if h.get("best_fitness") is not None else None,
                        "qd_score": float(h.get("qd_score")) if h.get("qd_score") is not None else None,
                        "coverage": float(h.get("coverage")) if h.get("coverage") is not None else None,
                    }
                    for h in run.history
                ]
            pipeline_data["per_seed"].append(seed_data)

        result_summary["pipelines"][name] = pipeline_data

    import gc
    import jax
    jax.clear_caches()
    gc.collect()

    result_file = exp_output / "parity_results_qdax.json"
    with open(result_file, "w") as f:
        json.dump(result_summary, f, indent=2, default=str)

    print(comparison.summary_table())
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Results saved: {result_file}")

    return result_summary


def main():
    parser = argparse.ArgumentParser(description="H1 Parity: MalthusJAX Native vs QDAX")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--functions", nargs="+", type=str)
    parser.add_argument("--dims", nargs="+", type=int)
    parser.add_argument("--pops", nargs="+", type=int)
    parser.add_argument("--gens", nargs="+", type=int)
    parser.add_argument("--seeds", type=int)
    args = parser.parse_args()

    defaults = SMOKE_DEFAULTS if args.smoke else FULL_DEFAULTS

    functions = args.functions or defaults["functions"]
    dims = args.dims or defaults["dims"]
    pops = args.pops or defaults["pops"]
    gens = args.gens or defaults["gens"]
    num_seeds = args.seeds or defaults["num_seeds"]
    output_base = Path(defaults["output_dir"])

    print(f"Starting H1 QDAX Parity Suite ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Output directory: {output_base}")
    
    seeds = list(range(1, num_seeds + 1))
    
    for fn in functions:
        for d in dims:
            for p in pops:
                for g in gens:
                    run_single_parity(
                        fn_name=fn,
                        num_dims=d,
                        pop_size=p,
                        generations=g,
                        seeds=seeds,
                        output_dir=output_base
                    )

if __name__ == "__main__":
    main()
