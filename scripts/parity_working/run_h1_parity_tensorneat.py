#!/usr/bin/env python3
"""H1 TensorNEAT Parity: MalthusJAX Native MAP-Elites vs TensorNEAT Pure.

This script runs the parity experiment for TensorNEAT integration.
It runs two pipelines on the XOR problem:
  1. tensorneat_baseline — Pure TensorNEAT loop inside JAX.
  2. malthusjax_native   — MalthusJAX engine natively mapping MapElites onto TensorNEAT graphs.

Usage:
    # Smoke test (local, ~30 seconds)
    python scripts/parity_working/run_h1_parity_tensorneat.py --smoke

    # Full run (cluster)
    python scripts/parity_working/run_h1_parity_tensorneat.py
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

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from tensorneat.problem import XOR
from tensorneat.genome import DefaultGenome
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import State as TNState

from malthusjax.core.genome.tensorneat_genome import TensorNeatPopulation, TensorNeatGenome
from malthusjax.core.fitness.qd import TensorNeatQDEvaluator, TensorNeatEvaluatorConfig
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter
from malthusjax.engine.qd.map_elites import MapElitesState

try:
    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
except ImportError:
    MapElitesRepertoire = Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_DEFAULTS = {
    "functions": ["xor"],
    "pops": [128],
    "gens": [20],
    "num_seeds": 3,
    "output_dir": "results/h1_parity_tensorneat_smoke",
}

FULL_DEFAULTS = {
    "functions": ["xor"],
    "pops": [512, 1024],
    "gens": [100, 500],
    "num_seeds": 10,
    "output_dir": "results/h1_parity_tensorneat",
}

def compute_ci(data: list[float], confidence: float = 0.95) -> dict[str, float]:
    """Compute mean and confidence interval for a list of floats."""
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    if n <= 1:
        return {"mean": float(m), "ci_margin": 0.0, "ci_lower": float(m), "ci_upper": float(m)}
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return {
        "mean": float(m),
        "ci_margin": float(h),
        "ci_lower": float(m - h),
        "ci_upper": float(m + h),
    }

def run_single_parity(
    fn_name: str,
    pop_size: int,
    generations: int,
    seeds: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Run one H1 parity comparison: Pure TensorNEAT vs MalthusJAX Native."""

    experiment_name = f"h1_tensorneat_{fn_name}_p{pop_size}_g{generations}"
    exp_output = output_dir / experiment_name
    exp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  H1 PARITY TENSORNEAT: {fn_name} | P={pop_size} G={generations}")
    print(f"  Seeds: {len(seeds)} | Output: {exp_output}")
    print(f"{'=' * 70}")

    # Shared TensorNEAT operators
    problem = XOR()
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)
    algorithm = NEAT(pop_size=pop_size, species_size=2, genome=genome)
    
    # -----------------------------------------------------------------------
    # Pipeline 1: Pure TensorNEAT (Adapter-like logic)
    # -----------------------------------------------------------------------
    @jax.jit
    def pure_step(state: TNState) -> tuple[TNState, float]:
        pop = algorithm.ask(state)
        nodes, conns = pop
        transformed = jax.vmap(lambda s, n, c: algorithm.transform(s, (n, c)), in_axes=(None, 0, 0))(state, nodes, conns)
        keys = jax.random.split(state.randkey, pop_size)
        fitness = jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(state, keys, algorithm.forward, transformed)
        
        # Replace NaNs
        fitness = jnp.where(jnp.isnan(fitness), -jnp.inf, fitness)
        max_fit = jnp.max(fitness)
        
        state = algorithm.tell(state, fitness)
        return state, max_fit

    pure_best_fitness = []
    pure_exec_times = []
    pure_total_times = []

    print("Running Pipeline 1: Pure TensorNEAT...")
    for seed in tqdm.tqdm(seeds, desc="Seeds"):
        start_t = time.time()
        
        # Init
        key = jax.random.PRNGKey(seed)
        state = TNState(randkey=key)
        state = algorithm.setup(state)
        
        compile_t = time.time()
        
        # JIT warmup on step 0
        state, bfit = pure_step(state)
        bfit.block_until_ready()
        
        exec_start = time.time()
        
        max_fitness = float(bfit)
        
        # Run generations
        for _ in range(generations - 1):
            state, bfit = pure_step(state)
            max_fitness = max(max_fitness, float(bfit))
            
        exec_end = time.time()
        
        pure_best_fitness.append(max_fitness)
        pure_exec_times.append(exec_end - exec_start)
        pure_total_times.append(exec_end - start_t)

    # -----------------------------------------------------------------------
    # Pipeline 2: MalthusJAX Native Mapping
    # -----------------------------------------------------------------------
    evaluator = TensorNeatQDEvaluator.create(
        algorithm=algorithm, 
        problem=problem, 
        forward_fn=algorithm.forward,
        config=TensorNeatEvaluatorConfig(seed=42)
    )
    
    emitter = TensorNeatEmitter(_batch_size=pop_size, genome=genome)
    
    @jax.jit
    def native_init_step(k):
        # Generate initial TN population
        tn_state = TNState(randkey=k)
        tn_state = algorithm.setup(tn_state)
        pop_values = algorithm.ask(tn_state)
        
        initial_genes = TensorNeatGenome(values=pop_values)
        initial_pop = TensorNeatPopulation(
            genes=initial_genes,
            fitness=jnp.full(pop_size, -jnp.inf),
            config=None,
            info={}
        )
        
        evaluated_pop = evaluator.evaluate_population(initial_pop)
        
        # Inject dummy descriptors
        new_info = dict(evaluated_pop.info)
        new_info["descriptors"] = jnp.full((pop_size, 1), 0.5)
        evaluated_pop = evaluated_pop.replace(info=new_info)
        
        repertoire = MapElitesRepertoire.init(
            genotypes=getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
            fitnesses=evaluated_pop.fitness,
            descriptors=evaluated_pop.info["descriptors"],
            centroids=jnp.array([[0.5]])
        )
        emitter_state = emitter.init(k, evaluated_pop, params=None)
        
        best_genome_idx = jnp.argmax(evaluated_pop.fitness)
        best_genome_values = jax.tree_util.tree_map(lambda x: x[best_genome_idx], getattr(evaluated_pop.genes, "values", evaluated_pop.genes))
        if hasattr(evaluated_pop.genes, "replace"):
            best_genome = evaluated_pop.genes.replace(values=best_genome_values)
        else:
            best_genome = best_genome_values
            
        return MapElitesState(
            population=evaluated_pop,
            best_genome=best_genome,
            generation=0,
            best_fitness=jnp.max(evaluated_pop.fitness),
            repertoire=repertoire,
            emitter_state=emitter_state,
            rng_key=k
        )
        
    @jax.jit
    def native_train_step(state):
        k, key_ask_seed, key_eval = jax.random.split(state.rng_key, 3)
        key_ask = jax.random.split(key_ask_seed, emitter.num_keys())
        
        offspring_pop, new_emitter_state = emitter.ask(
            state.emitter_state, state.repertoire, key_ask
        )
        
        evaluated_pop = evaluator.evaluate_population(offspring_pop)
        new_info = dict(evaluated_pop.info)
        new_info["descriptors"] = jnp.full((pop_size, 1), 0.5)
        evaluated_pop = evaluated_pop.replace(info=new_info)
        
        new_repertoire = state.repertoire.add(
            getattr(evaluated_pop.genes, "values", evaluated_pop.genes),
            evaluated_pop.info["descriptors"],
            evaluated_pop.fitness
        )
        
        best_genome_idx = jnp.argmax(evaluated_pop.fitness)
        best_genome_values = jax.tree_util.tree_map(lambda x: x[best_genome_idx], getattr(evaluated_pop.genes, "values", evaluated_pop.genes))
        if hasattr(evaluated_pop.genes, "replace"):
            best_genome = evaluated_pop.genes.replace(values=best_genome_values)
        else:
            best_genome = best_genome_values
            
        return state.replace(
            population=evaluated_pop,
            best_genome=best_genome,
            generation=state.generation + 1,
            best_fitness=jnp.maximum(state.best_fitness, jnp.max(evaluated_pop.fitness)),
            repertoire=new_repertoire,
            emitter_state=new_emitter_state,
            rng_key=k
        )

    native_best_fitness = []
    native_exec_times = []
    native_total_times = []

    print("Running Pipeline 2: MalthusJAX Native...")
    for seed in tqdm.tqdm(seeds, desc="Seeds"):
        start_t = time.time()
        
        key = jax.random.PRNGKey(seed)
        
        state = native_init_step(key)
        state.best_fitness.block_until_ready()
        
        exec_start = time.time()
        
        for _ in range(generations - 1):
            state = native_train_step(state)
            
        state.best_fitness.block_until_ready()
        exec_end = time.time()
        
        native_best_fitness.append(float(state.best_fitness))
        native_exec_times.append(exec_end - exec_start)
        native_total_times.append(exec_end - start_t)

    # -----------------------------------------------------------------------
    # Results formatting
    # -----------------------------------------------------------------------
    results = {
        "tensorneat_baseline": {
            "best_fitness": compute_ci(pure_best_fitness),
            "time_execution": compute_ci(pure_exec_times),
            "time_total": compute_ci(pure_total_times),
        },
        "malthusjax_native": {
            "best_fitness": compute_ci(native_best_fitness),
            "time_execution": compute_ci(native_exec_times),
            "time_total": compute_ci(native_total_times),
        },
    }

    out_file = exp_output / "parity_results_tensorneat.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=2))
    print(f"  Results saved: {out_file}\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="H1 Parity TensorNEAT Runner")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    args = parser.parse_args()

    cfg = SMOKE_DEFAULTS if args.smoke else FULL_DEFAULTS

    if args.smoke:
        print("Starting H1 TensorNEAT Parity Suite (SMOKE)")
    else:
        print("Starting H1 TensorNEAT Parity Suite (FULL)")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    seeds = list(range(1, cfg["num_seeds"] + 1))

    for fn_name in cfg["functions"]:
        for pop_size in cfg["pops"]:
            for generations in cfg["gens"]:
                run_single_parity(
                    fn_name=fn_name,
                    pop_size=pop_size,
                    generations=generations,
                    seeds=seeds,
                    output_dir=out_dir,
                )

if __name__ == "__main__":
    main()
