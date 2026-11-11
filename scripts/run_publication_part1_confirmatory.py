#!/usr/bin/env python3
"""Part 1 Confirmatory Runner: EvoSAX and QDax Parity.

Implements the three-leg comparison (A: direct original, B: adapter, C: native mimic)
across BBOB benchmark functions with synchronized quality and timing evaluation.

CLI Contract:
    python scripts/run_publication_part1_confirmatory.py --config PATH --stage preflight|smoke|pilot|full --output NEW_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def digest(array: Any) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def installed_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_evosax_direct(
    env: Any,
    initial_pop: Any,
    pop_size: int,
    dim: int,
    generations: int,
    rng_key: Any,
    crossover_rate: float = 0.3,
    elite_ratio: float = 0.16666666,
) -> Dict[str, Any]:
    """Execute direct upstream EvoSAX SimpleGA."""
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import SimpleGA

    strategy = SimpleGA(population_size=pop_size, solution=jnp.zeros(dim))
    strategy.elite_ratio = elite_ratio
    params = strategy.default_params.replace(crossover_rate=crossover_rate)

    k_init, k_run = jax.random.split(rng_key)

    def eval_pop(pop_arr: Any) -> Any:
        return jax.vmap(env.evaluate)(pop_arr)

    f_init = eval_pop(initial_pop)
    state = strategy.init(k_init, initial_pop, f_init, params)

    best_history = [float(state.best_fitness)]

    for g in range(generations):
        k_g = jax.random.fold_in(k_run, g)
        k_ask, k_tell = jax.random.split(k_g)
        x, state = strategy.ask(k_ask, state, params)
        f = eval_pop(x)
        state, _ = strategy.tell(k_tell, x, f, state, params)
        best_history.append(float(state.best_fitness))

    return {
        "status": "success",
        "best_fitness": float(state.best_fitness),
        "history": best_history,
        "total_evaluations": pop_size * (generations + 1),
    }


def run_qdax_direct(
    env: Any,
    initial_pop: Any,
    centroids: Any,
    pop_size: int,
    dim: int,
    generations: int,
    rng_key: Any,
    mutation_sigma: float = 0.1,
    variation_pct: float = 0.5,
) -> Dict[str, Any]:
    """Execute direct upstream QDax MAP-Elites."""
    import jax
    import jax.numpy as jnp
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites

    def scoring_fn(x: Any, k: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        # QDax maximizes fitness internally
        raw_fit = jax.vmap(env.evaluate)(x)
        fit = -raw_fit
        desc = jnp.clip((x[:, :2] + 5.0) / 10.0, 0.0, 1.0)
        return fit, desc, {}

    def mutation(x: Any, rng: Any) -> Any:
        return x + mutation_sigma * jax.random.normal(rng, x.shape)

    def variation(x: Any, y: Any, rng: Any) -> Any:
        return jnp.where(jax.random.bernoulli(rng, variation_pct, x.shape), x, y)

    emitter = MixingEmitter(
        mutation_fn=mutation,
        variation_fn=variation,
        variation_percentage=variation_pct,
        batch_size=pop_size,
    )
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=lambda rep: {},
    )

    k_init, k_run = jax.random.split(rng_key)
    repertoire, emitter_state, _ = map_elites.init(initial_pop, centroids, k_init)

    best_raw_history = [-float(jnp.max(repertoire.fitnesses))]

    for g in range(generations):
        k_g = jax.random.fold_in(k_run, g)
        repertoire, emitter_state, _ = map_elites.update(repertoire, emitter_state, k_g)
        best_raw_history.append(-float(jnp.max(repertoire.fitnesses)))

    valid_mask = repertoire.fitnesses > -jnp.inf
    coverage = float(100.0 * jnp.sum(valid_mask) / centroids.shape[0])
    qd_score = float(jnp.sum(jnp.where(valid_mask, repertoire.fitnesses, 0.0)))
    best_fitness = -float(jnp.max(repertoire.fitnesses))  # raw minimization objective

    return {
        "status": "success",
        "best_fitness": best_fitness,
        "coverage": coverage,
        "qd_score": qd_score,
        "history": best_raw_history,
        "total_evaluations": pop_size * (generations + 1),
    }


def measure_steady_timing(
    fn: Any,
    args: Tuple[Any, ...],
    repetitions: int = 5,
) -> Dict[str, Any]:
    """Capture first prepared call, warmup, and repeated steady executions."""
    import jax
    import jax.numpy as jnp

    # Reset buffers before each trial
    def clone_args(a: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return jax.tree_util.tree_map(
            lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else x, a
        )

    # 1. First prepared execution (lowering / compile + first call)
    t_args = clone_args(args)
    jax.block_until_ready(t_args)
    t0 = time.perf_counter()
    out = fn(*t_args)
    jax.block_until_ready(out)
    first_call_sec = time.perf_counter() - t0

    # 2. Warmup call
    t_args = clone_args(args)
    jax.block_until_ready(t_args)
    t0 = time.perf_counter()
    out = fn(*t_args)
    jax.block_until_ready(out)
    warmup_sec = time.perf_counter() - t0

    # 3. Steady repeats
    repeat_times = []
    for _ in range(repetitions):
        t_args = clone_args(args)
        jax.block_until_ready(t_args)
        t0 = time.perf_counter()
        out = fn(*t_args)
        jax.block_until_ready(out)
        repeat_times.append(time.perf_counter() - t0)

    return {
        "first_call_seconds": first_call_sec,
        "warmup_seconds": warmup_sec,
        "steady_repeat_seconds": repeat_times,
        "steady_median_seconds": float(np.median(repeat_times)),
        "steady_mean_seconds": float(np.mean(repeat_times)),
        "steady_std_seconds": float(np.std(repeat_times)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/publication/part1_confirmatory.json"))
    parser.add_argument("--stage", choices=["preflight", "smoke", "pilot", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", choices=["evosax", "qdax", "both"], default="both")
    parser.add_argument("--timing", action="store_true", default=True)
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.output.mkdir(parents=True, exist_ok=True)

    # Stage 1: PREFLIGHT
    if args.stage == "preflight":
        full_grid = cfg["full_grid"]
        pilot_grid = cfg["pilot_grid"]

        num_fns_full = len(full_grid["functions"])
        num_inst_full = len(full_grid["instances"])
        num_dims_full = len(full_grid["dims"])
        num_pops_full = len(full_grid["populations"])
        num_gens_full = len(full_grid["generations"])
        total_cells_full = num_fns_full * num_inst_full * num_dims_full * num_pops_full * num_gens_full

        num_seeds_full = full_grid["candidate_n"]
        conditions_per_lib = 3
        libs = 2 if args.library == "both" else 1
        total_runs_full = total_cells_full * num_seeds_full * (conditions_per_lib * libs)

        # Pilot grid calculation
        num_fns_pilot = len(pilot_grid["functions"])
        num_inst_pilot = len(pilot_grid["instances"])
        num_dims_pilot = len(pilot_grid["dims"])
        num_pops_pilot = len(pilot_grid["populations"])
        num_gens_pilot = len(pilot_grid["generations"])
        total_cells_pilot = num_fns_pilot * num_inst_pilot * num_dims_pilot * num_pops_pilot * num_gens_pilot
        num_seeds_pilot = len(pilot_grid["seeds"])
        total_runs_pilot = total_cells_pilot * num_seeds_pilot * (conditions_per_lib * libs)

        preflight_summary = {
            "schema_version": 1,
            "stage": "preflight",
            "status": "pass",
            "full_grid": {
                "total_cells": total_cells_full,
                "candidate_n": num_seeds_full,
                "total_runs": total_runs_full,
                "factors": {
                    "functions": full_grid["functions"],
                    "instances": full_grid["instances"],
                    "dims": full_grid["dims"],
                    "populations": full_grid["populations"],
                    "generations": full_grid["generations"],
                },
            },
            "pilot_grid": {
                "total_cells": total_cells_pilot,
                "pilot_seeds": pilot_grid["seeds"],
                "total_runs": total_runs_pilot,
                "factors": {
                    "functions": pilot_grid["functions"],
                    "instances": pilot_grid["instances"],
                    "dims": pilot_grid["dims"],
                    "populations": pilot_grid["populations"],
                    "generations": pilot_grid["generations"],
                },
            },
            "candidate_equivalence_margin": cfg["primary_metric"]["candidate_equivalence_margin"],
            "primary_metric": cfg["primary_metric"],
            "devices": [str(d) for d in jax.devices()] if "jax" in sys.modules else "pending_import",
        }

        out_path = args.output / "resolution.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preflight_summary, f, indent=2)
        print(f"Preflight resolution complete. Written to {out_path}")
        print(f"  • Full Grid: {total_cells_full} cells, {total_runs_full} condition runs ({num_seeds_full} seeds)")
        print(f"  • Pilot Grid: {total_cells_pilot} cells, {total_runs_pilot} condition runs ({num_seeds_pilot} seeds)")
        return 0

    # For SMOKE, PILOT, and FULL: import JAX and framework
    if args.stage == "smoke":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    from malthusjax.composer import Composer
    from malthusjax.composer.strategies.core import MapElitesStrategy, QDAXStrategy
    from malthusjax.core.fitness.composable.environments import BBOBEnv
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter
    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
    from qdax.core.emitters.standard_emitters import MixingEmitter

    active_grid = cfg["smoke_grid"] if args.stage == "smoke" else cfg["pilot_grid"]
    if args.stage == "full":
        active_grid = cfg["full_grid"]
        active_grid["seeds"] = list(range(active_grid["seeds_range"][0], active_grid["seeds_range"][1] + 1))
        active_grid["timing_repetitions"] = cfg["timing_protocol"]["steady_repetitions"]

    functions = active_grid["functions"]
    instances = active_grid.get("instances", [1])
    dims = active_grid["dims"]
    populations = active_grid["populations"]
    generations_list = active_grid["generations"]
    seeds = active_grid["seeds"]
    repetitions = active_grid.get("timing_repetitions", 3)
    epsilon = cfg["primary_metric"]["epsilon"]

    composer = Composer.create_default()

    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "library": args.library,
        "python": sys.version,
        "platform": platform.platform(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "devices": [str(d) for d in jax.devices()],
        "versions": {
            name: installed_version(name) for name in ["jax", "jaxlib", "evosax", "qdax", "numpy"]
        },
        "grid": active_grid,
        "primary_metric": cfg["primary_metric"],
    }
    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    blocks: List[Dict[str, Any]] = []
    failures = 0
    records: List[Dict[str, Any]] = []

    print(f"\n===========================================================================")
    print(f"  MALTHUSJAX PART 1 CONFIRMATORY (Stage: {args.stage.upper()} | Backend: {jax.default_backend()})")
    print(f"  Destination: {args.output}")
    print(f"===========================================================================\n")

    for fn_name in functions:
        for inst_seed in instances:
            for dim in dims:
                for pop in populations:
                    for gens in generations_list:
                        # Instantiate BBOB environment for this cell
                        env = BBOBEnv.create(fn_name=fn_name, num_dims=dim, seed=inst_seed)
                        f_opt = float(env.f_opt)

                        for seed in seeds:
                            cell_id = f"{fn_name}_i{inst_seed}_d{dim}_p{pop}_g{gens}_s{seed}"
                            block_dir = args.output / cell_id
                            block_dir.mkdir(parents=True, exist_ok=True)

                            key = jax.random.PRNGKey(seed)
                            k_init = jax.random.fold_in(key, 101)
                            k_cent = jax.random.fold_in(key, 102)

                            initial_pop = jax.random.uniform(k_init, (pop, dim), minval=-5.0, maxval=5.0)
                            num_centroids = 16 if args.stage == "smoke" else 100
                            centroids = compute_cvt_centroids(2, 5000, num_centroids, 0.0, 1.0, k_cent)

                            np.save(block_dir / "initial_population.npy", np.asarray(initial_pop))
                            np.save(block_dir / "centroids.npy", np.asarray(centroids))

                            block_data: Dict[str, Any] = {
                                "cell_id": cell_id,
                                "function": fn_name,
                                "instance": inst_seed,
                                "dim": dim,
                                "pop_size": pop,
                                "generations": gens,
                                "seed": seed,
                                "f_opt": f_opt,
                                "initial_pop_hash": digest(initial_pop),
                                "centroids_hash": digest(centroids),
                                "runs": {},
                            }

                            # ---------------------------------------------------------
                            # EVOSAX TRIAD
                            # ---------------------------------------------------------
                            if args.library in ("both", "evosax"):
                                # A. Direct
                                t0 = time.perf_counter()
                                try:
                                    res_dir = run_evosax_direct(
                                        env=env,
                                        initial_pop=initial_pop,
                                        pop_size=pop,
                                        dim=dim,
                                        generations=gens,
                                        rng_key=key,
                                        crossover_rate=0.3,
                                        elite_ratio=0.16666666,
                                    )
                                    dur_dir = time.perf_counter() - t0
                                    delta_f = max(res_dir["best_fitness"] - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    res_dir.update({
                                        "collection_seconds": dur_dir,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                    })
                                    block_data["runs"]["evosax_direct"] = res_dir
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["evosax_direct"] = {"status": "error", "error": str(e)}

                                # B. Adapter
                                t0 = time.perf_counter()
                                try:
                                    res_ad = composer.quick_run(
                                        backend="evosax",
                                        evosax_strategy="SimpleGA",
                                        strategy_params={"crossover_rate": 0.3, "elite_ratio": 0.16666666},
                                        fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
                                        pop_size=pop,
                                        generations=gens,
                                        genome_length=dim,
                                        bounds=(-5.0, 5.0),
                                        maximize=False,
                                        initial_population=initial_pop,
                                        seeds=(seed,),
                                    )
                                    dur_ad = time.perf_counter() - t0
                                    run_obj = res_ad.runs[0]
                                    bf = float(run_obj.metrics.get("best_fitness", np.nan))
                                    delta_f = max(bf - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    block_data["runs"]["evosax_adapter"] = {
                                        "status": "success",
                                        "best_fitness": bf,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                        "collection_seconds": dur_ad,
                                    }
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["evosax_adapter"] = {"status": "error", "error": str(e)}

                                # C. Native Mimic
                                t0 = time.perf_counter()
                                try:
                                    res_nat = composer.quick_run(
                                        backend="malthusjax",
                                        selection=f"evosax_mimic_selection:num_selections={pop},elite_k={max(2, int(pop / 6))}",
                                        crossover="evosax_uniform_crossover:crossover_rate=0.3",
                                        mutation="evosax_gaussian:mutation_strength=1.0",
                                        elitism=0,
                                        fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
                                        pop_size=pop,
                                        generations=gens,
                                        genome_length=dim,
                                        bounds=(-5.0, 5.0),
                                        maximize=False,
                                        initial_population=initial_pop,
                                        seeds=(seed,),
                                    )
                                    dur_nat = time.perf_counter() - t0
                                    run_obj = res_nat.runs[0]
                                    bf = float(run_obj.metrics.get("best_fitness", np.nan))
                                    delta_f = max(bf - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    block_data["runs"]["evosax_native"] = {
                                        "status": "success",
                                        "best_fitness": bf,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                        "collection_seconds": dur_nat,
                                    }
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["evosax_native"] = {"status": "error", "error": str(e)}

                            # ---------------------------------------------------------
                            # QDAX TRIAD
                            # ---------------------------------------------------------
                            if args.library in ("both", "qdax"):
                                def q_mut(x: Any, rng: Any) -> Any:
                                    return x + 0.1 * jax.random.normal(rng, x.shape)

                                def q_var(x: Any, y: Any, rng: Any) -> Any:
                                    return jnp.where(jax.random.bernoulli(rng, 0.5, x.shape), x, y)

                                upstream_emitter = MixingEmitter(
                                    mutation_fn=q_mut,
                                    variation_fn=q_var,
                                    variation_percentage=0.5,
                                    batch_size=pop,
                                )

                                # A. Direct
                                t0 = time.perf_counter()
                                try:
                                    res_qdir = run_qdax_direct(
                                        env=env,
                                        initial_pop=initial_pop,
                                        centroids=centroids,
                                        pop_size=pop,
                                        dim=dim,
                                        generations=gens,
                                        rng_key=key,
                                    )
                                    dur_qdir = time.perf_counter() - t0
                                    delta_f = max(res_qdir["best_fitness"] - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    res_qdir.update({
                                        "collection_seconds": dur_qdir,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                    })
                                    block_data["runs"]["qdax_direct"] = res_qdir
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["qdax_direct"] = {"status": "error", "error": str(e)}

                                # B. Adapter
                                t0 = time.perf_counter()
                                try:
                                    res_qad = composer.quick_run(
                                        backend="qdax",
                                        strategy=QDAXStrategy(
                                            emitter=upstream_emitter,
                                            centroids=centroids,
                                            num_centroids=num_centroids,
                                        ),
                                        fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
                                        pop_size=pop,
                                        generations=gens,
                                        genome_length=dim,
                                        bounds=(-5.0, 5.0),
                                        maximize=False,
                                        initial_population=initial_pop,
                                        seeds=(seed,),
                                    )
                                    dur_qad = time.perf_counter() - t0
                                    run_obj = res_qad.runs[0]
                                    bf = float(run_obj.metrics.get("best_fitness", np.nan))
                                    delta_f = max(bf - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    block_data["runs"]["qdax_adapter"] = {
                                        "status": "success",
                                        "best_fitness": bf,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                        "coverage": float(run_obj.metrics.get("coverage", 0.0)),
                                        "qd_score": float(run_obj.metrics.get("qd_score", 0.0)),
                                        "collection_seconds": dur_qad,
                                    }
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["qdax_adapter"] = {"status": "error", "error": str(e)}

                                # C. Native Mimic
                                native_emitter = QDAXReplicaMixingEmitter(
                                    mutation_fn=q_mut,
                                    variation_fn=q_var,
                                    variation_percentage=0.5,
                                    _batch_size=pop,
                                    genome_config=RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0)),
                                )
                                t0 = time.perf_counter()
                                try:
                                    res_qnat = composer.quick_run(
                                        backend="malthusjax",
                                        strategy=MapElitesStrategy(
                                            emitter=native_emitter,
                                            centroids=centroids,
                                            num_centroids=num_centroids,
                                        ),
                                        fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
                                        pop_size=pop,
                                        generations=gens,
                                        genome_length=dim,
                                        bounds=(-5.0, 5.0),
                                        maximize=False,
                                        initial_population=initial_pop,
                                        seeds=(seed,),
                                    )
                                    dur_qnat = time.perf_counter() - t0
                                    run_obj = res_qnat.runs[0]
                                    bf = float(run_obj.metrics.get("best_fitness", np.nan))
                                    delta_f = max(bf - f_opt, 0.0)
                                    log10_err = float(np.log10(max(delta_f, epsilon)))
                                    block_data["runs"]["qdax_native"] = {
                                        "status": "success",
                                        "best_fitness": bf,
                                        "delta_f": delta_f,
                                        "log10_error": log10_err,
                                        "floor_hit": bool(delta_f <= epsilon),
                                        "coverage": float(run_obj.metrics.get("coverage", 0.0)),
                                        "qd_score": float(run_obj.metrics.get("qd_score", 0.0)),
                                        "collection_seconds": dur_qnat,
                                    }
                                except Exception as e:
                                    failures += 1
                                    block_data["runs"]["qdax_native"] = {"status": "error", "error": str(e)}

                            # Record paired differences
                            rec = {
                                "cell_id": cell_id,
                                "function": fn_name,
                                "instance": inst_seed,
                                "dim": dim,
                                "pop_size": pop,
                                "generations": gens,
                                "seed": seed,
                            }
                            if "evosax_native" in block_data["runs"] and "evosax_adapter" in block_data["runs"]:
                                r_nat = block_data["runs"]["evosax_native"]
                                r_ad = block_data["runs"]["evosax_adapter"]
                                if r_nat["status"] == "success" and r_ad["status"] == "success":
                                    rec["evosax_diff_log10"] = r_nat["log10_error"] - r_ad["log10_error"]
                                    rec["evosax_diff_raw"] = r_nat["best_fitness"] - r_ad["best_fitness"]
                            if "qdax_native" in block_data["runs"] and "qdax_adapter" in block_data["runs"]:
                                r_qnat = block_data["runs"]["qdax_native"]
                                r_qad = block_data["runs"]["qdax_adapter"]
                                if r_qnat["status"] == "success" and r_qad["status"] == "success":
                                    rec["qdax_diff_log10"] = r_qnat["log10_error"] - r_qad["log10_error"]
                                    rec["qdax_diff_raw"] = r_qnat["best_fitness"] - r_qad["best_fitness"]

                            records.append(rec)
                            with open(block_dir / "block.json", "w", encoding="utf-8") as f:
                                json.dump(block_data, f, indent=2)
                            blocks.append(block_data)

                            status_str = " | ".join(
                                f"{k}: {v['status']}" for k, v in block_data["runs"].items()
                            )
                            print(f"[{cell_id}] {status_str}", flush=True)
                            jax.clear_caches()

    # Write summary
    summary_data: Dict[str, Any] = {
        "schema_version": 1,
        "stage": args.stage,
        "total_blocks": len(blocks),
        "total_condition_runs": sum(len(b["runs"]) for b in blocks),
        "failed_runs": failures,
        "success": bool(failures == 0),
    }

    # Pilot analysis: calculate paired SD and power projection
    if args.stage == "pilot":
        evosax_diffs = [r["evosax_diff_log10"] for r in records if "evosax_diff_log10" in r]
        qdax_diffs = [r["qdax_diff_log10"] for r in records if "qdax_diff_log10" in r]

        evo_sd = float(np.std(evosax_diffs, ddof=1)) if len(evosax_diffs) > 1 else 0.0
        evo_mean = float(np.mean(evosax_diffs)) if evosax_diffs else 0.0
        qd_sd = float(np.std(qdax_diffs, ddof=1)) if len(qdax_diffs) > 1 else 0.0
        qd_mean = float(np.mean(qdax_diffs)) if qdax_diffs else 0.0

        margin = cfg["primary_metric"]["candidate_equivalence_margin"]

        # Approximate power calculation for TOST at alpha=0.05, true effect delta = margin/2
        def calc_n(sd: float, marg: float, power: float = 0.90) -> int:
            if sd <= 1e-9:
                return 10
            z_alpha = 1.64485  # alpha = 0.05 (one-sided for each TOST bound)
            z_beta = 1.28155   # 1 - beta = 0.90
            delta_target = marg / 2.0
            n_req = int(np.ceil(((z_alpha + z_beta) * sd / delta_target) ** 2))
            return max(10, n_req)

        # Wall-clock timing forecast
        all_times = []
        for b in blocks:
            for r in b["runs"].values():
                if "collection_seconds" in r:
                    all_times.append(r["collection_seconds"])
        mean_sec_per_run = float(np.mean(all_times)) if all_times else 1.0

        full_grid_runs = cfg["full_grid"]["candidate_n"] * 108 * 6  # 108 cells * n * 6 runs
        forecast_gpu_hours = (full_grid_runs * mean_sec_per_run) / 3600.0

        analysis = {
            "pilot_sample_size": len(seeds),
            "evosax_paired_mean_diff_log10": evo_mean,
            "evosax_paired_sd_log10": evo_sd,
            "evosax_required_n_at_90pct_power": calc_n(evo_sd, margin),
            "qdax_paired_mean_diff_log10": qd_mean,
            "qdax_paired_sd_log10": qd_sd,
            "qdax_required_n_at_90pct_power": calc_n(qd_sd, margin),
            "mean_wall_seconds_per_condition_run": mean_sec_per_run,
            "forecast_full_108_cell_runs": full_grid_runs,
            "forecast_full_gpu_hours_at_n100": forecast_gpu_hours,
        }
        summary_data["pilot_analysis"] = analysis

        with open(args.output / "pilot_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

    with open(args.output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n===========================================================================")
    print(f"  Stage {args.stage.upper()} Complete. Failures: {failures} | Blocks: {len(blocks)}")
    print(f"  Summary saved to: {args.output / 'summary.json'}")
    print(f"===========================================================================\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
