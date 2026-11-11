#!/usr/bin/env python3
"""Part 1 Confirmatory Runner: EvoSAX and QDax Parity & Steady-State Speed.

Implements the three-leg comparison (A: direct original, B: adapter, C: native mimic)
across the 6-function BBOB benchmark grid with:
1. Shared initial populations and archive centroids per seed block
2. Phase-separated AOT compilation and retained-executable steady-state timing
   (setup_seconds, compile_seconds, first_call_seconds, warmup_seconds, 5 steady repeats)
3. Counterbalanced condition execution order per seed block
4. Raw per-run (runs.csv) and per-repeat (timings.csv) tabular exports
5. Paired TOST equivalence, Wilcoxon signed-rank, 10,000-resample bootstrap CIs,
   Holm-Bonferroni / Benjamini-Hochberg adjustments, and power/GPU-budget sizing.

CLI Contract:
    python scripts/run_publication_part1_confirmatory.py \
        --config configs/publication/part1_confirmatory.json \
        --stage preflight|smoke|pilot|full \
        --output NEW_DIR \
        [--shard-index 0 --num-shards 1] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np


def digest(array: Any) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def installed_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def compute_archive_entropy(fitnesses: Any) -> float:
    """Normalized Shannon entropy in [0, 1] of shifted positive fitness mass across archive cells."""
    arr = np.asarray(fitnesses, dtype=np.float64)
    occupied = np.isfinite(arr)
    n_occ = int(np.sum(occupied))
    n_total = int(arr.size)
    if n_occ <= 1 or n_total <= 1:
        return 0.0
    vals = arr[occupied]
    weights = (vals - np.min(vals)) + 1.0
    probs = weights / np.sum(weights)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-15))))
    return float(entropy / np.log(n_total))


def capture_adapter_init(
    adapter: Any,
    seed: int,
    initial_pop: Any,
    centroids: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Capture production initialization state before AOT compilation/execution."""
    import jax
    from malthusjax.core.timing import TimingCapture, block_until_ready

    captured: Dict[str, Any] = {}

    class _Initialized(Exception):
        pass

    def _stop_after_init(self_cap: Any, inputs: Any, engine: Any) -> None:
        block_until_ready(inputs, effects=True)
        captured["inputs"] = inputs
        captured["engine"] = engine
        raise _Initialized()

    target = getattr(adapter, "engine", adapter)
    adapter.initial_population = initial_pop
    if hasattr(target, "framework_params"):
        target.initial_population = initial_pop
    if hasattr(adapter, "centroids") and centroids is not None:
        adapter.centroids = centroids
    if hasattr(target, "framework_params") and isinstance(target.framework_params, dict):
        if "init_variables" in target.framework_params:
            target.framework_params["init_variables"] = initial_pop
        if "centroids" in target.framework_params and centroids is not None:
            target.framework_params["centroids"] = centroids

    with patch.object(TimingCapture, "initialized", _stop_after_init):
        try:
            adapter.run_once(jax.random.PRNGKey(seed))
        except _Initialized:
            pass

    if "inputs" not in captured:
        raise RuntimeError(f"Adapter {type(adapter).__name__} did not trigger TimingCapture.initialized")
    return captured["engine"], captured["inputs"]


class PreparedConditionRunner:
    """Holds a cell-compiled AOT executable and extracts unified quality + timing metrics."""

    def __init__(
        self,
        name: str,
        setup_seconds: float,
        compile_seconds: float,
        init_args_fn: Callable[[int, Any, Any], Tuple[Any, ...]],
        executable: Any,
        extract_metrics_fn: Callable[[Any, Any], Dict[str, Any]],
        evaluations_per_scan: int,
    ) -> None:
        self.name = name
        self.setup_seconds = setup_seconds
        self.compile_seconds = compile_seconds
        self.init_args_fn = init_args_fn
        self.executable = executable
        self.extract_metrics_fn = extract_metrics_fn
        self.evaluations_per_scan = evaluations_per_scan

    def run_seed(
        self,
        seed: int,
        initial_pop: Any,
        centroids: Any,
        f_opt: float,
        epsilon: float,
        repetitions: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        import jax
        import jax.numpy as jnp
        from malthusjax.core.timing import block_until_ready

        def clone_args(a: Tuple[Any, ...]) -> Tuple[Any, ...]:
            return jax.tree_util.tree_map(
                lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else x, a
            )

        # 1. Seed state initialization
        t0 = time.perf_counter()
        base_args = self.init_args_fn(seed, initial_pop, centroids)
        block_until_ready(base_args, effects=True)
        init_state_seconds = time.perf_counter() - t0

        timing_rows: List[Dict[str, Any]] = []

        # 2. First prepared execution (also used to extract quality metrics)
        args_first = clone_args(base_args)
        block_until_ready(args_first, effects=True)
        t0 = time.perf_counter()
        out_first = self.executable(*args_first)
        block_until_ready(out_first, effects=True)
        first_call_seconds = time.perf_counter() - t0

        final_state, scan_history = out_first
        metrics = self.extract_metrics_fn(final_state, scan_history)

        ref_leaves, ref_struct = jax.tree_util.tree_flatten(out_first)
        ref_host = [
            np.asarray(jax.random.key_data(x))
            if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
            else np.asarray(x)
            for x in ref_leaves
        ]

        timing_rows.append({
            "repeat_idx": 0,
            "phase": "prepared_first_execution",
            "seconds": first_call_seconds,
            "evaluations_per_second": self.evaluations_per_scan / max(first_call_seconds, 1e-12),
            "output_agrees": True,
        })

        # 3. Discarded warmup call
        args_warm = clone_args(base_args)
        block_until_ready(args_warm, effects=True)
        t0 = time.perf_counter()
        out_warm = self.executable(*args_warm)
        block_until_ready(out_warm, effects=True)
        warmup_seconds = time.perf_counter() - t0

        timing_rows.append({
            "repeat_idx": 1,
            "phase": "warmup",
            "seconds": warmup_seconds,
            "evaluations_per_second": self.evaluations_per_scan / max(warmup_seconds, 1e-12),
            "output_agrees": True,
        })

        # 4. Synchronized steady-state repeats
        steady_times: List[float] = []
        all_agree = True
        for r_idx in range(repetitions):
            args_rep = clone_args(base_args)
            block_until_ready(args_rep, effects=True)
            t0 = time.perf_counter()
            out_rep = self.executable(*args_rep)
            block_until_ready(out_rep, effects=True)
            sec = time.perf_counter() - t0
            steady_times.append(sec)

            leaves, struct = jax.tree_util.tree_flatten(out_rep)
            host = [
                np.asarray(jax.random.key_data(x))
                if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else np.asarray(x)
                for x in leaves
            ]
            agrees = struct == ref_struct and all(
                np.array_equal(x, y, equal_nan=True) for x, y in zip(host, ref_host)
            )
            all_agree = all_agree and bool(agrees)
            timing_rows.append({
                "repeat_idx": 2 + r_idx,
                "phase": "prepared_execution_repeat",
                "seconds": sec,
                "evaluations_per_second": self.evaluations_per_scan / max(sec, 1e-12),
                "output_agrees": bool(agrees),
            })

        bf = float(metrics["best_fitness"])
        delta_f = max(bf - f_opt, 0.0)
        log10_err = float(np.log10(max(delta_f, epsilon)))
        steady_arr = np.asarray(steady_times, dtype=np.float64)
        q75, q25 = np.percentile(steady_arr, [75, 25]) if steady_arr.size > 0 else (0.0, 0.0)
        steady_median = float(np.median(steady_arr)) if steady_arr.size > 0 else 0.0

        result_dict: Dict[str, Any] = {
            "status": "success" if np.isfinite(bf) and all_agree else "error",
            "best_fitness": bf,
            "f_opt": f_opt,
            "delta_f": delta_f,
            "log10_error": log10_err,
            "floor_hit": bool(delta_f <= epsilon),
            "metrics": {
                "best_fitness": bf,
                "coverage": metrics.get("coverage"),
                "qd_score": metrics.get("qd_score"),
                "max_fitness": metrics.get("max_fitness"),
                "archive_entropy": metrics.get("archive_entropy"),
                "occupied_cells": metrics.get("occupied_cells"),
            },
            "coverage": metrics.get("coverage"),
            "qd_score": metrics.get("qd_score"),
            "max_fitness": metrics.get("max_fitness"),
            "archive_entropy": metrics.get("archive_entropy"),
            "occupied_cells": metrics.get("occupied_cells"),
            "total_evaluations": self.evaluations_per_scan + int(initial_pop.shape[0]),
            "setup_seconds": self.setup_seconds,
            "compile_seconds": self.compile_seconds,
            "init_state_seconds": init_state_seconds,
            "first_call_seconds": first_call_seconds,
            "warmup_seconds": warmup_seconds,
            "steady_repeat_seconds": steady_times,
            "steady_median_seconds": steady_median,
            "steady_iqr_seconds": float(q75 - q25),
            "steady_mean_seconds": float(np.mean(steady_arr)) if steady_arr.size > 0 else 0.0,
            "steady_std_seconds": float(np.std(steady_arr)) if steady_arr.size > 0 else 0.0,
            "evaluations_per_second": float(self.evaluations_per_scan / max(steady_median, 1e-12)),
            "collection_wall_seconds": init_state_seconds + first_call_seconds,
            "repeat_output_agreement": all_agree,
        }
        return result_dict, timing_rows


def build_cell_runners(
    fn_name: str,
    inst_seed: int,
    dim: int,
    pop: int,
    gens: int,
    num_centroids: int,
    seed0: int,
    init0: Any,
    cent0: Any,
    library: str,
    cfg: Dict[str, Any],
) -> Dict[str, PreparedConditionRunner]:
    """Instantiate and AOT-compile all requested conditions for a single grid cell."""
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import SimpleGA
    from malthusjax.composer.engine_factory_v2 import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig
    from malthusjax.composer.strategies.core import MapElitesStrategy, QDAXStrategy
    from malthusjax.core.fitness.composable.environments import BBOBEnv
    from malthusjax.core.fitness.qd.metrics import archive_metrics
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.core.timing import block_until_ready
    from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites

    env = BBOBEnv.create(fn_name=fn_name, num_dims=dim, seed=inst_seed)
    evo_params = cfg["condition_parameters"]["evosax"]
    crossover_rate = float(evo_params["crossover_rate"])
    elite_ratio = float(evo_params["elite_ratio"])

    qd_params = cfg["condition_parameters"]["qdax"]
    var_pct = float(qd_params["qd_variation_percentage"])
    mut_sigma = float(qd_params["qd_mutation_sigma"])
    evals_per_scan = pop * gens

    runners: Dict[str, PreparedConditionRunner] = {}

    if library in ("both", "evosax"):
        # 1A. evosax_direct
        t0 = time.perf_counter()
        strategy = SimpleGA(population_size=pop, solution=jnp.zeros(dim))
        strategy.elite_ratio = elite_ratio
        params = strategy.default_params.replace(crossover_rate=crossover_rate)

        def _evo_dir_init(s: int, ipop: Any, _cent: Any) -> Tuple[Any, ...]:
            k_loop, k_init, _ = jax.random.split(jax.random.PRNGKey(s), 3)
            f_init = jax.vmap(env.evaluate)(ipop)
            st_init = strategy.init(k_init, ipop, f_init, params)
            return (k_loop, st_init)

        def _evo_dir_scan(k_loop: Any, st_init: Any) -> Tuple[Any, Any]:
            def _step(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
                rng, st = carry
                rng, k_step = jax.random.split(rng)
                k_ask, _k_eval, k_tell = jax.random.split(k_step, 3)
                x, st = strategy.ask(k_ask, st, params)
                f = jax.vmap(env.evaluate)(x)
                st, _ = strategy.tell(k_tell, x, f, st, params)
                return (rng, st), {"best_fitness": st.best_fitness}

            (_, st_final), hist = jax.lax.scan(_step, (k_loop, st_init), None, length=gens, unroll=1)
            return st_final, hist

        setup_sec = time.perf_counter() - t0
        args0 = _evo_dir_init(seed0, init0, cent0)
        block_until_ready(args0, effects=True)
        t_c0 = time.perf_counter()
        exe_evo_dir = jax.jit(_evo_dir_scan).lower(*args0).compile()
        compile_sec = time.perf_counter() - t_c0

        runners["evosax_direct"] = PreparedConditionRunner(
            name="evosax_direct",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=_evo_dir_init,
            executable=exe_evo_dir,
            extract_metrics_fn=lambda st, hist: {"best_fitness": float(hist["best_fitness"][-1])},
            evaluations_per_scan=evals_per_scan,
        )

        # 1B. evosax_adapter
        t0 = time.perf_counter()
        opts_evo_ad = dict(
            backend="evosax",
            evosax_strategy="SimpleGA",
            strategy_params={"crossover_rate": crossover_rate, "elite_ratio": elite_ratio},
            fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
            pop_size=pop,
            generations=gens,
            genome_length=dim,
            bounds=(-5.0, 5.0),
            maximize=False,
            initial_population=init0,
        )
        ad_evo = EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**opts_evo_ad))
        setup_sec = time.perf_counter() - t0
        eng_evo_ad, inp0_evo_ad = capture_adapter_init(ad_evo, seed0, init0)
        t_c0 = time.perf_counter()
        exe_evo_ad = eng_evo_ad._build_jit_loop().lower(*tuple(inp0_evo_ad)).compile()
        compile_sec = time.perf_counter() - t_c0

        runners["evosax_adapter"] = PreparedConditionRunner(
            name="evosax_adapter",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=lambda s, ipop, _c, _ad=ad_evo: tuple(capture_adapter_init(_ad, s, ipop)[1]),
            executable=exe_evo_ad,
            extract_metrics_fn=lambda _st, hist: {"best_fitness": float(hist["best_fitness"][-1])},
            evaluations_per_scan=evals_per_scan,
        )

        # 1C. evosax_native
        t0 = time.perf_counter()
        opts_evo_nat = dict(
            backend="malthusjax",
            selection=f"evosax_mimic_selection:num_selections={pop},elite_k={max(2, int(pop / 6))}",
            crossover=f"evosax_uniform_crossover:crossover_rate={crossover_rate}",
            mutation="evosax_gaussian:mutation_strength=1.0",
            elitism=0,
            fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
            pop_size=pop,
            generations=gens,
            genome_length=dim,
            bounds=(-5.0, 5.0),
            maximize=False,
            initial_population=init0,
        )
        ad_evo_nat = EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**opts_evo_nat))
        setup_sec = time.perf_counter() - t0
        eng_evo_nat, inp0_evo_nat = capture_adapter_init(ad_evo_nat, seed0, init0)
        t_c0 = time.perf_counter()
        exe_evo_nat = eng_evo_nat.prepare_run(inp0_evo_nat).executable
        compile_sec = time.perf_counter() - t_c0

        runners["evosax_native"] = PreparedConditionRunner(
            name="evosax_native",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=lambda s, ipop, _c, _ad=ad_evo_nat: (capture_adapter_init(_ad, s, ipop)[1],),
            executable=exe_evo_nat,
            extract_metrics_fn=lambda st, _hist: {"best_fitness": float(st.best_fitness)},
            evaluations_per_scan=evals_per_scan,
        )

    if library in ("both", "qdax"):
        def _q_mut(x: Any, rng: Any) -> Any:
            return x + mut_sigma * jax.random.normal(rng, x.shape)

        def _q_var(x: Any, y: Any, rng: Any) -> Any:
            return jnp.where(jax.random.bernoulli(rng, var_pct, x.shape), x, y)

        # 2A. qdax_direct
        t0 = time.perf_counter()
        up_emitter_dir = MixingEmitter(
            mutation_fn=_q_mut,
            variation_fn=_q_var,
            variation_percentage=var_pct,
            batch_size=pop,
        )

        def _scoring_fn(x: Any, _k: Any) -> Tuple[Any, Any, Dict[str, Any]]:
            fit = -jax.vmap(env.evaluate)(x)
            desc = jnp.clip((x[:, :2] + 5.0) / 10.0, 0.0, 1.0)
            return fit, desc, {}

        map_elites = MAPElites(
            scoring_function=_scoring_fn,
            emitter=up_emitter_dir,
            metrics_function=lambda _r: {},
        )

        def _qd_dir_init(s: int, ipop: Any, cent: Any) -> Tuple[Any, ...]:
            _k_loop, k_init, _ = jax.random.split(jax.random.PRNGKey(s), 3)
            rep_init, em_init, _ = map_elites.init(ipop, cent, k_init)
            return ((rep_init, em_init, k_init),)

        def _qd_dir_scan(state_init: Any) -> Tuple[Any, Any]:
            def _step(carry: Tuple[Any, Any, Any], _: Any) -> Tuple[Tuple[Any, Any, Any], Any]:
                rep, em_st, rkey = carry
                rkey, subk = jax.random.split(rkey)
                rep, em_st, _ = map_elites.update(rep, em_st, subk)
                return (rep, em_st, rkey), archive_metrics(rep.fitnesses)

            final_carry, hist = jax.lax.scan(_step, state_init, None, length=gens, unroll=1)
            return final_carry, hist

        setup_sec = time.perf_counter() - t0
        args0_qd = _qd_dir_init(seed0, init0, cent0)
        block_until_ready(args0_qd, effects=True)
        t_c0 = time.perf_counter()
        exe_qd_dir = jax.jit(_qd_dir_scan).lower(*args0_qd).compile()
        compile_sec = time.perf_counter() - t_c0

        runners["qdax_direct"] = PreparedConditionRunner(
            name="qdax_direct",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=_qd_dir_init,
            executable=exe_qd_dir,
            extract_metrics_fn=lambda st, hist: {
                "best_fitness": -float(hist["max_fitness"][-1]),
                "max_fitness": float(hist["max_fitness"][-1]),
                "qd_score": float(hist["qd_score"][-1]),
                "coverage": float(hist["coverage"][-1]),
                "occupied_cells": int(hist["occupied_cells"][-1]),
                "archive_entropy": compute_archive_entropy(st[0].fitnesses),
            },
            evaluations_per_scan=evals_per_scan,
        )

        # 2B. qdax_adapter
        t0 = time.perf_counter()
        up_emitter_ad = MixingEmitter(
            mutation_fn=_q_mut,
            variation_fn=_q_var,
            variation_percentage=var_pct,
            batch_size=pop,
        )
        opts_qd_ad = dict(
            backend="qdax",
            strategy=QDAXStrategy(
                emitter=up_emitter_ad,
                centroids=cent0,
                num_centroids=num_centroids,
            ),
            fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
            pop_size=pop,
            generations=gens,
            genome_length=dim,
            bounds=(-5.0, 5.0),
            maximize=False,
            initial_population=init0,
        )
        ad_qd = EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**opts_qd_ad))
        setup_sec = time.perf_counter() - t0
        eng_qd_ad, inp0_qd_ad = capture_adapter_init(ad_qd, seed0, init0, cent0)
        t_c0 = time.perf_counter()
        exe_qd_ad = eng_qd_ad._build_jit_loop().lower(*tuple(inp0_qd_ad)).compile()
        compile_sec = time.perf_counter() - t_c0

        runners["qdax_adapter"] = PreparedConditionRunner(
            name="qdax_adapter",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=lambda s, ipop, cent, _ad=ad_qd: tuple(
                capture_adapter_init(_ad, s, ipop, cent)[1]
            ),
            executable=exe_qd_ad,
            extract_metrics_fn=lambda st, hist: {
                "best_fitness": float(hist["best_fitness"][-1]),
                "max_fitness": -float(hist["best_fitness"][-1]),
                "qd_score": float(hist["qd_score"][-1]),
                "coverage": float(hist["coverage"][-1]),
                "occupied_cells": int(np.sum(np.isfinite(np.asarray(st[0].fitnesses)))),
                "archive_entropy": compute_archive_entropy(st[0].fitnesses),
            },
            evaluations_per_scan=evals_per_scan,
        )

        # 2C. qdax_native
        t0 = time.perf_counter()
        nat_emitter = QDAXReplicaMixingEmitter(
            mutation_fn=_q_mut,
            variation_fn=_q_var,
            variation_percentage=var_pct,
            _batch_size=pop,
            genome_config=RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0)),
        )
        opts_qd_nat = dict(
            backend="malthusjax",
            strategy=MapElitesStrategy(
                emitter=nat_emitter,
                centroids=cent0,
                num_centroids=num_centroids,
            ),
            fitness=f"bbob:fn_name={fn_name},num_dims={dim},maximize=false,seed={inst_seed}",
            pop_size=pop,
            generations=gens,
            genome_length=dim,
            bounds=(-5.0, 5.0),
            maximize=False,
            initial_population=init0,
        )
        ad_qd_nat = EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**opts_qd_nat))
        setup_sec = time.perf_counter() - t0
        eng_qd_nat, inp0_qd_nat = capture_adapter_init(ad_qd_nat, seed0, init0, cent0)
        t_c0 = time.perf_counter()
        exe_qd_nat = eng_qd_nat.prepare_run(inp0_qd_nat).executable
        compile_sec = time.perf_counter() - t_c0

        runners["qdax_native"] = PreparedConditionRunner(
            name="qdax_native",
            setup_seconds=setup_sec,
            compile_seconds=compile_sec,
            init_args_fn=lambda s, ipop, cent, _ad=ad_qd_nat: (
                capture_adapter_init(_ad, s, ipop, cent)[1],
            ),
            executable=exe_qd_nat,
            extract_metrics_fn=lambda st, hist: {
                "best_fitness": float(st.best_fitness),
                "max_fitness": -float(st.best_fitness),
                "qd_score": float(hist.qd_score[-1]),
                "coverage": float(hist.coverage[-1]),
                "occupied_cells": int(np.sum(np.isfinite(np.asarray(st.repertoire.fitnesses)))),
                "archive_entropy": compute_archive_entropy(st.repertoire.fitnesses),
            },
            evaluations_per_scan=evals_per_scan,
        )

    return runners


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni FWER adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, p_adj)
        adjusted[idx] = running_max
    return [float(x) for x in adjusted]


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Step-up Benjamini-Hochberg FDR adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        rank = rank_rev + 1
        p_adj = min(1.0, p_values[idx] * m / rank)
        running_min = min(running_min, p_adj)
        adjusted[idx] = running_min
    return [float(x) for x in adjusted]


def compute_paired_stats(
    diffs: List[float],
    margin: float,
    sensitivity_margins: List[float],
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """Compute parametric paired TOST, Wilcoxon, and bootstrap CIs."""
    from scipy import stats

    arr = np.asarray([d for d in diffs if np.isfinite(d)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "status": "empty"}

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    max_abs = float(np.max(np.abs(arr)))

    if n < 2 or max_abs == 0.0:
        sd = 0.0
        se = 0.0
        tost_p = 0.0 if abs(mean) < margin else 1.0
        ci90 = [mean, mean]
        ci95 = [mean, mean]
        wilcoxon_p = 1.0
        boot_ci95_median = [median, median]
    else:
        sd = float(np.std(arr, ddof=1))
        se = sd / math.sqrt(n)
        df = n - 1
        t_crit90 = float(stats.t.ppf(0.95, df))
        t_crit95 = float(stats.t.ppf(0.975, df))
        ci90 = [float(mean - t_crit90 * se), float(mean + t_crit90 * se)]
        ci95 = [float(mean - t_crit95 * se), float(mean + t_crit95 * se)]

        if se <= 1e-15:
            tost_p = 0.0 if abs(mean) < margin else 1.0
        else:
            t_low = (mean - (-margin)) / se
            t_high = (margin - mean) / se
            p_low = float(1.0 - stats.t.cdf(t_low, df))
            p_high = float(1.0 - stats.t.cdf(t_high, df))
            tost_p = max(p_low, p_high)

        try:
            nonzero = arr[arr != 0.0]
            if nonzero.size >= 2:
                _, wilcoxon_p = stats.wilcoxon(arr)
                wilcoxon_p = float(wilcoxon_p)
            else:
                wilcoxon_p = 1.0
        except Exception:
            wilcoxon_p = 1.0

        rng = np.random.default_rng(42)
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot_medians = np.median(arr[boot_idx], axis=1)
        boot_ci95_median = [
            float(np.percentile(boot_medians, 2.5)),
            float(np.percentile(boot_medians, 97.5)),
        ]

    def calc_req_n(s_d: float, marg: float, power: float = 0.90) -> int:
        if s_d <= 1e-9:
            return 10
        z_alpha = 1.64485  # one-sided alpha = 0.05
        z_beta = 1.28155 if power >= 0.90 else 0.84162
        delta_target = marg / 2.0
        return max(10, int(math.ceil(((z_alpha + z_beta) * s_d / delta_target) ** 2)))

    sens: Dict[str, Any] = {}
    for m_val in sensitivity_margins:
        if n >= 2 and se > 1e-15:
            tl = (mean - (-m_val)) / se
            th = (m_val - mean) / se
            tp = max(float(1.0 - stats.t.cdf(tl, n - 1)), float(1.0 - stats.t.cdf(th, n - 1)))
        else:
            tp = 0.0 if abs(mean) < m_val else 1.0
        sens[f"margin_{m_val}"] = {
            "margin": m_val,
            "tost_p": tp,
            "equivalent_at_alpha_0_05": bool(tp < 0.05),
            "required_n_80pct_power": calc_req_n(sd, m_val, 0.80),
            "required_n_90pct_power": calc_req_n(sd, m_val, 0.90),
        }

    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "median_diff": median,
        "iqr_diff": iqr,
        "max_abs_diff": max_abs,
        "exact_zero_parity": bool(max_abs == 0.0),
        "tost_90_ci": ci90,
        "ordinary_95_ci": ci95,
        "bootstrap_95_ci_median": boot_ci95_median,
        "tost_p_value": tost_p,
        "equivalent_at_candidate_margin": bool(tost_p < 0.05),
        "wilcoxon_p_value": wilcoxon_p,
        "required_n_80pct_power": calc_req_n(sd, margin, 0.80),
        "required_n_90pct_power": calc_req_n(sd, margin, 0.90),
        "margin_sensitivity": sens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/publication/part1_confirmatory.json"))
    parser.add_argument("--stage", choices=["preflight", "smoke", "pilot", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", choices=["evosax", "qdax", "both"], default="both")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip already completed seed blocks")
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("Require 0 <= --shard-index < --num-shards")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.output.mkdir(parents=True, exist_ok=True)

    # Stage 1: PREFLIGHT
    if args.stage == "preflight":
        full_grid = cfg["full_grid"]
        pilot_grid = cfg["pilot_grid"]

        total_cells_full = (
            len(full_grid["functions"])
            * len(full_grid["instances"])
            * len(full_grid["dims"])
            * len(full_grid["populations"])
            * len(full_grid["generations"])
        )
        num_seeds_full = full_grid["candidate_n"]
        libs = 2 if args.library == "both" else 1
        total_runs_full = total_cells_full * num_seeds_full * (3 * libs)

        total_cells_pilot = (
            len(pilot_grid["functions"])
            * len(pilot_grid["instances"])
            * len(pilot_grid["dims"])
            * len(pilot_grid["populations"])
            * len(pilot_grid["generations"])
        )
        num_seeds_pilot = len(pilot_grid["seeds"])
        total_runs_pilot = total_cells_pilot * num_seeds_pilot * (3 * libs)

        preflight_summary = {
            "schema_version": 2,
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
            "timing_protocol": cfg["timing_protocol"],
        }

        out_path = args.output / "resolution.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preflight_summary, f, indent=2)
        print(f"Preflight resolution complete. Written to {out_path}")
        print(f"  • Full Grid:  {total_cells_full} cells, {total_runs_full} condition runs ({num_seeds_full} seeds)")
        print(f"  • Pilot Grid: {total_cells_pilot} cells, {total_runs_pilot} condition runs ({num_seeds_pilot} seeds)")
        return 0

    if args.stage == "smoke":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp

    active_grid = dict(cfg["smoke_grid"] if args.stage == "smoke" else cfg["pilot_grid"])
    if args.stage == "full":
        active_grid = dict(cfg["full_grid"])
        active_grid["seeds"] = list(
            range(active_grid["seeds_range"][0], active_grid["seeds_range"][1] + 1)
        )
        active_grid["timing_repetitions"] = cfg["timing_protocol"]["steady_repetitions"]

    functions = active_grid["functions"]
    instances = active_grid.get("instances", [1])
    dims = active_grid["dims"]
    populations = active_grid["populations"]
    generations_list = active_grid["generations"]
    seeds = active_grid["seeds"]
    repetitions = int(active_grid.get("timing_repetitions", 5))
    epsilon = float(cfg["primary_metric"]["epsilon"])
    margin = float(cfg["primary_metric"]["candidate_equivalence_margin"])
    sens_margins = [float(x) for x in cfg["primary_metric"].get("sensitivity_margins", [0.025, 0.05, 0.10])]

    all_cells: List[Tuple[str, int, int, int, int]] = [
        (fn_name, inst_seed, dim, pop, gens)
        for fn_name in functions
        for inst_seed in instances
        for dim in dims
        for pop in populations
        for gens in generations_list
    ]
    my_cells = [c for idx, c in enumerate(all_cells) if (idx % args.num_shards) == args.shard_index]

    manifest = {
        "schema_version": 2,
        "stage": args.stage,
        "library": args.library,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_cells_in_stage": len(all_cells),
        "cells_in_this_shard": len(my_cells),
        "python": sys.version,
        "platform": platform.platform(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "devices": [str(d) for d in jax.devices()],
        "versions": {
            name: installed_version(name) for name in ["jax", "jaxlib", "evosax", "qdax", "numpy", "scipy"]
        },
        "grid": active_grid,
        "primary_metric": cfg["primary_metric"],
        "timing_protocol": cfg["timing_protocol"],
    }
    with open(args.output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    blocks: List[Dict[str, Any]] = []
    run_csv_rows: List[Dict[str, Any]] = []
    timing_csv_rows: List[Dict[str, Any]] = []
    paired_records: List[Dict[str, Any]] = []
    failures = 0

    print(f"\n===========================================================================")
    print(f"  MALTHUSJAX PART 1 CONFIRMATORY (Stage: {args.stage.upper()} | Backend: {jax.default_backend()})")
    print(f"  Cells: {len(my_cells)}/{len(all_cells)} | Seeds/cell: {len(seeds)} | Steady repeats: {repetitions}")
    print(f"  Destination: {args.output}")
    print(f"===========================================================================\n")

    from malthusjax.core.fitness.composable.environments import BBOBEnv

    num_centroids = 16 if args.stage == "smoke" else int(cfg["condition_parameters"]["qdax"]["num_centroids"])

    for cell_idx, (fn_name, inst_seed, dim, pop, gens) in enumerate(my_cells):
        cell_coord = f"{fn_name}_i{inst_seed}_d{dim}_p{pop}_g{gens}"
        t_cell_start = time.perf_counter()

        # Clear caches at cell boundary so each cell records genuine cold compilation
        jax.clear_caches()

        env = BBOBEnv.create(fn_name=fn_name, num_dims=dim, seed=inst_seed)
        f_opt = float(env.f_opt)

        seed0 = seeds[0]
        k0 = jax.random.PRNGKey(seed0)
        init0 = jax.random.uniform(jax.random.fold_in(k0, 101), (pop, dim), minval=-5.0, maxval=5.0)
        cent0 = jax.random.uniform(jax.random.fold_in(k0, 102), (num_centroids, 2))

        runners = build_cell_runners(
            fn_name=fn_name,
            inst_seed=inst_seed,
            dim=dim,
            pop=pop,
            gens=gens,
            num_centroids=num_centroids,
            seed0=seed0,
            init0=init0,
            cent0=cent0,
            library=args.library,
            cfg=cfg,
        )
        base_cond_names = list(runners.keys())

        for s_idx, seed in enumerate(seeds):
            cell_id = f"{cell_coord}_s{seed}"
            block_dir = args.output / cell_id
            block_json_path = block_dir / "block.json"

            if args.resume and block_json_path.exists():
                with open(block_json_path, "r", encoding="utf-8") as f:
                    cached_block = json.load(f)
                if all(r.get("status") == "success" for r in cached_block.get("runs", {}).values()):
                    blocks.append(cached_block)
                    continue

            block_dir.mkdir(parents=True, exist_ok=True)
            key = jax.random.PRNGKey(seed)
            k_init = jax.random.fold_in(key, 101)
            k_cent = jax.random.fold_in(key, 102)

            initial_pop = jax.random.uniform(k_init, (pop, dim), minval=-5.0, maxval=5.0)
            centroids = jax.random.uniform(k_cent, (num_centroids, 2))

            np.save(block_dir / "initial_population.npy", np.asarray(initial_pop))
            np.save(block_dir / "centroids.npy", np.asarray(centroids))

            # Counterbalanced condition execution order per seed block
            shift = s_idx % len(base_cond_names)
            ordered_conds = base_cond_names[shift:] + base_cond_names[:shift]

            init_hash = digest(initial_pop)
            cent_hash = digest(centroids)

            block_data: Dict[str, Any] = {
                "cell_id": cell_id,
                "coordinate": cell_coord,
                "function": fn_name,
                "instance": inst_seed,
                "dim": dim,
                "pop_size": pop,
                "generations": gens,
                "seed": seed,
                "f_opt": f_opt,
                "order": ordered_conds,
                "initial_pop_hash": init_hash,
                "centroids_hash": cent_hash,
                "runs": {},
            }

            for order_rank, cond_name in enumerate(ordered_conds):
                runner = runners[cond_name]
                try:
                    res_dict, t_rows = runner.run_seed(
                        seed=seed,
                        initial_pop=initial_pop,
                        centroids=centroids,
                        f_opt=f_opt,
                        epsilon=epsilon,
                        repetitions=repetitions,
                    )
                    res_dict["execution_order_rank"] = order_rank
                    res_dict["input_population_hash"] = init_hash
                    if res_dict["status"] != "success":
                        failures += 1
                    block_data["runs"][cond_name] = res_dict

                    with open(block_dir / f"{cond_name}.json", "w", encoding="utf-8") as f:
                        json.dump(res_dict, f, indent=2)

                    run_csv_rows.append({
                        "cell_id": cell_id,
                        "coordinate": cell_coord,
                        "function": fn_name,
                        "instance": inst_seed,
                        "dim": dim,
                        "pop_size": pop,
                        "generations": gens,
                        "seed": seed,
                        "condition": cond_name,
                        "execution_order_rank": order_rank,
                        "status": res_dict["status"],
                        "best_fitness": res_dict["best_fitness"],
                        "f_opt": f_opt,
                        "delta_f": res_dict["delta_f"],
                        "log10_error": res_dict["log10_error"],
                        "floor_hit": res_dict["floor_hit"],
                        "qd_score": res_dict["qd_score"],
                        "coverage": res_dict["coverage"],
                        "max_fitness": res_dict["max_fitness"],
                        "archive_entropy": res_dict["archive_entropy"],
                        "occupied_cells": res_dict["occupied_cells"],
                        "setup_seconds": res_dict["setup_seconds"],
                        "compile_seconds": res_dict["compile_seconds"],
                        "init_state_seconds": res_dict["init_state_seconds"],
                        "first_call_seconds": res_dict["first_call_seconds"],
                        "warmup_seconds": res_dict["warmup_seconds"],
                        "steady_median_seconds": res_dict["steady_median_seconds"],
                        "steady_iqr_seconds": res_dict["steady_iqr_seconds"],
                        "steady_mean_seconds": res_dict["steady_mean_seconds"],
                        "steady_std_seconds": res_dict["steady_std_seconds"],
                        "evaluations_per_second": res_dict["evaluations_per_second"],
                    })

                    for tr in t_rows:
                        timing_csv_rows.append({
                            "cell_id": cell_id,
                            "coordinate": cell_coord,
                            "function": fn_name,
                            "instance": inst_seed,
                            "dim": dim,
                            "pop_size": pop,
                            "generations": gens,
                            "seed": seed,
                            "condition": cond_name,
                            "execution_order_rank": order_rank,
                            **tr,
                        })
                except Exception as exc:
                    failures += 1
                    err_obj = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    block_data["runs"][cond_name] = err_obj
                    with open(block_dir / f"{cond_name}.json", "w", encoding="utf-8") as f:
                        json.dump(err_obj, f, indent=2)

            # Paired record per seed block
            rec: Dict[str, Any] = {
                "cell_id": cell_id,
                "coordinate": cell_coord,
                "function": fn_name,
                "instance": inst_seed,
                "dim": dim,
                "pop_size": pop,
                "generations": gens,
                "seed": seed,
            }
            runs_map = block_data["runs"]
            if all(k in runs_map and runs_map[k]["status"] == "success" for k in ("evosax_direct", "evosax_adapter", "evosax_native")):
                r_d, r_b, r_c = runs_map["evosax_direct"], runs_map["evosax_adapter"], runs_map["evosax_native"]
                rec["evosax_B_minus_A_log10"] = r_b["log10_error"] - r_d["log10_error"]
                rec["evosax_C_minus_B_log10"] = r_c["log10_error"] - r_b["log10_error"]
                rec["evosax_C_minus_A_log10"] = r_c["log10_error"] - r_d["log10_error"]
                rec["evosax_speedup_C_vs_B"] = r_b["steady_median_seconds"] / max(r_c["steady_median_seconds"], 1e-12)
                rec["evosax_speedup_C_vs_A"] = r_d["steady_median_seconds"] / max(r_c["steady_median_seconds"], 1e-12)

            if all(k in runs_map and runs_map[k]["status"] == "success" for k in ("qdax_direct", "qdax_adapter", "qdax_native")):
                q_d, q_b, q_c = runs_map["qdax_direct"], runs_map["qdax_adapter"], runs_map["qdax_native"]
                rec["qdax_B_minus_A_log10"] = q_b["log10_error"] - q_d["log10_error"]
                rec["qdax_C_minus_B_log10"] = q_c["log10_error"] - q_b["log10_error"]
                rec["qdax_C_minus_A_log10"] = q_c["log10_error"] - q_d["log10_error"]
                rec["qdax_B_minus_A_qd_score"] = q_b["qd_score"] - q_d["qd_score"]
                rec["qdax_C_minus_B_qd_score"] = q_c["qd_score"] - q_b["qd_score"]
                rec["qdax_C_minus_B_qd_score_rel"] = (q_c["qd_score"] - q_b["qd_score"]) / max(abs(q_b["qd_score"]), 1.0)
                rec["qdax_C_minus_B_coverage"] = q_c["coverage"] - q_b["coverage"]
                rec["qdax_C_minus_B_entropy"] = q_c["archive_entropy"] - q_b["archive_entropy"]
                rec["qdax_speedup_C_vs_B"] = q_b["steady_median_seconds"] / max(q_c["steady_median_seconds"], 1e-12)
                rec["qdax_speedup_C_vs_A"] = q_d["steady_median_seconds"] / max(q_c["steady_median_seconds"], 1e-12)

            paired_records.append(rec)
            with open(block_json_path, "w", encoding="utf-8") as f:
                json.dump(block_data, f, indent=2)
            blocks.append(block_data)

        cell_elapsed = time.perf_counter() - t_cell_start
        print(
            f"[Cell {cell_idx + 1}/{len(my_cells)}: {cell_coord}] "
            f"{len(seeds)} seeds × {len(base_cond_names)} conditions completed in {cell_elapsed:.2f}s",
            flush=True,
        )

    # Export flat CSV artifacts
    if run_csv_rows:
        runs_csv_path = args.output / "runs.csv"
        with open(runs_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_csv_rows)

    if timing_csv_rows:
        timings_csv_path = args.output / "timings.csv"
        with open(timings_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(timing_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timing_csv_rows)

    # Compute confirmatory / pilot statistical analysis
    contrast_defs = [
        ("evosax_B_vs_A_log10", "evosax_B_minus_A_log10", margin),
        ("evosax_C_vs_B_log10", "evosax_C_minus_B_log10", margin),
        ("evosax_C_vs_A_log10", "evosax_C_minus_A_log10", margin),
        ("qdax_B_vs_A_log10", "qdax_B_minus_A_log10", margin),
        ("qdax_C_vs_B_log10", "qdax_C_minus_B_log10", margin),
        ("qdax_C_vs_A_log10", "qdax_C_minus_A_log10", margin),
        ("qdax_C_vs_B_qd_score_rel", "qdax_C_minus_B_qd_score_rel", 0.05),
        ("qdax_C_vs_B_coverage", "qdax_C_minus_B_coverage", 5.0),
        ("qdax_C_vs_B_entropy", "qdax_C_minus_B_entropy", 0.05),
    ]

    pooled_contrasts: Dict[str, Any] = {}
    primary_p_vals: List[float] = []
    primary_keys: List[str] = []

    for c_label, field_key, c_margin in contrast_defs:
        vals = [r[field_key] for r in paired_records if field_key in r]
        if vals:
            st = compute_paired_stats(vals, c_margin, sens_margins)
            pooled_contrasts[c_label] = st
            primary_p_vals.append(st["tost_p_value"])
            primary_keys.append(c_label)

    holm_adj = holm_bonferroni(primary_p_vals)
    bh_adj = benjamini_hochberg(primary_p_vals)
    for k, hp, bp in zip(primary_keys, holm_adj, bh_adj):
        pooled_contrasts[k]["tost_p_holm_bonferroni"] = hp
        pooled_contrasts[k]["tost_p_benjamini_hochberg"] = bp

    # Per-cell breakdown for variance sizing & FDR
    by_coord: Dict[str, List[Dict[str, Any]]] = {}
    for r in paired_records:
        by_coord.setdefault(r["coordinate"], []).append(r)

    cell_summaries: Dict[str, Any] = {}
    evo_cell_req_n: List[int] = []
    qd_cell_req_n: List[int] = []
    cell_evo_p: List[float] = []
    cell_qd_p: List[float] = []
    cell_coords_ordered = sorted(by_coord.keys())

    for coord in cell_coords_ordered:
        c_recs = by_coord[coord]
        c_evo = compute_paired_stats(
            [r["evosax_C_minus_B_log10"] for r in c_recs if "evosax_C_minus_B_log10" in r],
            margin,
            sens_margins,
            n_boot=500,
        )
        c_qd = compute_paired_stats(
            [r["qdax_C_minus_B_log10"] for r in c_recs if "qdax_C_minus_B_log10" in r],
            margin,
            sens_margins,
            n_boot=500,
        )
        if c_evo.get("n", 0) > 1:
            evo_cell_req_n.append(c_evo["required_n_90pct_power"])
            cell_evo_p.append(c_evo["tost_p_value"])
        if c_qd.get("n", 0) > 1:
            qd_cell_req_n.append(c_qd["required_n_90pct_power"])
            cell_qd_p.append(c_qd["tost_p_value"])
        cell_summaries[coord] = {
            "evosax_C_vs_B_log10": c_evo,
            "qdax_C_vs_B_log10": c_qd,
        }

    # Timing & GPU budget projection (separating per-cell compile from per-seed steady run)
    cond_steady_medians: Dict[str, List[float]] = {}
    cond_compile_times: Dict[str, List[float]] = {}
    for row in run_csv_rows:
        c = row["condition"]
        cond_steady_medians.setdefault(c, []).append(float(row["steady_median_seconds"]))
        cond_compile_times.setdefault(c, []).append(float(row["compile_seconds"]))

    timing_summary: Dict[str, Any] = {}
    total_compile_per_cell = 0.0
    total_exec_per_seed_block = 0.0
    for c_name, med_list in cond_steady_medians.items():
        comp_mean = float(np.mean(cond_compile_times[c_name]))
        steady_med = float(np.median(med_list))
        steady_q75, steady_q25 = np.percentile(med_list, [75, 25])
        timing_summary[c_name] = {
            "mean_compile_seconds": comp_mean,
            "median_steady_seconds": steady_med,
            "iqr_steady_seconds": float(steady_q75 - steady_q25),
        }
        total_compile_per_cell += comp_mean
        total_exec_per_seed_block += steady_med * (2 + repetitions)

    full_cells_count = 108
    candidate_n_full = int(cfg["full_grid"]["candidate_n"])
    projected_full_seconds = (
        full_cells_count * total_compile_per_cell
        + full_cells_count * candidate_n_full * total_exec_per_seed_block
    )

    evo_speedups = [r["evosax_speedup_C_vs_B"] for r in paired_records if "evosax_speedup_C_vs_B" in r]
    qd_speedups = [r["qdax_speedup_C_vs_B"] for r in paired_records if "qdax_speedup_C_vs_B" in r]

    analysis = {
        "schema_version": 2,
        "stage": args.stage,
        "num_cells_evaluated": len(my_cells),
        "seeds_per_cell": len(seeds),
        "total_seed_blocks": len(blocks),
        "pooled_contrasts": pooled_contrasts,
        "cell_level_power_sizing": {
            "evosax_median_required_n_90pct": int(np.median(evo_cell_req_n)) if evo_cell_req_n else 10,
            "evosax_p75_required_n_90pct": int(np.percentile(evo_cell_req_n, 75)) if evo_cell_req_n else 10,
            "qdax_median_required_n_90pct": int(np.median(qd_cell_req_n)) if qd_cell_req_n else 10,
            "qdax_p75_required_n_90pct": int(np.percentile(qd_cell_req_n, 75)) if qd_cell_req_n else 10,
        },
        "steady_state_speedup": {
            "evosax_native_vs_adapter_median_speedup": float(np.median(evo_speedups)) if evo_speedups else 1.0,
            "qdax_native_vs_adapter_median_speedup": float(np.median(qd_speedups)) if qd_speedups else 1.0,
        },
        "timing_by_condition": timing_summary,
        "budget_forecast_108_cells": {
            "compile_seconds_per_cell": total_compile_per_cell,
            "steady_bundle_seconds_per_seed_block": total_exec_per_seed_block,
            "projected_gpu_hours_at_n50": (
                full_cells_count * total_compile_per_cell + full_cells_count * 50 * total_exec_per_seed_block
            ) / 3600.0,
            "projected_gpu_hours_at_n100": projected_full_seconds / 3600.0,
        },
        "cell_summaries": cell_summaries,
    }

    with open(args.output / "pilot_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    summary_data: Dict[str, Any] = {
        "schema_version": 2,
        "stage": args.stage,
        "total_blocks": len(blocks),
        "total_condition_runs": len(run_csv_rows),
        "failed_conditions": failures,
        "failed_runs": failures,
        "success": bool(failures == 0),
        "blocks": [{"block": b["cell_id"], "conditions": b["order"]} for b in blocks],
        "pilot_analysis": analysis,
    }
    with open(args.output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n===========================================================================")
    print(f"  Stage {args.stage.upper()} Complete. Failures: {failures} | Blocks: {len(blocks)}")
    print(f"  Artifacts written to: {args.output}")
    print(f"    • summary.json & pilot_analysis.json")
    print(f"    • runs.csv ({len(run_csv_rows)} rows) & timings.csv ({len(timing_csv_rows)} rows)")
    print(f"===========================================================================\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
