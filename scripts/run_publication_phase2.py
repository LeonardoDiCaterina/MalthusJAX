#!/usr/bin/env python3
"""RL feasibility and optional preregistered task-scoped ranking collection.

Run from the checkout root with PYTHONPATH=src. --smoke uses the genuine
requested environment, P=4/G=1/H=2 and one seed unless explicitly supplied.
No environment fallback, dependency installation or GPU selection occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASE_CONDITIONS = ("native_ga", "evosax_open_es", "qdax_map_elites")
CONDITIONS = ("native_ga", "native_ga_cosine", "evosax_open_es", "qdax_map_elites")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def array_hash(value):
    import numpy as np

    arr = np.asarray(value)
    return hashlib.sha256(str((arr.shape, str(arr.dtype))).encode() + arr.tobytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True, help="New exclusive output directory")
    p.add_argument("--seeds", type=int, nargs="+", help="Distinct training-block seeds")
    p.add_argument("--conditions", choices=CONDITIONS, nargs="+")
    p.add_argument("--study-mode", choices=["feasibility", "ranking"], default="feasibility")
    p.add_argument("--environment", choices=("hopper", "gymnax-cartpole"), default="hopper")
    p.add_argument("--population", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--eval-episodes", type=int)
    return p


def build_environment(name, horizon):
    from malthusjax.core.fitness.composable.environments import GymnaxEnv, MuJoCoMJXEnv

    if name == "hopper":
        env = MuJoCoMJXEnv.create(env_name="hopper", max_steps=horizon)
        # The current factory catches Brax errors and returns Cartpole while
        # retaining the requested name. Never publish that fallback as Hopper.
        if env.backend_pipeline is None:
            raise RuntimeError("Hopper creation fell back to Cartpole; refusing substituted task")
        return env
    env = GymnaxEnv.create("CartPole-v1")
    return env.replace(env_params=env.env_params.replace(max_steps_in_episode=horizon))


def execute(engine, key, initial, population_config, condition):
    """Use existing engine preparation/run APIs and retain their final states.

    Universal adapters' public run_once discards final state. Its existing
    init/compiled-loop hooks are used here with precisely the same key split,
    permitting held-out evaluation without retraining or algorithm changes.
    """
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from malthusjax.core.fitness.base import dispatch_evaluate_population
    from malthusjax.core.genome.real_genome import RealGenome, RealPopulation
    from malthusjax.core.timing import block_until_ready

    started = time.perf_counter()
    if condition in ("native_ga", "native_ga_cosine"):
        local = engine.genetic_engine
        state = local.init_state(key)
        pop = RealPopulation.from_array(initial, population_config, RealGenome, axis=0)
        evaluated = dispatch_evaluate_population(local.evaluator, pop, key)
        idx = jnp.argmin(evaluated.fitness)
        state = replace(
            state,
            population=evaluated,
            best_genome=evaluated.genes[idx],
            best_fitness=evaluated.fitness[idx],
        )
        block_until_ready(state, effects=True)
        prepared = local.prepare_run(state)
        prepared_at = time.perf_counter()
        final, history, _ = local.run(state, compile=True, time_it=False, prepared_run=prepared)
        block_until_ready((final, history), effects=True)
        selected = final.best_genome.values
        curves = {
            k: getattr(history, k)
            for k in ("best_fitness", "mean_fitness", "std_fitness")
            if hasattr(history, k)
        }
        initial_calls = 2 * initial.shape[0]  # random init evaluated, then shared override
    else:
        local = engine.engine
        loop_key, init_key, _ = jax.random.split(key, 3)
        state = local.init_fn(local.framework_obj, init_key, local.framework_params, initial)
        block_until_ready((loop_key, state), effects=True)
        executable = local._build_jit_loop().lower(loop_key, state).compile()
        prepared_at = time.perf_counter()
        final, curves = executable(loop_key, state)
        block_until_ready((final, curves), effects=True)
        if condition == "evosax_open_es":
            selected = final.best_solution
        else:
            rep = final[0]
            selected = rep.genotypes[jnp.argmax(rep.fitnesses)]
        initial_calls = initial.shape[0]
    finished = time.perf_counter()
    return (
        final,
        curves,
        selected,
        {
            "initialization_and_preparation_seconds": prepared_at - started,
            "prepared_execution_seconds": finished - prepared_at,
            "train_end_to_end_seconds": finished - started,
            "timing_scope": "single prepared invocation, descriptive feasibility only",
            "initial_candidate_evaluations": initial_calls,
        },
    )


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from malthusjax.composer.engine_factory_v2 import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig
    from malthusjax.composer.strategies.core import QDAXStrategy
    from malthusjax.core.fitness.composable.base import (
        QDOutput,
        RLActionSummaryDescriptorFn,
        ScalarOutput,
    )
    from publication_rl import InstrumentedRLEvaluator as RLEvaluator
    from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
    from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
    from malthusjax.core.timing import block_until_ready

    # Explicit observation accumulation makes this an action descriptor, not
    # the existing helper's fallback to the first policy weights.
    class MeanActionDescriptor(RLActionSummaryDescriptorFn):
        def observe_step(self, env_state, obs, action, reward):
            if args.environment == "gymnax-cartpole":
                action = jax.nn.one_hot(action, 2)
            return {"action": action}

        def compute_rl(self, genome, final_state, step_infos):
            actions = step_infos["action"]  # (evaluation episodes, time, action dimension)
            mask = step_infos["valid_step"][..., None]
            axes = tuple(range(actions.ndim - 1))
            means = jnp.sum(jnp.where(mask, actions, 0.0), axis=axes) / jnp.maximum(
                jnp.sum(mask, axis=axes), 1
            )
            features = jnp.pad(means, (0, max(0, 2 - means.shape[0])))[:2]
            if args.environment == "gymnax-cartpole":
                return features  # empirical frequencies of the two discrete actions
            return jnp.clip((features + 1.0) / 2.0, 0.0, 1.0)

    env = build_environment(args.environment, args.horizon)
    write_json(
        args.output / "environment.json",
        {
            "requested": args.environment,
            "resolved_wrapper": type(env).__name__,
            "resolved_pipeline": type(getattr(env, "backend_pipeline", None)).__name__,
            "hopper_fallback_rejected": args.environment == "hopper",
            "observation_dimension": env.obs_dim,
            "action_dimension": env.action_dim,
            "rollout_horizon": args.horizon,
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
    )
    interpreter = MLPInterpreter(
        input_dim=env.obs_dim,
        output_dim=env.action_dim,
        hidden=(4,) if args.smoke else (64, 64),
        activation="tanh",
    )
    scalar = RLEvaluator(
        env=env,
        interpreter=interpreter,
        output=ScalarOutput(maximize=True),
        max_steps=args.horizon,
        num_eval_envs=1,
    )
    qd = scalar.replace(
        output=QDOutput(
            scalar_output=ScalarOutput(maximize=True),
            descriptor_fn=MeanActionDescriptor(num_descriptors=2, bounds=(-1.0, 1.0)),
        )
    )
    genome_config = RealGenomeConfig(shape=(interpreter.num_params,), bounds=(-1.0, 1.0))
    centroid_count = 4 if args.smoke else 64
    rows = []
    for seed in args.seeds:
        # Domain-separated keys: common initialization, centroid and held-out
        # episode seeds are distinct from algorithm execution streams.
        base = jax.random.PRNGKey(seed)
        init_key = jax.random.fold_in(base, 1101)
        initial = jax.random.uniform(
            init_key, (args.population, interpreter.num_params), minval=-1.0, maxval=1.0
        )
        centroids = jax.random.uniform(jax.random.fold_in(base, 1102), (centroid_count, 2))
        initial_digest = array_hash(initial)
        block_dir = args.output / f"seed_{seed}"
        block_dir.mkdir()
        np.save(block_dir / "initial_population.npy", np.asarray(initial), allow_pickle=False)
        np.save(block_dir / "centroids.npy", np.asarray(centroids), allow_pickle=False)
        for condition in args.conditions:
            cond_dir = block_dir / condition
            cond_dir.mkdir()
            row = {
                "seed": seed,
                "condition": condition,
                "study_mode": args.study_mode,
                "configuration": {
                    "population": args.population,
                    "generations": args.generations,
                    "horizon": args.horizon,
                    "heldout_episodes": args.eval_episodes,
                },
                "status": "failure",
                "initial_population_sha256": initial_digest,
                "centroids_sha256": array_hash(centroids),
                "initialization_semantics": (
                    "shared candidates evaluated; Open_ES distribution starts at their mean"
                    if condition == "evosax_open_es"
                    else "shared initial candidate array"
                ),
                "objective_encoding": "internal fitness=-episodic_return; lower is better",
                "interpretation": "task-scoped ranking collection; inference is separate"
                if args.study_mode == "ranking"
                else "feasibility only; no algorithm superiority claim",
            }
            write_json(cond_dir / "result.json", row)
            try:
                config_kwargs = dict(
                    fitness=qd if condition == "qdax_map_elites" else scalar,
                    genome_length=interpreter.num_params,
                    pop_size=args.population,
                    generations=args.generations,
                    bounds=(-1.0, 1.0),
                    maximize=False,
                    seeds=(seed,),
                    initial_population=initial,
                )
                if condition in ("native_ga", "native_ga_cosine"):
                    from malthusjax.engine.schedules import (
                        ScheduleType,
                        TrackBest,
                        compute_scheduled_strength,
                    )
                    from malthusjax.operators.mutation.real import GaussianMutation

                    cosine = condition == "native_ga_cosine"
                    mutation = GaussianMutation(
                        mutation_rate=0.2,
                        mutation_strength=0.2,
                        schedule_type=ScheduleType.COSINE_ANNEAL
                        if cosine
                        else ScheduleType.CONSTANT,
                        final_strength=0.01 if cosine else 0.2,
                        max_generations=args.generations,
                    )
                    row["mutation_protocol"] = {
                        "type": "cosine" if cosine else "constant",
                        "rate": 0.2,
                        "initial_strength": 0.2,
                        "final_strength": 0.01 if cosine else 0.2,
                        "generation_strengths": [
                            float(
                                compute_scheduled_strength(
                                    mutation.schedule_type,
                                    g,
                                    args.generations,
                                    mutation.mutation_strength,
                                    mutation.final_strength,
                                )
                            )
                            for g in range(args.generations)
                        ],
                    }
                    config_kwargs.update(
                        backend="malthusjax",
                        elitism=0,
                        selection=f"tournament:num_selections={args.population},tournament_size=2",
                        crossover="simulated_binary:eta=15.0",
                        mutation=mutation,
                    )
                    if args.study_mode == "ranking":
                        # Historical best policy, not LIGHT's final-population approximation.
                        config_kwargs["track_best"] = TrackBest.FULL
                elif condition == "evosax_open_es":
                    config_kwargs.update(backend="evosax", evosax_strategy="Open_ES")
                else:
                    config_kwargs.update(
                        backend="qdax",
                        strategy=QDAXStrategy(
                            centroids=centroids,
                            num_centroids=centroid_count,
                            num_descriptors=2,
                            mutation_sigma=0.1,
                        ),
                    )
                config = ExperimentConfig.from_quick_run_kwargs(**config_kwargs)
                engine = EngineFactory.build(config)
                supplied = engine.initial_population
                if array_hash(supplied) != initial_digest:
                    raise RuntimeError("Engine initial candidate array differs from shared block")
                final, curves, selected, cost = execute(
                    engine, jax.random.fold_in(base, 2201), initial, genome_config, condition
                )
                if not curves:
                    raise RuntimeError("Engine returned no history metrics")
                for metric, values in curves.items():
                    values = np.asarray(values)
                    if values.shape != (args.generations,):
                        raise RuntimeError(f"Unexpected {metric} history shape: {values.shape}")
                    if metric in ("best_fitness", "mean_fitness") and not np.all(
                        np.isfinite(values)
                    ):
                        raise RuntimeError(f"Nonfinite {metric} history")
                selected = jnp.asarray(selected)
                if selected.shape != (interpreter.num_params,) or not bool(
                    jnp.all(jnp.isfinite(selected))
                ):
                    raise RuntimeError("Selected policy has invalid shape or nonfinite parameters")
                np.save(cond_dir / "selected_policy.npy", np.asarray(selected), allow_pickle=False)
                # Store every state array leaf plus its path, without unsafe pickle.
                paths, _ = jax.tree_util.tree_flatten_with_path(final)
                arrays = {
                    f"leaf_{i}": np.asarray(v)
                    for i, (_, v) in enumerate(paths)
                    if hasattr(v, "dtype")
                }
                np.savez_compressed(cond_dir / "final_state.npz", **arrays)
                write_json(cond_dir / "state_paths.json", [str(path) for path, _ in paths])
                history = jsonable(curves)
                write_json(cond_dir / "history.json", history)
                # Holdout uses the selected final policy only, never selects on
                # the evaluation episodes, and retains each episode separately.
                holdout = []
                diagnostics = []
                for episode in range(args.eval_episodes):
                    eval_key = jax.random.fold_in(jax.random.fold_in(base, 3301), episode)
                    encoded, rollout_info = scalar.evaluate(
                        RealGenome(values=selected), RealGenome(values=selected), eval_key
                    )
                    block_until_ready(encoded, effects=True)
                    holdout.append(-float(encoded))
                    diagnostics.append(jsonable(rollout_info))
                if not all(math.isfinite(x) for x in holdout):
                    raise RuntimeError("Nonfinite held-out return")
                generational_evals = args.population * args.generations
                row.update(
                    status="success",
                    cost=cost,
                    history_file="history.json",
                    heldout_episode_returns=holdout,
                    heldout_rollout_diagnostics=diagnostics,
                    heldout_mean_return=float(np.mean(holdout)),
                    selected_policy_rule="best training objective in final returned state/archive",
                    evaluations={
                        "generational_candidates": generational_evals,
                        "initialization_candidates": cost["initial_candidate_evaluations"],
                        "heldout_episodes": args.eval_episodes,
                        "training_rollouts_per_candidate": 1,
                        "actual_terminated_environment_steps": None,
                        "actual_steps_unavailable_reason": "aggregate training transitions not accumulated; per-policy heldout and retained state diagnostics available",
                        "fixed_scan_training_steps": (
                            generational_evals + cost["initial_candidate_evaluations"]
                        )
                        * args.horizon,
                    },
                    descriptor=(
                        (
                            "two discrete-action frequencies; valid pretermination steps only"
                            if args.environment == "gymnax-cartpole"
                            else "first two mean action components, normalized/clipped from [-1,1]; valid pretermination steps only"
                        )
                        if condition == "qdax_map_elites"
                        else None
                    ),
                )
                if condition == "qdax_map_elites":
                    from malthusjax.core.fitness.qd.metrics import archive_metrics

                    row["archive_metrics"] = jsonable(archive_metrics(final[0].fitnesses))
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}")
                (cond_dir / "traceback.txt").write_text(traceback.format_exc())
            write_json(cond_dir / "result.json", row)
            rows.append(row)
            print(f"seed={seed} condition={condition} status={row['status']}", flush=True)
            # Separate compiled objects for tiny runs; this script makes no
            # architecture-overhead estimate from condition wall-clock costs.
            jax.clear_caches()
    write_json(
        args.output / "summary.json",
        {
            "attempted": len(rows),
            "successful": sum(r["status"] == "success" for r in rows),
            "failed": sum(r["status"] != "success" for r in rows),
            "runs": rows,
            "publication_readiness": "descriptive runner; workload/protocol and supported initialization semantics need review",
        },
    )
    return 0 if all(r["status"] == "success" for r in rows) else 1


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.conditions = args.conditions or list(
        CONDITIONS if args.study_mode == "ranking" else BASE_CONDITIONS
    )
    if not args.smoke and not args.seeds:
        p.error("Non-smoke runs require an explicit --seeds list")
    args.seeds = args.seeds or [17]
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s > 2**32 - 1 for s in args.seeds):
        p.error("Seeds must be distinct unsigned 32-bit integers")
    args.population = args.population if args.population is not None else (4 if args.smoke else 128)
    args.generations = (
        args.generations if args.generations is not None else (1 if args.smoke else 200)
    )
    args.horizon = args.horizon if args.horizon is not None else (2 if args.smoke else 1000)
    args.eval_episodes = (
        args.eval_episodes if args.eval_episodes is not None else (1 if args.smoke else 20)
    )
    if min(args.population, args.generations, args.horizon, args.eval_episodes) <= 0:
        p.error("Workload dimensions must be positive")
    if args.population < 4 or args.population % 2:
        p.error("Population must be even and >=4 for the selected GA/ES panel")
    if len(set(args.conditions)) != len(args.conditions):
        p.error("Conditions must not repeat")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    packages = {}
    for package in (
        "jax",
        "jaxlib",
        "numpy",
        "evosax",
        "qdax",
        "mujoco",
        "mujoco-mjx",
        "brax",
        "gymnax",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "schema_version": 1,
        "phase": 2,
        "arguments": vars(args),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rollout_instrumentation_sha256": hashlib.sha256(
            Path(__file__).with_name("publication_rl.py").read_bytes()
        ).hexdigest(),
        "device_environment": {
            k: os.environ.get(k) for k in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")
        },
        "claim": "Hopper configuration ranking by held-out mean return; task/budget scoped"
        if args.study_mode == "ranking"
        else "quantified feasibility, not benchmark/superiority",
        "primary_metric": "mean return across fresh held-out episodes per independent training run",
        "analysis_unit": "independent training seed block, not episodes or policies",
        "limitations": [
            "Open_ES shares evaluated initial candidates but starts its distribution at their mean",
            "Rollout diagnostics retain invalid/reset/termination counts; inspect them before interpreting task results",
            "Fixed scanned work and per-evaluation live/invalid diagnostics are distinct",
            "QD action descriptors are protocol candidates, not preapproved scientific endpoints",
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    try:
        return run(args)
    except Exception as exc:
        write_json(
            args.output / "failure.json",
            {"status": "failure", "error": f"{type(exc).__name__}: {exc}"},
        )
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        print(f"Phase 2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
