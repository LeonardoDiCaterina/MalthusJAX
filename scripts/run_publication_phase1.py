#!/usr/bin/env python3
"""Scoped Phase 1 collection; no statistical conclusion or cluster submission.

Run from checkout root with PYTHONPATH=src. --smoke forces CPU and tiny shapes.
Every seed is a fresh block; immutable input arrays are shared across conditions.
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


def digest(array):
    import numpy as np

    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def collect_prepared_timing(options, seed, repetitions):
    """Capture actual production initialization, then retain and invoke its AOT executable.

    TimingCapture is an existing observation hook. The local patch stops the
    adapter before compilation/execution; no production source is modified.
    """
    import logging
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp
    import numpy as np

    from malthusjax.composer.engine_factory_v2 import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig
    from malthusjax.core.timing import TimingCapture, block_until_ready
    from malthusjax.core.timing_records import input_signature

    class Initialized(Exception):
        pass

    captured = {}

    def stop_after_initialization(self, inputs, engine):
        block_until_ready(inputs, effects=True)
        captured.update(inputs=inputs, engine=engine)
        raise Initialized()

    adapter = EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**options))
    start = time.perf_counter()
    with patch.object(TimingCapture, "initialized", stop_after_initialization):
        try:
            adapter.run_once(jax.random.key(seed))
        except Initialized:
            pass
    if not captured:
        raise RuntimeError("Adapter does not expose supported initialization capture")
    initialization_seconds = time.perf_counter() - start
    engine, inputs = captured["engine"], captured["inputs"]
    block_until_ready(inputs, effects=True)
    start = time.perf_counter()
    if hasattr(engine, "prepare_run"):
        executable = engine.prepare_run(inputs).executable
        arguments = (inputs,)
    elif hasattr(engine, "_build_jit_loop") and not engine.use_python_loop:
        arguments = tuple(inputs)
        executable = engine._build_jit_loop().lower(*arguments).compile()
    else:
        raise RuntimeError("No supported AOT executable path")
    preparation_seconds = time.perf_counter() - start
    context = {"phase": "idle"}
    compile_events = []

    class Observe(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if context["phase"] == "execution" and any(
                token in message.lower() for token in ("compiling", "tracing", "compilation")
            ):
                compile_events.append(message)

    observer = Observe()
    root = logging.getLogger()
    root.addHandler(observer)
    trials = []
    reference = None
    try:
        with jax.log_compiles(True):
            for trial in range(2 + repetitions):
                # Native executable donates state; reset independent device buffers
                # outside timing so every trial starts at exactly the same state.
                trial_args = jax.tree_util.tree_map(
                    lambda x: jnp.array(x, copy=True) if isinstance(x, jax.Array) else x, arguments
                )
                block_until_ready(trial_args, effects=True)
                phase = (
                    "prepared_first_execution"
                    if trial == 0
                    else "warmup"
                    if trial == 1
                    else "prepared_execution_repeat"
                )
                context["phase"] = "execution"
                start = time.perf_counter()
                output = executable(*trial_args)
                block_until_ready(output, effects=True)
                seconds = time.perf_counter() - start
                context["phase"] = "validation"
                leaves, structure = jax.tree_util.tree_flatten(output)
                host = [
                    np.asarray(jax.random.key_data(x))
                    if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                    else np.asarray(x)
                    for x in leaves
                ]
                if reference is None:
                    reference = (structure, host)
                agrees = structure == reference[0] and all(
                    np.array_equal(x, y, equal_nan=True) for x, y in zip(host, reference[1])
                )
                trials.append(
                    {
                        "trial_id": trial,
                        "phase": phase,
                        "seconds": seconds,
                        "evaluations_per_second": options["pop_size"]
                        * options["generations"]
                        / seconds,
                        "same_initialized_state_output_agreement": agrees,
                    }
                )
                context["phase"] = "idle"
    finally:
        root.removeHandler(observer)
    return {
        "schema_version": 1,
        "target_phase": "prepared_execution_repeat",
        "status": "valid"
        if not compile_events and all(t["same_initialized_state_output_agreement"] for t in trials)
        else "invalid",
        "scope": "retained AOT scan output; input-copy/reset and host validation excluded",
        "initialization_seconds": initialization_seconds,
        "preparation_seconds": preparation_seconds,
        "input_signature": input_signature(arguments),
        "repetitions": repetitions,
        "warmup_calls": 1,
        "prepared_first_calls": 1,
        "observed_execution_compilation_log_events": compile_events,
        "compilation_observation": "JAX compilation/tracing log messages during timed execution; retained compiled callable used for every trial",
        "trials": trials,
        "evaluations_per_scan": options["pop_size"] * options["generations"],
        "reset": "copy exact initialized state before every trial; copies synchronized outside timer",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Also collect retained-executable first/warm/repeated calls",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--library", choices=["evosax", "qdax", "both"], default="both")
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=["sphere", "rastrigin", "rosenbrock"],
        default=["sphere", "rastrigin", "rosenbrock"],
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--populations", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--generations", type=int, default=500)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s > 2**31 - 1 for s in args.seeds):
        parser.error("seeds must be unique nonnegative int32 values")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.smoke:
        args.repetitions = min(args.repetitions, 2)
    if args.smoke and len(args.seeds) > 3:
        parser.error("--smoke allows at most three seed blocks")
    if args.smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        args.functions, args.dims, args.populations, args.generations = ["sphere"], [4], [16], 2
    if min(args.dims) < 2 or min(args.populations) < 12 or args.generations < 1:
        parser.error("require dimensions >=2, populations >=12, generations >=1")
    args.output.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp
    import numpy as np

    from malthusjax.benchmarking.results import standard_json
    from malthusjax.composer import Composer
    from malthusjax.composer.strategies.core import MapElitesStrategy, QDAXStrategy

    if args.library in ("both", "qdax"):
        from qdax.core.emitters.standard_emitters import MixingEmitter

        from malthusjax.core.genome.real_genome import RealGenomeConfig
        from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter

    def installed_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def write(path, value):
        path.write_text(json.dumps(standard_json(value), indent=2, allow_nan=False) + "\n")

    manifest = vars(args).copy()
    manifest["output"] = str(args.output)
    manifest.update(
        schema_version=1,
        python=sys.version,
        platform=platform.platform(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        devices=[str(d) for d in jax.devices()],
        working_tree_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        versions={
            name: installed_version(name) for name in ["jax", "jaxlib", "evosax", "qdax", "numpy"]
        },
        collection_scope="quality plus optional retained AOT timing collection; process-isolated publication sessions are external",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_initial_population=True,
        fresh_initialization_per_seed=True,
        bounds=[-5.0, 5.0],
        confirmation_policy="UNFROZEN: no equivalence inference performed",
    )
    write(args.output / "manifest.json", manifest)
    failures = 0
    blocks = []
    for function in args.functions:
        for dim in args.dims:
            for pop in args.populations:
                for seed in args.seeds:
                    directory = args.output / f"{function}_d{dim}_p{pop}_s{seed}"
                    directory.mkdir()
                    # Distinct, explicit key namespaces; same arrays for every condition.
                    key = jax.random.PRNGKey(seed)
                    init_key = jax.random.fold_in(key, 101)
                    cent_key = jax.random.fold_in(key, 102)
                    initial = jax.random.uniform(init_key, (pop, dim), minval=-5.0, maxval=5.0)
                    count = 9 if args.smoke else 100
                    # Uniform random sites, shared exactly (not CVT); archive definition explicit.
                    centroids = jax.random.uniform(cent_key, (count, 2))
                    np.save(directory / "initial_population.npy", np.asarray(initial))
                    np.save(directory / "centroids.npy", np.asarray(centroids))
                    pipelines = {}
                    if args.library in ("both", "evosax"):
                        if max(2, int(pop / 6)) != max(1, int(pop * 0.16666666)):
                            raise ValueError(
                                "H1 native elite_k and upstream elite_ratio yield different counts; choose a matching population"
                            )
                        pipelines["evosax_adapter"] = dict(
                            backend="evosax",
                            evosax_strategy="SimpleGA",
                            strategy_params={"crossover_rate": 0.3, "elite_ratio": 0.16666666},
                        )
                        for name, cross, mut in [
                            ("evosax_standard", "evosax_uniform_crossover", "evosax_gaussian"),
                            ("evosax_batched", "batched_evosax_uniform", "batched_evosax_gaussian"),
                        ]:
                            pipelines[name] = dict(
                                backend="malthusjax",
                                selection=f"evosax_mimic_selection:num_selections={pop},elite_k={max(2, int(pop / 6))}",
                                crossover=f"{cross}:crossover_rate=0.3",
                                mutation=f"{mut}:mutation_strength=1.0",
                                elitism=0,
                            )
                    if args.library in ("both", "qdax"):

                        def mutation(x, rng):
                            return x + 0.1 * jax.random.normal(rng, x.shape)

                        def variation(x, y, rng):
                            return jnp.where(jax.random.bernoulli(rng, 0.5, x.shape), x, y)

                        upstream_emitter = MixingEmitter(
                            mutation_fn=mutation,
                            variation_fn=variation,
                            variation_percentage=0.5,
                            batch_size=pop,
                        )
                        native_emitter = QDAXReplicaMixingEmitter(
                            mutation_fn=mutation,
                            variation_fn=variation,
                            variation_percentage=0.5,
                            _batch_size=pop,
                            genome_config=RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0)),
                        )
                        pipelines["qdax_adapter"] = dict(
                            backend="qdax",
                            strategy=QDAXStrategy(
                                emitter=upstream_emitter, centroids=centroids, num_centroids=count
                            ),
                        )
                        pipelines["qdax_native"] = dict(
                            backend="malthusjax",
                            strategy=MapElitesStrategy(
                                emitter=native_emitter, centroids=centroids, num_centroids=count
                            ),
                        )
                    # Deterministic counterbalanced order, retained, never select favorable timing.
                    names = list(pipelines)
                    shift = seed % len(names)
                    names = names[shift:] + names[:shift]
                    block = dict(
                        function=function,
                        dim=dim,
                        population=pop,
                        generations=args.generations,
                        seed=seed,
                        order=names,
                        initial_population_hash=digest(initial),
                        centroids_hash=digest(centroids),
                        centroid_method="uniform random sites",
                        num_centroids=count,
                        expected_objective_calls_per_condition=pop * (args.generations + 1),
                        objective_call_count_status="scheduled counts; run-reported values retained separately",
                        descriptor_contract="ensure_qd_evaluator / GenotypeBoxDescriptorFn: first 2 genotype coordinates mapped from bounds [-5,5] to [0,1]; source src/malthusjax/core/fitness/qd/evaluator.py and composable/base.py; review before publication",
                        key_schedule={
                            "base_seed": seed,
                            "initialization_fold_in": 101,
                            "centroids_fold_in": 102,
                            "algorithm_seed": seed,
                            "evaluator_instance_seed": 42,
                        },
                        elite_k_native=max(2, int(pop / 6)),
                        elite_k_upstream=max(1, int(pop * 0.16666666)),
                        qd_variation_percentage=0.5,
                        qd_mutation_sigma=0.1,
                        qd_variation="uniform per-gene parent selection, probability .5",
                        native_variation_imports="EvoSAX wrapper imports; QD native replica with shared supplied mutation/variation functions",
                        runs={},
                    )
                    for name in names:
                        start = time.perf_counter()
                        try:
                            result = Composer.create_default().quick_run(
                                **pipelines[name],
                                fitness=f"bbob:fn_name={function},num_dims={dim},maximize=false,seed=42",
                                pop_size=pop,
                                generations=args.generations,
                                genome_length=dim,
                                bounds=(-5.0, 5.0),
                                maximize=False,
                                initial_population=initial,
                                seeds=(seed,),
                            )
                            run = result.runs[0]
                            data = run.to_dict()
                            data["evaluation_accounting"] = {
                                "reported_generational": run.metrics.get("total_evaluations"),
                                "source_derived_discarded_setup_evaluations": pop
                                if name in ("evosax_standard", "evosax_batched")
                                else 0,
                                "source_derived_collection_objective_calls": pop
                                * (args.generations + 1)
                                + (pop if name in ("evosax_standard", "evosax_batched") else 0),
                                "initialization": pop,
                                "scheduled_generational": pop * args.generations,
                                "scheduled_total_including_initialization": pop
                                * (args.generations + 1),
                                "timing_repetitions_excluded": True,
                                "count_basis": "engine contract: one initial population plus one population per generation; initialization/compilation preparation may invoke additional work outside search budget",
                            }
                            if run.status != "success" or not np.isfinite(
                                run.metrics.get("best_fitness", np.nan)
                            ):
                                failures += 1
                            block["runs"][name] = data
                        except Exception as exc:
                            failures += 1
                            block["runs"][name] = {
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        if args.timing:
                            try:
                                timing_options = dict(
                                    pipelines[name],
                                    fitness=f"bbob:fn_name={function},num_dims={dim},maximize=false,seed=42",
                                    pop_size=pop,
                                    generations=args.generations,
                                    genome_length=dim,
                                    bounds=(-5.0, 5.0),
                                    maximize=False,
                                    initial_population=initial,
                                )
                                dedicated = collect_prepared_timing(
                                    timing_options, seed, args.repetitions
                                )
                                block["runs"][name]["dedicated_timing"] = dedicated
                                if dedicated["status"] != "valid":
                                    failures += 1
                            except Exception as exc:
                                failures += 1
                                block["runs"][name]["dedicated_timing"] = {
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                        block["runs"][name]["collection_wall_seconds"] = time.perf_counter() - start
                        block["runs"][name]["input_population_hash"] = digest(initial)
                        write(directory / f"{name}.json", block["runs"][name])
                        print(
                            f"{directory.name} {name}: {block['runs'][name]['status']}", flush=True
                        )
                        jax.clear_caches()
                    write(directory / "block.json", block)
                    blocks.append({"block": directory.name, "conditions": names})
    write(
        args.output / "summary.json",
        {
            "blocks": blocks,
            "failed_conditions": failures,
            "publication_ready": False,
            "limits": [
                "No approved margins/sample-size policy",
                "Existing descriptor semantics and algorithm correspondence require scientific review",
                "Shared input hashes establish supplied inputs, not equality of all internal states",
                "Collection wall time includes preparation and is not steady-state performance",
            ],
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
