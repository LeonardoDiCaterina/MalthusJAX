#!/usr/bin/env python3
"""PT5 diagnostic: actual adapters, raw trials, phase logs, and seeded references."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import tomllib

import malthusjax
from malthusjax.benchmarking.io import write_experiment_artifacts
from malthusjax.benchmarking.results import ExperimentResult, RunResult
from malthusjax.core.timing import TimingCapture, block_until_ready
from malthusjax.core.timing_records import (
    TimingPolicy,
    TimingRecord,
    environment_identity,
    json_safe,
    paired_timing_summary,
)

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("malthusjax_native", "qdax_baseline")


def validate_config(cfg):
    for name in ("dimensions", "population", "generations", "centroids", "repetitions"):
        if type(cfg.get(name)) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["dimensions"] < 2:
        raise ValueError("Two descriptors require at least two dimensions")
    TimingPolicy("prepared_execution_repeat", cfg["preconditioning"], cfg["repetitions"])
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if len(cfg["bounds"]) != 2 or not cfg["bounds"][0] < cfg["bounds"][1]:
        raise ValueError("Bounds must be increasing")


def schedule(cfg):
    """Freeze execution order before any durations are observed."""
    result = []
    for seed_order, seed in enumerate(cfg["seeds"]):
        for trial in range(1 + cfg["preconditioning"] + cfg["repetitions"]):
            order = BACKENDS if (seed_order + trial) % 2 == 0 else BACKENDS[::-1]
            for backend in order:
                result.append(
                    {
                        "seed": seed,
                        "seed_order": seed_order,
                        "trial_id": trial,
                        "backend": backend,
                        "run_order": len(result),
                    }
                )
    return result


def build_adapters(cfg):
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.composer.composer import Composer
    from malthusjax.composer.engine_factory_v2 import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig
    from malthusjax.composer.strategies.core import MapElitesStrategy
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.operators.emitters.genetic import GeneticMutationEmitter

    Composer.create_default()  # Register the production backend providers.
    bounds = tuple(cfg["bounds"])
    initial = jax.random.uniform(
        jax.random.PRNGKey(cfg["initial_population_seed"]),
        (cfg["population"], cfg["dimensions"]),
        minval=bounds[0],
        maxval=bounds[1],
    )
    block_until_ready(initial, effects=True)
    emitter = GeneticMutationEmitter(
        _batch_size=cfg["population"],
        mutation=OperatorCatalog().get(
            f"gaussian:mutation_rate=1.0,mutation_strength={cfg['mutation_sigma']}"
        ),
        genome_config=RealGenomeConfig(shape=(cfg["dimensions"],), bounds=bounds),
    )
    pipelines = {
        BACKENDS[0]: {
            "strategy": MapElitesStrategy(
                emitter=emitter, num_descriptors=2, num_centroids=cfg["centroids"]
            )
        },
        BACKENDS[1]: {
            "backend": "qdax",
            "qdax_strategy": "MAPElites",
            "qdax_num_descriptors": 2,
            "qdax_num_centroids": cfg["centroids"],
            "qdax_mutation_sigma": cfg["mutation_sigma"],
        },
    }
    shared = dict(
        fitness=f"rastrigin:dim={cfg['dimensions']}",
        pop_size=cfg["population"],
        generations=cfg["generations"],
        genome_length=cfg["dimensions"],
        maximize=False,
        bounds=bounds,
        initial_population=initial,
    )
    return {
        name: EngineFactory.build(ExperimentConfig.from_quick_run_kwargs(**shared, **options))
        for name, options in pipelines.items()
    }


class PhaseLog(logging.Handler):
    def __init__(self, path, context, events):
        super().__init__()
        self.stream = path.open("x")
        self.context, self.events = context, events
        self.count = 0
        self.compilation_signals = 0

    def emit(self, record):
        message = record.getMessage()
        self.count += 1
        is_compilation = any(
            token in message.lower() for token in ("compiling", "compilation", "tracing")
        )
        self.compilation_signals += int(is_compilation)
        row = {
            **self.context,
            "event": "jax_log",
            "logger": record.name,
            "message": message,
            "timestamp": time.perf_counter(),
        }
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        if (
            self.context.get("purpose") == "measurement"
            and self.context.get("phase") == "execution"
            and is_compilation
        ):
            self.events.append(row)

    def close(self):
        self.stream.close()
        super().close()


@contextmanager
def observe_phases(context, events):
    def wrap(name, phase):
        original = getattr(TimingCapture, name)

        def wrapped(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            context["phase"] = phase
            events.append({**context, "event": name, "timestamp": time.perf_counter()})
            return value

        return wrapped

    with (
        patch.object(TimingCapture, "initialized", wrap("initialized", "preparation")),
        patch.object(TimingCapture, "prepared", wrap("prepared", "execution")),
        patch.object(TimingCapture, "executed", wrap("executed", "host_postprocessing")),
    ):
        yield


def reference_run(name, adapter, key):
    if name == BACKENDS[1]:
        return adapter.run_once(key, compile=False)
    # Same initialized state/algorithm, ordinary JIT invocation without the
    # retained prepared executable or engine timer. Preparation remains outside
    # this reference invocation; none of its timings enter the comparison.
    engine_type = type(adapter.engine)
    original = engine_type.run

    def ordinary_run(self, *args, **kwargs):
        kwargs.update(prepared_run=None, time_it=False, compile=True)
        return original(self, *args, **kwargs)

    with patch.object(engine_type, "run", ordinary_run):
        return adapter.run_once(key)


def objective_agreement(left, right):
    """Compare reported numeric summary and full histories within one backend."""
    try:
        for field in ("summary", "history"):
            ll, lt = jax.tree_util.tree_flatten(left[field])
            rl, rt = jax.tree_util.tree_flatten(right[field])
            if lt != rt:
                return False
            for a, b in zip(ll, rl):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        return True
    except (AssertionError, TypeError, ValueError):
        return False


def run_validation(cfg, output, *, expected_revision, expected_backend, expected_device):
    validate_config(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context, events, contamination = {"phase": "setup"}, [], []
    handler = PhaseLog(output / "compilation.jsonl", context, contamination)
    logging.getLogger().addHandler(handler)
    report = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "config": cfg,
        "schedule": schedule(cfg),
        "raw_trials": [],
        "references": [],
        "limitations": [
            "No cross-backend trajectory equality assumed.",
            "Device contention cannot be excluded by this diagnostic.",
            "EvoSAX, TensorNEAT, MO and island integrations are outside this GPU workload.",
            "Prepared repeats do not establish a general steady-state throughput claim.",
        ],
    }
    try:
        identity = environment_identity()
        report["environment"] = identity
        report["scheduler"] = {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "LD_LIBRARY_PATH",
            )
        }
        report["packages"] = {p: importlib.metadata.version(p) for p in ("qdax", "jax", "jaxlib")}
        report["accelerator_packages"] = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name", "").lower().startswith(("jax", "nvidia", "cuda"))
        }
        if identity["revision"] != expected_revision:
            raise ValueError("Source revision differs from --expect-revision")
        if Path(malthusjax.__file__).resolve().parent != ROOT / "src/malthusjax":
            raise ValueError("Imported malthusjax is not the entrypoint's checkout")
        if identity["backend"] != expected_backend:
            raise ValueError(f"Expected backend {expected_backend}, got {identity['backend']}")
        if (
            len(jax.devices()) != 1
            or expected_device.lower() not in jax.devices()[0].device_kind.lower()
        ):
            raise ValueError(f"Expected one {expected_device} device, got {jax.devices()}")
        try:
            report["nvidia_smi"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
                    "--format=csv",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            report["nvidia_smi"] = str(exc)
        (output / "manifest.json").write_text(json.dumps(json_safe(report), indent=2))
        with jax.log_compiles(True), observe_phases(context, events):
            adapters = build_adapters(cfg)
            for item in report["schedule"]:
                context.update(item, phase="initialization", purpose="measurement")
                start = time.perf_counter()
                result = adapters[item["backend"]].run_once(jax.random.PRNGKey(item["seed"]))
                duration = time.perf_counter() - start
                record = TimingRecord.from_producer(
                    result.get("timings"),
                    result.get("timing_metadata", {}),
                    seed=item["seed"],
                    order=item["run_order"],
                    environment=identity,
                    duration=duration,
                ).to_dict()
                row = {**item, "result": result, "timing_record": record}
                report["raw_trials"].append(row)
                with (output / "raw_trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(json_safe(row)) + "\n")
            # References deliberately follow every measurement, preserving cold first-use evidence.
            for name, adapter in adapters.items():
                for seed in cfg["seeds"]:
                    context.update(
                        backend=name,
                        seed=seed,
                        trial_id=None,
                        run_order=None,
                        phase="reference",
                        purpose="reference",
                    )
                    result = reference_run(name, adapter, jax.random.PRNGKey(seed))
                    rows = [
                        r
                        for r in report["raw_trials"]
                        if r["backend"] == name and r["seed"] == seed
                    ]
                    agreements = [objective_agreement(r["result"], result) for r in rows]
                    report["references"].append(
                        {
                            "backend": name,
                            "seed": seed,
                            "agreement_by_trial": agreements,
                            "result": result,
                        }
                    )
        for row in report["raw_trials"]:
            reference = next(
                r
                for r in report["references"]
                if r["backend"] == row["backend"] and r["seed"] == row["seed"]
            )
            first = next(
                r
                for r in report["raw_trials"]
                if r["backend"] == row["backend"]
                and r["seed"] == row["seed"]
                and r["trial_id"] == 0
            )
            observed = [
                e
                for e in events
                if e.get("purpose") == "measurement" and e.get("run_order") == row["run_order"]
            ]
            problems = []
            if [e["event"] for e in observed] != ["initialized", "prepared", "executed"]:
                problems.append("Missing or inconsistent phase events")
            if any(e.get("run_order") == row["run_order"] for e in contamination):
                problems.append("Tracing or compilation observed inside execution boundary")
            if not reference["agreement_by_trial"][row["trial_id"]]:
                problems.append("Timed output differs from the same-seed reference")
            if row["timing_record"]["signature"] != first["timing_record"]["signature"]:
                problems.append("Executable/input signature changed between reset-state trials")
            row["validation_problems"] = problems
            if problems:
                row["timing_record"].update(status="unavailable", reason="; ".join(problems))
        experiments = {}
        for name in BACKENDS:
            runs = []
            for seed in cfg["seeds"]:
                rows = [
                    r for r in report["raw_trials"] if r["backend"] == name and r["seed"] == seed
                ]
                first = rows[0]
                record = deepcopy(first["timing_record"])
                record.update(
                    target_phase="prepared_execution_repeat",
                    policy=TimingPolicy(
                        "prepared_execution_repeat", cfg["preconditioning"], cfg["repetitions"]
                    ).to_dict(),
                    trial_results=json_safe(rows[1:]),
                )
                record["trials"] = []
                for row in rows:
                    trial = deepcopy(row["timing_record"]["trials"][0])
                    trial.update(
                        trial_id=row["trial_id"],
                        phase="prepared_first_execution"
                        if row["trial_id"] == 0
                        else "preconditioning"
                        if row["trial_id"] <= cfg["preconditioning"]
                        else "prepared_execution_repeat",
                    )
                    record["trials"].append(trial)
                    if row["timing_record"]["status"] != "valid":
                        record.update(status="unavailable", reason=row["timing_record"]["reason"])
                runs.append(
                    RunResult(
                        seed=seed,
                        status="success",
                        metrics=first["result"]["summary"],
                        history=first["result"]["history"],
                        timings=first["result"]["timings"],
                        timing_record=record,
                    )
                )
            exp = ExperimentResult(name=name, runs=runs)
            write_experiment_artifacts(exp, output / name)
            experiments[name] = exp
        summary = paired_timing_summary(
            experiments[BACKENDS[0]].runs,
            experiments[BACKENDS[1]].runs,
            "prepared_execution_repeat",
        )
        report["timing_summary"] = summary
        if summary["status"] == "valid":
            report["qdax_native_time_ratio"] = summary["right_median"] / summary["left_median"]
        passed = (
            summary["status"] == "valid"
            and not contamination
            and handler.compilation_signals > 0
            and all(all(r["agreement_by_trial"]) for r in report["references"])
        )
        report["status"] = "passed" if passed else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        report["compilation_log_summary"] = {
            "records": handler.count,
            "compilation_signals": handler.compilation_signals,
        }
        report.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            phase_events=events,
            execution_compilation_events=contamination,
        )
        (output / "report.json").write_text(json.dumps(json_safe(report), indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/timing/pt5.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-revision", required=True)
    parser.add_argument("--expect-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--expect-device", default="H100")
    args = parser.parse_args(argv)
    cfg = tomllib.loads(args.config.read_text())
    report = run_validation(
        cfg,
        args.output,
        expected_revision=args.expect_revision,
        expected_backend=args.expect_backend,
        expected_device=args.expect_device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "timing_summary": report.get("timing_summary"),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
