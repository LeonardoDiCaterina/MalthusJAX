"""Versioned timing observations; objective success is independent of timing validity."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import jax
import jaxlib
import numpy as np

SCHEMA_VERSION = 1
FIRST = "prepared_first_execution"
ALIASES = {
    "execution": FIRST,
    "execution_time": FIRST,
    "time_execution": FIRST,
    "duration_seconds": "runner_end_to_end",
    "total": "adapter_through_execution",
    "warmup": "initialization_and_compilation",
}


def json_safe(value):
    """Keep invalid raw values visible without emitting nonstandard JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"raw_nonfinite": repr(value)}
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def environment_identity() -> dict[str, Any]:
    """Capture lightweight provenance once, outside any measured adapter call."""

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_path": str(Path(__file__).resolve()),
        "backend": jax.default_backend(),
        "devices": [d.device_kind for d in jax.devices()],
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "diff_identity": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
    }


def input_signature(value: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "shape": list(getattr(x, "shape", ())),
            "dtype": str(getattr(x, "dtype", type(x).__name__)),
            "weak_type": bool(getattr(x, "weak_type", False)),
            "sharding": str(getattr(x, "sharding", None)),
        }
        for path, x in jax.tree_util.tree_flatten_with_path(value)[0]
    ]


@dataclass(frozen=True)
class TimingPolicy:
    target_phase: str = FIRST
    preconditioning: int = 0
    repetitions: int = 0

    def __post_init__(self):
        if self.target_phase not in (FIRST, "prepared_execution_repeat"):
            raise ValueError("Unknown target timing phase")
        if any(type(n) is not int or n < 0 for n in (self.preconditioning, self.repetitions)):
            raise ValueError("Timing counts must be nonnegative")
        if self.target_phase == "prepared_execution_repeat" and self.repetitions < 1:
            raise ValueError("Repeated timing requires a predetermined positive repetition count")
        if self.target_phase == FIRST and (self.preconditioning or self.repetitions):
            raise ValueError("Extra trials require an explicit repeated-execution target")

    def to_dict(self):
        return {
            **asdict(self),
            "reset": "fresh_initial_state_same_seed_each_trial",
            "cache": "retained_executable; cache_not_cleared",
            "estimator": "median_within_seed"
            if self.repetitions
            else "one_first_execution_per_seed",
        }


@dataclass
class TimingTrial:
    phase: str
    seconds: float | None
    trial_id: int = 0
    status: str = "valid"
    reason: str | None = None
    raw_value: str | None = None

    def __post_init__(self):
        if (
            not isinstance(self.seconds, Real)
            or isinstance(self.seconds, bool)
            or not math.isfinite(self.seconds)
            or self.seconds <= 0
        ):
            self.raw_value = repr(self.seconds)
            self.seconds = None
            self.status = "unavailable"
            self.reason = self.reason or "duration must be finite and positive"


@dataclass
class TimingRecord:
    schema_version: int = SCHEMA_VERSION
    status: str = "legacy/unknown"
    reason: str | None = "No versioned producer timing contract"
    target_phase: str = FIRST
    seed: int | None = None
    run_order: int | None = None
    scope: str = "unknown"
    backend_mode: str = "unknown"
    policy: dict[str, Any] = field(
        default_factory=lambda: {
            "preconditioning": 0,
            "repetitions": 0,
            "reset": "fresh_initial_state_per_seed",
            "cache": "retained_executable; cache_not_cleared",
            "estimator": "one_first_execution_per_seed",
        }
    )
    signature: Any = None
    environment: dict[str, Any] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)
    phases: dict[str, Any] = field(default_factory=dict)
    legacy_aliases: dict[str, str] = field(default_factory=lambda: dict(ALIASES))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_producer(cls, timings, metadata, *, seed, order, environment, duration):
        record = cls(seed=seed, run_order=order, environment=environment)
        record.phases = {
            "initialization": None,
            "compilation": None,
            "host_postprocessing": None,
            "runner_end_to_end": duration,
            "initialization_and_compilation": (timings or {}).get("warmup"),
            "adapter_through_execution": (timings or {}).get("total"),
        }
        if metadata.get("schema_version") != SCHEMA_VERSION:
            return record
        record.status = metadata.get("status", "unavailable")
        record.reason = metadata.get("reason")
        record.scope = metadata.get("scope", "unknown")
        record.backend_mode = metadata.get("backend_mode", "prepared_aot")
        record.signature = metadata.get("signature")
        record.phases.update(metadata.get("phases", {}))
        record.trials = [asdict(TimingTrial(FIRST, (timings or {}).get("execution")))]
        if record.trials[0]["status"] != "valid":
            record.status, record.reason = (
                "unavailable",
                record.reason or record.trials[0]["reason"],
            )
        required = ("initialization", "compilation", "host_postprocessing")
        if record.signature is None or any(
            not isinstance(record.phases.get(p), Real)
            or not math.isfinite(record.phases[p])
            or record.phases[p] < 0
            for p in required
        ):
            record.status = "unavailable"
            record.reason = (
                record.reason or "Producer phase or executable/input signature is missing"
            )
        return record


def raw_record(run: Any) -> dict[str, Any]:
    return (
        run.get("timing_record") if isinstance(run, dict) else getattr(run, "timing_record", None)
    ) or {}


def select_timing(run: Any, phase: str = FIRST) -> tuple[float, tuple]:
    """Never substitute another phase, omit an invalid trial, or infer legacy validity."""
    record = raw_record(run)
    if not isinstance(record, dict):
        raise ValueError("Invalid timing record")
    phase = ALIASES.get(phase, phase)
    if record.get("schema_version") != SCHEMA_VERSION or record.get("status") != "valid":
        raise ValueError(
            f"Timing unavailable: {record.get('reason') or 'legacy/unknown timing schema'}"
        )
    if record.get("target_phase") != phase or phase not in (FIRST, "prepared_execution_repeat"):
        raise ValueError("Timing phase mismatch with declared target")
    seed = run.get("seed") if isinstance(run, dict) else getattr(run, "seed", None)
    if (
        record.get("seed") != seed
        or type(record.get("run_order")) is not int
        or record["run_order"] < 0
    ):
        raise ValueError("Timing seed or run order is missing or inconsistent")
    if record.get("backend_mode") != "prepared_aot":
        raise ValueError("Unsupported timing backend mode")
    phases = record.get("phases", {})
    if not isinstance(phases, dict) or any(
        not isinstance(phases.get(p), Real) or not math.isfinite(phases[p]) or phases[p] < 0
        for p in ("initialization", "compilation", "host_postprocessing")
    ):
        raise ValueError("Timing phase durations are missing or invalid")
    all_trials = record.get("trials", [])
    policy = record.get("policy", {})
    if not isinstance(policy, dict) or not isinstance(all_trials, list):
        raise ValueError("Invalid timing policy or trials")
    for key in ("preconditioning", "repetitions"):
        if type(policy.get(key)) is not int or policy[key] < 0:
            raise ValueError("Missing or invalid predetermined timing counts")
    if any(not isinstance(t, dict) for t in all_trials):
        raise ValueError("Malformed timing trials")
    expected_phases = (
        [FIRST]
        + ["preconditioning"] * policy["preconditioning"]
        + ["prepared_execution_repeat"] * policy["repetitions"]
    )
    if [t.get("phase") for t in all_trials] != expected_phases:
        raise ValueError("Timing trial phases differ from declared policy")
    expected = 1 + policy.get("preconditioning", 0) + policy.get("repetitions", 0)
    if len(all_trials) != expected or [t.get("trial_id") for t in all_trials] != list(
        range(expected)
    ):
        raise ValueError("Timing trial count or order does not match declared policy")
    if not all_trials or all_trials[0].get("phase") != FIRST:
        raise ValueError("First execution must remain recorded")
    trials = [t for t in all_trials if t.get("phase") == phase]
    count = 1 if phase == FIRST else policy.get("repetitions", 0)
    if len(trials) != count or count < 1 or any(t.get("status") != "valid" for t in all_trials):
        raise ValueError("Timing requires every declared trial to be valid")
    for trial in all_trials:
        value = trial.get("seconds")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Timing duration must be finite and positive")
    value = float(np.median([t["seconds"] for t in trials]))
    if (
        not record.get("signature")
        or not record.get("environment")
        or record.get("scope") in (None, "unknown")
    ):
        raise ValueError("Timing provenance or finalization scope is missing")
    compatibility = (
        SCHEMA_VERSION,
        phase,
        record["scope"],
        record.get("backend_mode"),
        json.dumps(record.get("policy"), sort_keys=True),
        json.dumps(record["environment"], sort_keys=True),
    )
    return float(value), compatibility


def timing_values(runs, phase=FIRST):
    values, contract, seeds = [], None, set()
    for run in runs:
        seed = run.get("seed") if isinstance(run, dict) else run.seed
        if seed in seeds:
            raise ValueError(f"Duplicate timing seed {seed}")
        seeds.add(seed)
        value, current = select_timing(run, phase)
        if contract is not None and current != contract:
            raise ValueError("Incompatible timing phases, policies, scopes, or environments")
        contract = current
        values.append(value)
    if not values:
        raise ValueError("No timing observations")
    return values, contract


def paired_timing_summary(left, right, phase=FIRST):
    """Descriptive per-seed measurements only; inference belongs to P4."""
    try:
        lv, lc = timing_values(left, phase)
        rv, rc = timing_values(right, phase)
        if lc != rc or len(lv) != len(rv):
            raise ValueError("Incompatible timing contracts or sample counts")
        return {
            "status": "valid",
            "phase": phase,
            "estimator": "median_across_seeds",
            "count": len(lv),
            "left_values": lv,
            "right_values": rv,
            "left_median": float(np.median(lv)),
            "right_median": float(np.median(rv)),
        }
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc), "count": 0}
