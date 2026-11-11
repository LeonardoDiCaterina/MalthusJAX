#!/usr/bin/env python3
"""Expand small study work units; execution is explicit and allocation-only."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def work_units(protocol, phase):
    p = protocol[f"phase{phase}"]
    units = []
    if phase == 1:
        for function, dim, pop in itertools.product(p["functions"], p["dims"], p["populations"]):
            size = p["seeds_per_work_unit"]
            for offset in range(0, len(p["seeds"]), size):
                seeds = p["seeds"][offset : offset + size]
                units.append(
                    {
                        "seeds": seeds,
                        "conditions": 5,
                        "arguments": [
                            "--functions",
                            function,
                            "--dims",
                            str(dim),
                            "--populations",
                            str(pop),
                            "--generations",
                            str(p["generations"]),
                        ],
                    }
                )
    elif phase == 2:
        for seed, condition in itertools.product(p["seeds"], p["conditions"]):
            units.append(
                {
                    "seeds": [seed],
                    "conditions": 1,
                    "arguments": [
                        "--study-mode",
                        "ranking",
                        "--conditions",
                        condition,
                        "--population",
                        str(p["population"]),
                        "--generations",
                        str(p["generations"]),
                        "--horizon",
                        str(p["horizon"]),
                        "--eval-episodes",
                        str(p["eval_episodes"]),
                    ],
                }
            )
    else:
        for seed, topology in itertools.product(p["seeds"], p["topologies"]):
            arguments = ["--topologies", topology]
            for field in (
                "population",
                "islands",
                "generations",
                "horizon",
                "migration_interval",
                "migrants_per_island",
                "heldout_policy",
                "heldout_episodes",
                "max_nodes",
                "max_conns",
            ):
                arguments += ["--" + field.replace("_", "-"), str(p[field])]
            units.append({"seeds": [seed], "conditions": 1, "arguments": arguments})
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--unit", type=int)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/publication/study_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-revision", required=True)
    parser.add_argument(
        "--estimation-only",
        action="store_true",
        help="Explicit Phase 1 collection without equivalence claims",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Requires Slurm; omission only writes a schedule"
    )
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != args.expect_revision:
        parser.error("unexpected checkout revision")
    protocol = json.loads(args.protocol.read_text())
    units = work_units(protocol, args.phase)
    if args.unit is not None and not 0 <= args.unit < len(units):
        parser.error("work-unit index outside schedule")
    if args.execute:
        if args.unit is None:
            parser.error("execution requires --unit; no implicit full campaign")
        if args.phase == 1 and not args.estimation_only:
            parser.error(
                "Phase 1 requires --estimation-only while equivalence bounds remain pending"
            )
        if (
            not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_GPUS") != "1"
            or not os.environ.get("CUDA_VISIBLE_DEVICES")
        ):
            parser.error("requires Slurm physical GPU 1 with preserved CUDA mapping")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = range(len(units)) if args.unit is None else [args.unit]
    schedule = []
    for index in selected:
        unit = units[index]
        command = [
            sys.executable,
            str(ROOT / f"scripts/run_publication_phase{args.phase}.py"),
            "--output",
            str(output / f"unit{index}"),
            "--seeds",
            *map(str, unit["seeds"]),
            *unit["arguments"],
        ]
        schedule.append({"index": index, **unit, "command": command})
    manifest = {
        "stage": "study_v2",
        "phase": args.phase,
        "status": "planned",
        "revision": revision,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "scripts").glob("*publication*.py")
        },
        "work_units": schedule,
        "execute": args.execute,
        "phase1_equivalence_available": False,
        "phase1_estimation_only": args.estimation_only,
        "confirmation_eligible": False if args.phase == 1 else None,
    }

    def save():
        temporary = output / "study_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "study_manifest.json")

    save()
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "total_units": len(units),
                    "training_runs": sum(len(u["seeds"]) * u["conditions"] for u in units),
                    "manifest": str(output / "study_manifest.json"),
                    "execution": "NOT STARTED",
                    "budget_approval": protocol["resources"]["full_campaign_budget_approval"],
                },
                indent=2,
            )
        )
        return 0
    import jax

    if jax.default_backend() != "gpu" or not all("H100" in d.device_kind for d in jax.devices()):
        parser.error("allocated H100 backend required")
    manifest["status"] = "running"
    save()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "docs/handoff/capture_cluster_environment.py"),
            "--probe-jax",
            "--output",
            str(output / "environment.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    started = time.perf_counter()
    with (output / "worker.out").open("x") as stdout, (output / "worker.err").open("x") as stderr:
        result = subprocess.run(
            schedule[0]["command"], cwd=ROOT, env=env, stdout=stdout, stderr=stderr
        )
    manifest.update(
        status="success" if result.returncode == 0 else "failed",
        exit_code=result.returncode,
        wall_seconds=time.perf_counter() - started,
    )
    save()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
