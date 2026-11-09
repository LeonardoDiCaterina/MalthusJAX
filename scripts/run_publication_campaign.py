#!/usr/bin/env python3
"""Serial study collection with bounded allocation time and retained attempts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_publication_study import ROOT, work_units


def source_hashes(phase):
    names = [f"scripts/run_publication_phase{phase}.py"]
    if phase in (2, 3):
        names.append("scripts/publication_rl.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def valid_result(directory, phase, index, unit, sources):
    manifest = json.loads((directory / "study_manifest.json").read_text())
    if (
        manifest.get("phase") != phase
        or manifest.get("status") != "success"
        or manifest.get("exit_code") != 0
    ):
        raise ValueError(f"not a successful matching phase unit: {directory}")
    stored = manifest["work_units"]
    if len(stored) != 1 or any(stored[0].get(k) != v for k, v in {"index": index, **unit}.items()):
        raise ValueError(f"unit arguments differ: {directory}")
    if any(manifest["source_sha256"].get(k) != v for k, v in sources.items()):
        raise ValueError(f"scientific worker source differs: {directory}")
    if subprocess.run(
        ["git", "diff", "--quiet", manifest["revision"], "HEAD", "--", "src", "packages"], cwd=ROOT
    ).returncode:
        raise ValueError("underlying library source changed since retained result")
    child = directory / f"unit{index}"
    summary = json.loads((child / ("report.json" if phase == 3 else "summary.json")).read_text())
    if phase == 1:
        if summary.get("failed_conditions") != 0 or len(summary.get("blocks", [])) != len(
            unit["seeds"]
        ):
            raise ValueError("incomplete Phase 1 result")
        packages = json.loads((child / "manifest.json").read_text())["versions"]
    elif phase == 2:
        if summary.get("successful") != 1 or summary.get("failed") != 0:
            raise ValueError("incomplete Phase 2 result")
        run = summary["runs"][0]
        if run.get("seed") != unit["seeds"][0] or run.get("status") != "success":
            raise ValueError("Phase 2 identity mismatch")
        expected_condition = unit["arguments"][unit["arguments"].index("--conditions") + 1]
        if run.get("condition") != expected_condition:
            raise ValueError("Phase 2 condition mismatch")
        packages = json.loads((child / "manifest.json").read_text())["packages"]
    else:
        if summary.get("status") != "success" or len(summary.get("runs", [])) != 1:
            raise ValueError("incomplete Phase 3 result")
        packages = {
            name: summary["environment"].get(name)
            for name in ("jax", "jaxlib", "tensorneat", "mujoco", "malthus-neat")
        }
    for package, expected_version in packages.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current != expected_version:
            raise ValueError(f"package changed since retained result: {package}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-revision", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-results", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--retry-units",
        type=int,
        nargs="*",
        default=[],
        help="Explicit retry of failed/incomplete attempts; originals retained",
    )
    parser.add_argument(
        "--budget-seconds", type=int, default=28200, help="Allocation work budget; default 7h50m"
    )
    parser.add_argument("--unit-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    started = time.monotonic()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != args.expect_revision:
        parser.error("unexpected deployed revision")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "src", "packages"], cwd=ROOT
    ).returncode:
        parser.error("underlying tracked scientific source has uncommitted changes")
    if args.unit_timeout_seconds < 1 or args.budget_seconds < args.unit_timeout_seconds + 60:
        parser.error("budget must allow a complete work-unit timeout plus one-minute reserve")
    protocol_path = ROOT / "configs/publication/study_v2.json"
    protocol = json.loads(protocol_path.read_text())
    phase_protocol = protocol[f"phase{args.phase}"]
    scientific_identity = {
        "phase_protocol": phase_protocol,
        "scientific_sources": source_hashes(args.phase),
    }
    units = work_units(protocol, args.phase)
    output = args.output.resolve()
    ledger_path = output / "campaign.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger["scientific_identity"] != scientific_identity or ledger["revision"] != revision:
            parser.error("resume identity changed; use a new campaign root and review provenance")
    else:
        ledger = {
            "phase": args.phase,
            "revision": revision,
            "scientific_identity": scientific_identity,
            "protocol": protocol,
            "units": {},
            "allocations": [],
            "status": "planned",
        }
    for directory in args.reuse_results:
        directory = directory.resolve()
        manifest = json.loads((directory / "study_manifest.json").read_text())
        index = manifest["work_units"][0]["index"]
        if not 0 <= index < len(units):
            parser.error("reused unit index out of range")
        valid_result(
            directory, args.phase, index, units[index], scientific_identity["scientific_sources"]
        )
        previous = ledger["units"].get(str(index))
        if previous and previous.get("result_directory") != str(directory):
            parser.error(f"duplicate retained result for unit {index}")
        ledger["units"][str(index)] = {
            "status": "success",
            "result_directory": str(directory),
            "external": True,
        }
    # Never trust a skip flag without checking the underlying result again.
    for key, entry in ledger["units"].items():
        if entry["status"] == "success":
            valid_result(
                Path(entry["result_directory"]),
                args.phase,
                int(key),
                units[int(key)],
                scientific_identity["scientific_sources"],
            )
    done = sum(entry["status"] == "success" for entry in ledger["units"].values())
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "total_units": len(units),
                    "retained_units": done,
                    "remaining_units": len(units) - done,
                    "budget_seconds": args.budget_seconds,
                    "status": "dry_run",
                    "output": str(output),
                    "phase1_equivalence": "unavailable",
                },
                indent=2,
            )
        )
        return 0
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS") != "1"
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        parser.error("execution requires allocated physical GPU 1; preserve CUDA mapping")
    output.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(ledger_path)

    allocation = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "budget_seconds": args.budget_seconds,
        "started_at_unix": time.time(),
        "completed_here": [],
    }
    ledger["allocations"].append(allocation)
    ledger["status"] = "running"
    save()
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    for index, unit in enumerate(units):
        entry = ledger["units"].get(str(index))
        if entry and entry["status"] == "success":
            continue
        if entry and index not in args.retry_units:
            ledger["status"] = "needs_review"
            save()
            print(
                f"Unit {index} has an incomplete/failed attempt. Review before explicit --retry-units.",
                flush=True,
            )
            return 2
        if args.budget_seconds - (time.monotonic() - started) < args.unit_timeout_seconds + 60:
            ledger["status"] = "budget_exhausted"
            allocation["elapsed_seconds"] = time.monotonic() - started
            save()
            print(
                f"Budget reached; retained {sum(e['status'] == 'success' for e in ledger['units'].values())}/{len(units)} units. Resume explicitly later.",
                flush=True,
            )
            return 0
        attempt_number = len(entry.get("attempts", [])) + 1 if entry else 1
        directory = output / f"unit{index:03d}_attempt{attempt_number}"
        if directory.exists():
            raise ValueError(f"attempt path already exists: {directory}")
        attempts = entry.get("attempts", []) if entry else []
        attempts.append({"directory": str(directory), "status": "running"})
        ledger["units"][str(index)] = {"status": "running", "attempts": attempts}
        save()
        command = [
            sys.executable,
            str(ROOT / "scripts/run_publication_study.py"),
            "--phase",
            str(args.phase),
            "--unit",
            str(index),
            "--execute",
            "--expect-revision",
            revision,
            "--output",
            str(directory),
        ]
        if args.phase == 1:
            command.append("--estimation-only")
        print(
            f"Phase {args.phase} unit {index}/{len(units) - 1}, seeds={unit['seeds']}: starting",
            flush=True,
        )
        with (
            (output / f"unit{index:03d}_attempt{attempt_number}.out").open("x") as stdout,
            (output / f"unit{index:03d}_attempt{attempt_number}.err").open("x") as stderr,
        ):
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                code = process.wait(timeout=args.unit_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code = 124
        attempts[-1].update(status="success" if code == 0 else "failed", exit_code=code)
        if code == 0:
            valid_result(
                directory, args.phase, index, unit, scientific_identity["scientific_sources"]
            )
            ledger["units"][str(index)].update(status="success", result_directory=str(directory))
            allocation["completed_here"].append(index)
            save()
            print(f"Phase {args.phase} unit {index}: success", flush=True)
        else:
            ledger["units"][str(index)]["status"] = "failed"
            ledger["status"] = "needs_review"
            allocation["elapsed_seconds"] = time.monotonic() - started
            save()
            print(f"Stopped at unit {index}, exit={code}; attempts retained.", flush=True)
            return code
    ledger["status"] = "complete"
    allocation["elapsed_seconds"] = time.monotonic() - started
    save()
    print(f"Phase {args.phase}: all {len(units)} units complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
