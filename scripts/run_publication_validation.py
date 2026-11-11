#!/usr/bin/env python3
"""Expand/run a bounded allocated-GPU validation; never a confirmation campaign."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect-revision", required=True)
    p.add_argument("--stage", choices=["validation", "exploratory"], default="validation")
    p.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--dry-run", action="store_true", help="Write schedule without initializing JAX")
    args = p.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != args.expect_revision:
        p.error("checkout revision does not match --expect-revision")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ["--seeds", "17"]
    work = {
        "1": ["--functions", "sphere", "--dims", "4", "--populations", "16", "--generations", "2", "--timing", "--repetitions", "2"],
        "2": ["--population", "4", "--generations", "1", "--horizon", "2", "--eval-episodes", "1"],
        "3": ["--population", "6", "--islands", "3", "--generations", "2", "--horizon", "2", "--migration-interval", "1", "--migrants-per-island", "1", "--heldout-policy", "all", "--heldout-episodes", "1", "--max-nodes", "16", "--max-conns", "40", "--topologies", "single", "ring", "bidirectional", "fully_connected"],
    }
    exploratory_config = None
    if args.stage == "exploratory":
        config_path = ROOT / "configs/publication/exploratory.json"
        exploratory_config = json.loads(config_path.read_text())
        common = ["--seeds", *map(str, exploratory_config["seeds"])]
        work = exploratory_config["phase_arguments"]
    phases = list(work) if args.phase == "all" else [args.phase]
    commands = [[sys.executable, str(ROOT / f"scripts/run_publication_phase{phase}.py"),
                 "--output", str(output / f"phase{phase}"), *common, *work[phase]] for phase in phases]
    sources = sorted((ROOT / "scripts").glob("*publication*.py"))
    manifest = {"stage": "exploratory" if args.stage == "exploratory" else "cluster_validation_only", "publication_ready": False,
                "revision": revision, "commands": commands, "dry_run": args.dry_run,
                "source_sha256": {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                "scientific_policy": "UNFROZEN: no confirmation or equivalence claim",
                "completed": [], "status": "planned", "exploratory_protocol": exploratory_config,
                "confirmation_eligible": False}
    if exploratory_config is not None:
        manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    def save():
        temporary = output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")
    save()
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("execution requires a Slurm allocation; use --dry-run locally")
    import jax
    if jax.default_backend() != "gpu" or not all("H100" in str(d.device_kind) for d in jax.devices()):
        p.error("expected allocated H100 GPU backend")
    manifest["devices"] = [str(d) + ":" + d.device_kind for d in jax.devices()]
    manifest["allocation"] = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")}
    subprocess.run([sys.executable, str(ROOT / "docs/handoff/capture_cluster_environment.py"),
                    "--probe-jax", "--output", str(output / "environment_capture.json")], check=True, cwd=ROOT)
    manifest["status"] = "running"
    save()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for phase, command in zip(phases, commands):
        with (output / f"phase{phase}.out").open("x") as stdout, (output / f"phase{phase}.err").open("x") as stderr:
            import time
            started = time.perf_counter()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        manifest["completed"].append({"phase": phase, "exit_code": result.returncode,
                                      "wall_seconds": time.perf_counter() - started})
        if result.returncode:
            manifest["status"] = "failed"
            save()
            return result.returncode
        save()
    manifest["status"] = "success"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
