#!/usr/bin/env bash
# CPU-only analysis of completed ledgers; never glob or mutate running Phase 3.
set -euo pipefail
if [[ $# -gt 2 || "${1:-}" == "--help" ]]; then
    echo 'Usage: bash aggregate_publication_phases12.sh [CAMPAIGN_ROOT] [NEW_OUTPUT_DIR]'
    exit 0
fi
test -f scripts/analyze_publication_study.py || { echo 'Run from the repository root.' >&2; exit 2; }
campaign_root=${1:-results/paper-study-v2}
analysis_output=${2:-results/analysis-phases12-$(date -u +%Y%m%dT%H%M%SZ)}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
python - "$campaign_root" "$analysis_output" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from run_publication_campaign import valid_result
from run_publication_study import work_units

campaign, output = (Path(p).resolve() for p in sys.argv[1:])
if output.exists():
    raise SystemExit(f"Refusing existing output directory: {output}")

def read(path):
    return json.loads(path.read_text())

def write_csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

index, directories, ledgers = {}, {}, {}
for phase in (1, 2):
    path = campaign / f"phase{phase}" / "campaign.json"
    ledger = read(path)
    if ledger.get("phase") != phase or ledger.get("status") != "complete":
        raise SystemExit(f"Phase {phase} ledger is not complete: {path}")
    units = work_units(ledger["protocol"], phase)
    if set(ledger["units"]) != {str(i) for i in range(len(units))}:
        raise SystemExit(f"Phase {phase} unit index is incomplete/unexpected")
    resolved, seen = [], set()
    for i, unit in enumerate(units):
        entry = ledger["units"][str(i)]
        if entry.get("status") != "success":
            raise SystemExit(f"Phase {phase} unit {i} is not successful")
        directory = Path(entry["result_directory"])
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
        if directory in seen:
            raise SystemExit(f"Duplicate retained directory: {directory}")
        seen.add(directory)
        valid_result(directory, phase, i, unit, ledger["scientific_identity"]["scientific_sources"])
        resolved.append(directory / f"unit{i}")
    directories[phase], ledgers[phase] = resolved, ledger
    index[f"phase{phase}"] = {
        "ledger": str(path), "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "revision": ledger["revision"], "completed_units": len(units),
        "inputs": [str(p) for p in resolved],
        "expected_seeds": ledger["protocol"][f"phase{phase}"]["seeds"],
    }

# Flatten raw values without pooling seeds or converting benchmark orientation.
phase1_rows, phase2_rows, seen = [], [], set()
for directory in directories[1]:
    summary = read(directory / "summary.json")
    for block in summary["blocks"]:
        coordinate, seed = block["block"].rsplit("_s", 1)
        hashes = set()
        for condition in block["conditions"]:
            path = directory / block["block"] / f"{condition}.json"
            run = read(path)
            identity = coordinate, int(seed), condition
            if identity in seen or run.get("status") != "success":
                raise SystemExit(f"Duplicate/failed Phase 1 run: {identity}")
            seen.add(identity)
            if run.get("input_population_hash"):
                hashes.add(run["input_population_hash"])
            metrics = run.get("metrics", {})
            phase1_rows.append({"coordinate": coordinate, "seed": int(seed), "condition": condition,
                "best_fitness": metrics.get("best_fitness"), "coverage": metrics.get("coverage"),
                "qd_score": metrics.get("qd_score"), "collection_wall_seconds": run.get("collection_wall_seconds"),
                "source": str(path)})
        if len(hashes) > 1:
            raise SystemExit(f"Phase 1 shared initialization mismatch: {block['block']}")
for directory in directories[2]:
    phase2_rows.extend(read(directory / "summary.json")["runs"])

output.mkdir(parents=True, exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(output / ".mplconfig")
(output / "input_index.json").write_text(json.dumps(index, indent=2) + "\n")
write_csv(output / "phase1_raw.csv", phase1_rows,
          ["coordinate", "seed", "condition", "best_fitness", "coverage", "qd_score", "collection_wall_seconds", "source"])
write_csv(output / "phase2_raw.csv", [{**r, **r.get("cost", {})} for r in phase2_rows],
          ["seed", "condition", "status", "heldout_mean_return", "initialization_and_preparation_seconds",
           "prepared_execution_seconds", "train_end_to_end_seconds", "initial_population_sha256"])
invalid = []
for r in phase2_rows:
    for episode, d in enumerate(r.get("heldout_rollout_diagnostics", [])):
        if d.get("rollout_invalid_live_steps", 0) or d.get("rollout_invalid_resets", 0):
            invalid.append({"seed": r["seed"], "condition": r["condition"], "episode": episode, "diagnostics": d})
(output / "rollout_diagnostics.json").write_text(json.dumps({"invalid_heldout_episodes": invalid}, indent=2) + "\n")

for phase in (1, 2):
    if phase == 2 and invalid:
        print("Phase 2 inferential analysis withheld: invalid held-out episodes; see rollout_diagnostics.json")
        continue
    inputs = directories[phase] if phase == 1 else [p / "summary.json" for p in directories[phase]]
    command = [sys.executable, str(root / "scripts/analyze_publication_study.py"),
               f"--phase{phase}", *map(str, inputs), "--expected-seeds",
               *map(str, index[f"phase{phase}"]["expected_seeds"]), "--output", str(output / f"phase{phase}")]
    subprocess.run(command, check=True)

def fmt(value):
    return f"{value:.3g}" if isinstance(value, (int, float)) else "-"

def interval(estimate):
    if not estimate:
        return "unavailable"
    ci = estimate.get("ordinary_95_ci")
    return fmt(estimate["mean"]) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else " [CI unavailable]")

lines = [f"PHASE 1: {len(phase1_rows)} runs; paired differences are native minus adapter.",
         "Ordinary 95% intervals; equivalence unavailable (bounds pending).",
         f"{'Coordinate':<24} {'Contrast':<22} {'n':>4} {'Fitness difference [95% CI]':>33}"]
report1 = read(output / "phase1/phase1_estimation.json")
flat = []
for c in report1["contrasts"]:
    e = c.get("estimate") or {}
    ci = e.get("ordinary_95_ci") or [None, None]
    flat.append({**c, "n": e.get("n"), "mean_difference": e.get("mean"), "sd_difference": e.get("sd"),
                 "ci_low": ci[0], "ci_high": ci[1], "missing_pairs": len(c["missing_or_failed_seeds"])})
    if c["metric"] == "best_fitness":
        label = c["left"].replace("evosax_", "").replace("qdax_native", "QD native")
        lines.append(f"{c['coordinate']:<24} {label:<22} {str(e.get('n','-')):>4} {interval(c.get('estimate')):>33}")
write_csv(output / "phase1_contrasts.csv", flat,
          ["coordinate", "left", "right", "metric", "n", "mean_difference", "sd_difference", "ci_low", "ci_high", "missing_pairs"])
if not invalid:
    report2 = read(output / "phase2/phase2_ranking.json")
    lines += ["", f"PHASE 2: {len(phase2_rows)} runs; held-out return (higher better).",
              f"Complete family: {report2['complete_family']}; inferential family available: {report2['inferential_family_available']}",
              f"{'Condition':<21} {'n':>4} {'Mean return [95% CI]':>30} {'Train setup+scan s':>20}"]
    for condition, e in report2["descriptives"].items():
        lines.append(f"{condition:<21} {e['n']:>4} {interval(e):>30} {fmt(e['mean_training_setup_and_scan_seconds']):>20}")
    lines += ["", "Paired return differences: left minus right; ordinary 95% CIs."]
    rows = []
    for c in report2["contrasts"]:
        e = c.get("estimate") or {}
        ci = e.get("ordinary_95_ci") or [None, None]
        rows.append({**c, "n": e.get("n"), "mean_difference": e.get("mean"), "ci_low": ci[0], "ci_high": ci[1]})
        lines.append(f"{c['left']} - {c['right']}: {interval(c.get('estimate'))}; Holm p={fmt(c['holm_p'])}")
    write_csv(output / "phase2_contrasts.csv", rows, ["left", "right", "n", "mean_difference", "ci_low", "ci_high", "raw_p", "holm_p"])
    lines.append("Sample-mean return–training-cost frontier: " + ", ".join(report2.get("sample_mean_cost_frontier", [])))
else:
    lines.append(f"PHASE 2: {len(invalid)} invalid held-out episodes; inference withheld, raw data retained.")
lines += ["", "Phase 1 collection wall time is not steady-state speed. Phase 2 cost excludes held-out evaluation.",
          "No forced winner or equivalence claim. CIs are not simultaneous. Phase 3 was not read or modified.",
          f"Output: {output}"]
text = "\n".join(lines) + "\n"
(output / "discussion.txt").write_text(text)
print(text)
PY
