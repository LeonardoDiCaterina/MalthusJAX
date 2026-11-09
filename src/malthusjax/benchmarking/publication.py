"""Read-only claim inventory and deterministic, separate-output evidence rebuilding."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from ..stats.paper import analyze_archive_records
from .results import standard_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(paper):
    paper = Path(paper)
    items = []
    for pattern in ("tables/*.tex", "figures/*.pdf"):
        for path in sorted(paper.glob(pattern)):
            name = path.name.lower()
            defects = ["A19", "A22"]
            action = "unavailable_pending_source_mapping"
            if "island" in name:
                defects += ["A06", "A07"]
                action = "experiment_rerun_required"
            if "qdax" in name:
                defects += ["A01", "A02", "A03", "A08"]
                action = "experiment_rerun_required"
            items.append(
                {
                    "artifact": str(path.relative_to(paper)),
                    "sha256": sha(path),
                    "classification": action,
                    "applicable_findings": defects,
                    "raw_sources": [],
                    "analysis_command": None,
                    "reason": "Claim-to-source mapping and predeclared policies must be reviewed; file presence alone is not provenance",
                }
            )
    claims = []
    for path in sorted(paper.rglob("*.tex")):
        if any(p in {"results", ".git"} for p in path.relative_to(paper).parts):
            continue
        for line, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if re.search(r"\d", text) and not text.lstrip().startswith("%"):
                claims.append(
                    {
                        "source": str(path.relative_to(paper)),
                        "line": line,
                        "claim_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "status": "requires_evidence_mapping",
                    }
                )
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(paper), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    candidates = [
        {"path": str(p.relative_to(paper)), "sha256": sha(p)}
        for p in sorted((paper / "results").rglob("summary.json"))
    ]
    return {
        "schema_version": 1,
        "paper_revision": revision,
        "raw_candidates_unmapped": candidates,
        "scope": "Inventory, not validation of numerical claims",
        "artifacts": items,
        "numerical_line_candidates": claims,
    }


def rebuild(left_manifest, right_manifest, policy_path, output, *, dry_run=False):
    """Recompute archive inference from immutable raw snapshots, never paper tables."""
    paths = [Path(left_manifest), Path(right_manifest), Path(policy_path)]
    sources = [json.loads(p.read_text()) for p in paths]
    experiments = []
    for source in sources[:2]:
        if source.get("schema_version") != 1 or "experiment" not in source:
            raise ValueError("Expected a versioned immutable run manifest")
        exp = source["experiment"]
        environment = exp.get("metadata", {}).get("timing_environment", {})
        if not environment.get("revision") or not environment.get("source_sha256"):
            raise ValueError("Source/environment provenance is incomplete")
        experiments.append(exp)

    def records(exp):
        records = []
        for run in exp["runs"]:
            if run["status"] != "success" or not run["history"]:
                raise ValueError("All publication input runs require success and history")
            records.append(
                {
                    "seed": run["seed"],
                    "metric_metadata": run.get("metric_metadata", {}),
                    **{
                        m: [entry[m] for entry in run["history"]]
                        for m in ("best_fitness", "coverage", "qd_score")
                    },
                }
            )
        return records

    left, right = map(records, experiments)
    from .empirical import qd_comparison_status

    analysis = analyze_archive_records(
        left,
        right,
        sources[2],
        comparable_qd=qd_comparison_status(left, right)["status"] == "comparable",
    )
    if analysis["status"] != "valid":
        raise ValueError("Unavailable inference cannot generate a publication table")
    result = {
        "schema_version": 1,
        "status": "review_required",
        "analysis": analysis,
        "sources": {str(p): sha(p) for p in paths},
        "warning": "Separate review artifact; no manuscript claim approved automatically",
    }
    if dry_run:
        return result
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(standard_json(result), indent=2, allow_nan=False)
    )

    def escape(text):
        return (
            str(text).replace("\\", " ").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        )

    rows = [r"\begin{tabular}{llr}", r"Hypothesis & Selected conclusion & Selected p \\"]
    for row in analysis["results"]:
        p = (
            row["adjusted_p_value"]
            if row["decision_scope"] == "family_corrected"
            else next(
                (
                    t["p_value"]
                    for name, t in row["tests"].items()
                    if row["decision_basis"].startswith(name)
                ),
                None,
            )
        )
        rows.append(f"{escape(row['label'])} & {escape(row['conclusion'])} & {p:.6g}" + r" \\")
    rows.append(r"\end{tabular}")
    (output / "corrected_inference.tex").write_text("\n".join(rows) + "\n")
    return result
