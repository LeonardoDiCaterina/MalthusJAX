from pathlib import Path
from collections import Counter
import hashlib, json, math, subprocess

root = Path.cwd()
paper = root.parent / "MalthusJAX_Paper"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


report = {
    "schema_version": 1,
    "purpose": "Read-only source census; candidate association is not verified scientific provenance",
    "framework_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "paper_revision": subprocess.check_output(
        ["git", "-C", str(paper), "rev-parse", "HEAD"], text=True
    ).strip(),
    "suites": [],
}
for name in ["h1_parity_lhs", "h2_ablation_final"]:
    files = sorted((root / "results/_archive" / name).rglob("benchmark_results.json"))
    pipes = {}
    sources = []
    for p in files:
        d = json.loads(p.read_text())
        sources.append(
            {"path": str(p.relative_to(root)), "sha256": digest(p), "coordinate": d.get("config")}
        )
        for pn, payload in d.get("pipelines", {}).items():
            stat = pipes.setdefault(pn, Counter())
            rows = payload.get("per_seed", [])
            seen = set()
            for r in rows:
                stat["records"] += 1
                stat["success" if r.get("status") == "success" else "non_success"] += 1
                seed = r.get("seed")
                stat["duplicate_seed_within_coordinate"] += seed in seen
                seen.add(seed)
                x = r.get("best_fitness")
                stat["nonfinite_or_missing_best"] += not (
                    isinstance(x, (int, float)) and math.isfinite(x)
                )
                stat["versioned_timing_records"] += bool(r.get("timing_record"))
                stat["versioned_metric_metadata"] += bool(r.get("metric_metadata"))
    report["suites"].append(
        {
            "name": name,
            "coordinate_files": len(files),
            "pipelines": pipes,
            "sources": sources,
            "disposition": "Quality eligible for provenance/semantic review only; mixed-phase legacy timings require new measurement",
        }
    )
report["empirical_candidates"] = []
for p in [root / "results/empirical_curves.json", paper / "figures/data/empirical_curves.json"]:
    d = json.loads(p.read_text())
    report["empirical_candidates"].append(
        {
            "path": str(p),
            "sha256": digest(p),
            "metadata": d.get("metadata"),
            "d5_has_per_seed": "per_seed" in d.get("dataset_5a", {}),
            "disposition": "legacy aggregate; not valid corrected inferential input",
        }
    )
Path("docs/publication_design/source_census.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
for s in report["suites"]:
    print(s["name"], s["coordinate_files"], s["pipelines"])
