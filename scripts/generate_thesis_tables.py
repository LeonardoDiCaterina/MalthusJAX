import argparse
import json
from pathlib import Path

import pandas as pd

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult


def process_result_json(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    pipelines_data = data.get("pipelines", data)

    pipelines = {}
    for name, pdata in pipelines_data.items():
        runs = []
        for seed_data in pdata.get("per_seed", []):
            # Repack raw keys into metrics dictionary so it matches RunResult spec
            metrics = {}
            for k in [
                "best_fitness",
                "final_generation",
                "total_evaluations",
                "qd_score",
                "coverage",
            ]:
                if k in seed_data and seed_data[k] is not None:
                    metrics[k] = seed_data[k]

            run = RunResult(
                seed=seed_data.get("seed", 1),
                status=seed_data.get("status", "success"),
                metrics=metrics,
                duration_seconds=seed_data.get("duration_seconds"),
                timings=seed_data.get("timings"),
                history=seed_data.get("convergence", []),
            )
            runs.append(run)
        if runs:
            pipelines[name] = ExperimentResult(name=name, runs=runs)

    if not pipelines:
        print(f"Skipping {json_path}: No valid runs found.")
        return

    # Use negate_map if needed, but since it's already negative/positive we can just let summary_table format it.
    # By default, MalthusJAX flips fitness, but since the raw data might already be flipped, let's leave negate_map empty
    comparison = ComparisonResult(
        pipelines=pipelines, shared_config={}, initial_population=None, negate_map={}
    )

    analysis_dir = json_path.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # 1. Output LaTeX
    latex_table = comparison.summary_table(latex=True)
    with open(analysis_dir / "comparison_table.tex", "w") as f:
        f.write(latex_table)

    # 2. Output CSV / Markdown
    table = comparison.summary_table()
    formatted = {}
    for pipe, metrics in table.items():
        formatted[pipe] = {}
        for k, v in metrics.items():
            if isinstance(v, dict) and "mean" in v:
                if v.get("ci_margin", 0.0) > 0.0:
                    formatted[pipe][k] = f"{v['mean']:.4g} ± {v['ci_margin']:.4g}"
                else:
                    formatted[pipe][k] = f"{v['mean']:.4g}"
            else:
                formatted[pipe][k] = str(v)

    df = pd.DataFrame(formatted).T
    df.to_csv(analysis_dir / "comparison_table.csv")

    md_text = df.to_markdown()
    with open(analysis_dir / "comparison_table.md", "w") as f:
        f.write(md_text)

    print(f"\n--- Results for {json_path.parent.name} ---")
    print(md_text)
    print("------------------------------------------\n")
    print(f"Generated tables in {analysis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="Base directory (e.g. results/h1_parity_qdax)"
    )
    args = parser.parse_args()

    base = Path(args.dir)
    print(f"Scanning {base} for results JSONs...")
    count = 0
    for p in base.rglob("*.json"):
        if "parity_results" in p.name or "ablation_results" in p.name:
            print(f"Processing {p}...")
            process_result_json(p)
            count += 1

    print(f"Done! Processed {count} experiments.")


if __name__ == "__main__":
    main()
