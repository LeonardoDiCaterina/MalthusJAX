#!/usr/bin/env python3
"""Run the Sphere 5D parity sweep directly with Composer.compare.

This mirrors examples/toy_gap_convergence.py more closely than TOML-driven
runs by:
- using shared_initial_population=True
- reusing the seed as pop_seed for each paired run
- passing evosax std_schedule=optax.constant_schedule(mutation_strength)
- keeping the toy's elite_k / crossover / mutation settings

The script runs one paired compare per seed so the initial population matches
what the toy script would generate for that seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optax
from scipy import stats

from malthusjax.composer import Composer


@dataclass(frozen=True)
class ComboSpec:
    label: str
    elite_k: int
    crossover_rate: float
    mutation_strength: float


COMBOS = (
    ComboSpec("Combo 1: elite_k=2, cr=0.3, ms=0.05", elite_k=2, crossover_rate=0.3, mutation_strength=0.05),
    ComboSpec("Combo 2: elite_k=4, cr=0.5, ms=0.1", elite_k=4, crossover_rate=0.5, mutation_strength=0.1),
    ComboSpec("Combo 3: elite_k=6, cr=0.7, ms=0.2", elite_k=6, crossover_rate=0.7, mutation_strength=0.2),
)

SEEDS = tuple(range(100))
DIMENSIONS = 5
POP_SIZE = 12
GENERATIONS = 20
BOUNDS = (-5.0, 5.0)
FITNESS_SPEC_TEMPLATE = "bbob:fn_name=sphere,num_dims=5,seed={seed},maximize=false"


def build_pipelines(combo: ComboSpec) -> dict[str, dict[str, Any]]:
    elite_ratio = float(combo.elite_k) / float(POP_SIZE)
    return {
        "malthusjax": {
            "backend": "malthusjax",
            "elitism": 0,
            "selection": f"elite_pool:num_selections={POP_SIZE},elite_k={combo.elite_k}",
            "crossover": f"evosax_uniform_crossover:crossover_rate={combo.crossover_rate}",
            "mutation": f"evosax_gaussian:mutation_strength={combo.mutation_strength}",
        },
        "evosax": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
            "strategy_params": {
                "crossover_rate": combo.crossover_rate,
                "elite_ratio": elite_ratio,
                "std_schedule": optax.constant_schedule(combo.mutation_strength),
            },
        },
    }


def extract_final_gap(run: Any) -> float:
    """Return the toy-style final gap from a pipeline run.

    For Sphere, the optimum is 0, so the toy reports |best_fitness|. We
    keep the same convention here so the comparison remains apples-to-apples.
    """
    if getattr(run, "metrics", None):
        best = run.metrics.get("best_fitness")
        if best is not None:
            return abs(float(best))
    history = getattr(run, "history", None) or []
    if history:
        best = history[-1].get("best_fitness")
        if best is not None:
            return abs(float(best))
    raise ValueError("Unable to extract final best_fitness from run")


def compute_metrics(mjx_gaps: list[float], evosax_gaps: list[float]) -> dict[str, float | int]:
    n = min(len(mjx_gaps), len(evosax_gaps))
    mjx = np.asarray(mjx_gaps[:n], dtype=float)
    evo = np.asarray(evosax_gaps[:n], dtype=float)

    mjx_wins = int(np.sum(mjx < evo))
    evosax_wins = int(np.sum(evo < mjx))
    ties = int(n - mjx_wins - evosax_wins)

    wilcoxon_pval = float(stats.wilcoxon(mjx, evo, alternative="two-sided").pvalue)
    sign_pos = int(np.sum(mjx - evo > 0))
    sign_test_pval = float(stats.binomtest(sign_pos, n, 0.5, alternative="two-sided").pvalue)
    ttest_pval = float(stats.ttest_rel(mjx, evo).pvalue)

    return {
        "n_paired": n,
        "mjx_wins": mjx_wins,
        "evosax_wins": evosax_wins,
        "ties": ties,
        "mjx_mean": float(np.mean(mjx)),
        "evosax_mean": float(np.mean(evo)),
        "mean_diff": float(np.mean(mjx) - np.mean(evo)),
        "wilcoxon_pval": wilcoxon_pval,
        "sign_test_pval": sign_test_pval,
        "ttest_pval": ttest_pval,
    }


def run_combo(composer: Composer, combo: ComboSpec, output_dir: Path) -> dict[str, Any]:
    pipelines = build_pipelines(combo)
    mjx_gaps: list[float] = []
    evosax_gaps: list[float] = []

    print(f"\n{'=' * 72}")
    print(combo.label)
    print(f"{'=' * 72}")

    for index, seed in enumerate(SEEDS, start=1):
        comp = composer.compare(
            pipelines=pipelines,
            seeds=(seed,),
            shared_initial_population=True,
            pop_seed=seed,
            fitness=FITNESS_SPEC_TEMPLATE.format(seed=seed),
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            bounds=BOUNDS,
            genome_type="real",
            genome_length=DIMENSIONS,
            maximize=False,
            elitism=0,
        )

        mjx_run = comp.pipelines["malthusjax"].runs[0]
        evo_run = comp.pipelines["evosax"].runs[0]

        mjx_gap = extract_final_gap(mjx_run)
        evo_gap = extract_final_gap(evo_run)
        mjx_gaps.append(mjx_gap)
        evosax_gaps.append(evo_gap)

        print(
            f"seed {seed:3d} ({index:3d}/{len(SEEDS)}): "
            f"MJX gap={mjx_gap:.6f}, EvoSAX gap={evo_gap:.6f}"
        )

    metrics = compute_metrics(mjx_gaps, evosax_gaps)
    metrics["combo"] = combo.label
    metrics["elite_k"] = combo.elite_k
    metrics["crossover_rate"] = combo.crossover_rate
    metrics["mutation_strength"] = combo.mutation_strength
    metrics["mjx_gaps"] = mjx_gaps
    metrics["evosax_gaps"] = evosax_gaps

    return metrics


def main() -> None:
    repo_root = Path(__file__).parent.parent
    report_dir = repo_root / "results" / "parity_sphere_d5_programmatic"
    report_dir.mkdir(parents=True, exist_ok=True)

    composer = Composer.create_default()
    results: list[dict[str, Any]] = []

    for combo in COMBOS:
        results.append(run_combo(composer, combo, report_dir))

    md = ["# Sphere 5D Programmatic Parity Sweep", "", "## Summary Table", ""]
    md.append("| Combo | Runs | MJX W | EvoSAX W | Ties | MJX Mean | EvoSAX Mean | Mean Diff | Wilcoxon p | t-test p |")
    md.append("|-------|------|-------|----------|------|----------|-------------|-----------|-----------|----------|")
    for row in results:
        md.append(
            f"| {row['combo']} | {row['n_paired']} | {row['mjx_wins']} | {row['evosax_wins']} | {row['ties']} | "
            f"{row['mjx_mean']:.6f} | {row['evosax_mean']:.6f} | {row['mean_diff']:.8f} | "
            f"{row['wilcoxon_pval']:.6f} | {row['ttest_pval']:.6f} |"
        )

    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("Parity criterion: Wilcoxon signed-rank test p-value > 0.05.")
    md.append("")

    for row in results:
        parity = "PASS" if row["wilcoxon_pval"] > 0.05 else "FAIL"
        md.append(f"### {row['combo']} {parity}")
        md.append("")
        md.append(f"- Paired runs: {row['n_paired']}")
        md.append(
            f"- Win distribution: MJX {row['mjx_wins']}/{row['n_paired']}, "
            f"EvoSAX {row['evosax_wins']}/{row['n_paired']}, Ties {row['ties']}"
        )
        md.append(f"- Mean gap: MJX={row['mjx_mean']:.6f}, EvoSAX={row['evosax_mean']:.6f}")
        md.append(f"- Mean difference (MJX - EvoSAX): {row['mean_diff']:.8f}")
        md.append(f"- Wilcoxon p-value: {row['wilcoxon_pval']:.6f}")
        md.append(f"- Paired t-test p-value: {row['ttest_pval']:.6f}")
        md.append(f"- Sign test p-value: {row['sign_test_pval']:.6f}")
        md.append("")

    json_data = {
        "summary": {
            "num_combos": len(results),
            "total_paired_runs": int(sum(row["n_paired"] for row in results)),
            "all_parity_pass": bool(all(row["wilcoxon_pval"] > 0.05 for row in results)),
        },
        "results": [
            {
                "combo": row["combo"],
                "elite_k": row["elite_k"],
                "crossover_rate": row["crossover_rate"],
                "mutation_strength": row["mutation_strength"],
                "n_paired": row["n_paired"],
                "mjx_wins": row["mjx_wins"],
                "evosax_wins": row["evosax_wins"],
                "ties": row["ties"],
                "mjx_mean": row["mjx_mean"],
                "evosax_mean": row["evosax_mean"],
                "mean_diff": row["mean_diff"],
                "wilcoxon_pval": row["wilcoxon_pval"],
                "sign_test_pval": row["sign_test_pval"],
                "ttest_pval": row["ttest_pval"],
                "parity_pass": row["wilcoxon_pval"] > 0.05,
            }
            for row in results
        ],
    }

    (report_dir / "summary_report.md").write_text("\n".join(md), encoding="utf-8")
    (report_dir / "parity_analysis.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    print(f"\nWrote reports to {report_dir}")
    print((report_dir / "summary_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
