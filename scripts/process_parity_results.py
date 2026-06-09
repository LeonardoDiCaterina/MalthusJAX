#!/usr/bin/env python3
"""Process paired parity sweep results from existing run artifacts.

The script scans a results directory for paired backend folders containing
"histories_combined.csv" and computes two classes of metrics:

1) Raw paired tests on final best_fitness values.
2) Distance-to-optimum paired tests (when gap_to_optimum is available or
   when --optimum is provided).

Outputs:
- JSON report with per-pair detailed metrics.
- Markdown summary table and interpretation blocks.

Example:
    python scripts/process_parity_results.py \
      --results-dir results \
      --output-dir results/parity_postprocess \
      --optimum 0.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class PipelineData:
    """Per-backend metrics aligned by seed."""

    name: str
    final_best_by_seed: dict[int, float]
    gap_by_seed: dict[int, float]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_generation_col(fieldnames: list[str]) -> str | None:
    for candidate in ("generation", "generation_counter", "final_generation"):
        if candidate in fieldnames:
            return candidate
    return None


def load_final_best_from_histories(csv_path: Path) -> dict[int, float]:
    """Load final best_fitness per seed from histories_combined.csv."""

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "seed" not in reader.fieldnames or "best_fitness" not in reader.fieldnames:
            raise ValueError(f"Missing required columns in {csv_path}")

        generation_col = _find_generation_col(reader.fieldnames)

        by_seed: dict[int, tuple[float, float]] = {}
        order = 0.0
        for row in reader:
            order += 1.0
            seed = int(row["seed"])
            best = float(row["best_fitness"])

            if generation_col is not None:
                generation = _safe_float(row.get(generation_col))
                sort_key = generation if generation is not None else order
            else:
                sort_key = order

            prev = by_seed.get(seed)
            if prev is None or sort_key >= prev[0]:
                by_seed[seed] = (sort_key, best)

    return {seed: best for seed, (_, best) in by_seed.items()}


def load_gap_from_summary(summary_path: Path) -> dict[int, float]:
    """Load gap_to_optimum per seed from summary.json, if present."""

    if not summary_path.exists():
        return {}

    data = json.loads(summary_path.read_text())
    runs = data.get("runs", [])

    gap_by_seed: dict[int, float] = {}
    for run in runs:
        seed = run.get("seed")
        metrics = run.get("metrics", {}) or {}
        gap = metrics.get("gap_to_optimum")
        if seed is None:
            continue
        gap_val = _safe_float(gap)
        if gap_val is not None:
            gap_by_seed[int(seed)] = gap_val

    return gap_by_seed


def load_pipeline_data(pipeline_dir: Path) -> PipelineData:
    csv_path = pipeline_dir / "histories_combined.csv"
    summary_path = pipeline_dir / "summary.json"
    final_best = load_final_best_from_histories(csv_path)
    gap = load_gap_from_summary(summary_path)
    return PipelineData(name=pipeline_dir.name, final_best_by_seed=final_best, gap_by_seed=gap)


def effect_size_cohen_dz(diffs: np.ndarray) -> float:
    if diffs.size < 2:
        return float("nan")
    std = float(np.std(diffs, ddof=1))
    if math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(diffs) / std)


def effect_size_rank_biserial(left: np.ndarray, right: np.ndarray) -> float | None:
    # Paired rank-biserial correlation from non-zero signed differences.
    diffs = left - right
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return None
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    denom = n_pos + n_neg
    if denom == 0:
        return None
    return float((n_pos - n_neg) / denom)


def paired_stats(left: np.ndarray, right: np.ndarray, direction: str) -> dict[str, Any]:
    """Compute paired tests and effect sizes.

    direction: expected direction for one-sided test, either:
      - "left_lt_right" means lower left is better (alternative='less')
      - "left_gt_right" means higher left is better (alternative='greater')
    """

    if left.size != right.size:
        raise ValueError("Paired arrays must have equal size")

    if left.size == 0:
        raise ValueError("No paired values")

    alternative = "less" if direction == "left_lt_right" else "greater"

    diffs = left - right
    wins_left = int(np.sum(left < right))
    wins_right = int(np.sum(right < left))
    ties = int(np.sum(left == right))

    if left.size >= 2:
        t_two = stats.ttest_rel(left, right, alternative="two-sided")
        t_one = stats.ttest_rel(left, right, alternative=alternative)
        t_two_p = float(t_two.pvalue)
        t_one_p = float(t_one.pvalue)
    else:
        t_two_p = float("nan")
        t_one_p = float("nan")

    try:
        w_two = stats.wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
        w_one = stats.wilcoxon(left, right, alternative=alternative, zero_method="wilcox")
        w_two_p = float(w_two.pvalue)
        w_one_p = float(w_one.pvalue)
    except ValueError:
        w_two_p = float("nan")
        w_one_p = float("nan")

    sign_pos = int(np.sum(diffs > 0))
    sign_neg = int(np.sum(diffs < 0))
    sign_n = sign_pos + sign_neg
    sign_k = sign_neg if direction == "left_lt_right" else sign_pos
    if sign_n > 0:
        sign_one = stats.binomtest(sign_k, sign_n, 0.5, alternative="greater")
        sign_one_p = float(sign_one.pvalue)
    else:
        sign_one_p = float("nan")

    return {
        "n": int(left.size),
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "left_median": float(np.median(left)),
        "right_median": float(np.median(right)),
        "mean_diff_left_minus_right": float(np.mean(diffs)),
        "median_diff_left_minus_right": float(np.median(diffs)),
        "wins_left": wins_left,
        "wins_right": wins_right,
        "ties": ties,
        "paired_t_two_sided_p": t_two_p,
        "paired_t_one_sided_p": t_one_p,
        "wilcoxon_two_sided_p": w_two_p,
        "wilcoxon_one_sided_p": w_one_p,
        "sign_test_one_sided_p": sign_one_p,
        "cohen_dz": effect_size_cohen_dz(diffs),
        "rank_biserial": effect_size_rank_biserial(left, right),
    }


def pair_key_from_name(name: str) -> str | None:
    lower = name.lower()
    if "malthusjax" in lower:
        return re.sub("malthusjax", "{backend}", lower)
    if "evosax" in lower:
        return re.sub("evosax", "{backend}", lower)
    return None


def discover_pipeline_pairs(results_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return (malthusjax_dir, evosax_dir, pair_key)."""

    candidates: dict[str, dict[str, Path]] = {}
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        if not (item / "histories_combined.csv").exists():
            continue
        key = pair_key_from_name(item.name)
        if key is None:
            continue

        slots = candidates.setdefault(key, {})
        lname = item.name.lower()
        if "malthusjax" in lname:
            slots["malthusjax"] = item
        elif "evosax" in lname:
            slots["evosax"] = item

    pairs: list[tuple[Path, Path, str]] = []
    for key, slots in sorted(candidates.items()):
        mj = slots.get("malthusjax")
        ev = slots.get("evosax")
        if mj is not None and ev is not None:
            pairs.append((mj, ev, key))

    return pairs


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Parity Results Postprocessing Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Pair | Seeds | Raw Wilcoxon p (2s) | Dist Wilcoxon p (2s) | Raw mean diff (MJX-EV) | Dist mean diff (MJX-EV) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for row in results:
        raw = row["raw_metrics"]
        dist = row.get("distance_metrics")
        dist_p = "n/a" if dist is None else f"{dist['wilcoxon_two_sided_p']:.6g}"
        dist_diff = "n/a" if dist is None else f"{dist['mean_diff_left_minus_right']:.6g}"
        lines.append(
            "| "
            f"{row['pair_name']} | {raw['n']} | {raw['wilcoxon_two_sided_p']:.6g} | "
            f"{dist_p} | {raw['mean_diff_left_minus_right']:.6g} | {dist_diff} |"
        )

    lines.append("")
    lines.append("## Per-pair Details")
    lines.append("")

    for row in results:
        lines.append(f"### {row['pair_name']}")
        lines.append("")
        lines.append(f"- Seeds paired: {row['raw_metrics']['n']}")
        lines.append("- Raw hypothesis: location shift (MJX < EV expected for minimization)")
        lines.append(
            f"- Raw p-values: Wilcoxon(2-sided)={row['raw_metrics']['wilcoxon_two_sided_p']:.6g}, "
            f"Wilcoxon(1-sided)={row['raw_metrics']['wilcoxon_one_sided_p']:.6g}, "
            f"t(2-sided)={row['raw_metrics']['paired_t_two_sided_p']:.6g}"
        )
        lines.append(
            f"- Raw effect sizes: cohen_dz={row['raw_metrics']['cohen_dz']:.6g}, "
            f"rank_biserial={row['raw_metrics']['rank_biserial']}"
        )

        dist = row.get("distance_metrics")
        if dist is None:
            lines.append("- Distance-to-optimum: unavailable (no gap metric and no --optimum)")
        else:
            lines.append("- Distance hypothesis: closer to optimum (MJX distance < EV distance)")
            lines.append(
                f"- Distance p-values: Wilcoxon(2-sided)={dist['wilcoxon_two_sided_p']:.6g}, "
                f"Wilcoxon(1-sided)={dist['wilcoxon_one_sided_p']:.6g}, "
                f"t(2-sided)={dist['paired_t_two_sided_p']:.6g}"
            )
            lines.append(
                f"- Distance effect sizes: cohen_dz={dist['cohen_dz']:.6g}, "
                f"rank_biserial={dist['rank_biserial']}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess existing parity result folders")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-dir", default="results/parity_postprocess", help="Output directory")
    parser.add_argument(
        "--min-paired-seeds",
        type=int,
        default=10,
        help="Skip pair groups with fewer than this many common seeds",
    )
    parser.add_argument(
        "--optimum",
        type=float,
        default=None,
        help="Known optimum for distance calculation when gap_to_optimum is missing",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pipeline_pairs(results_dir)
    if not pairs:
        raise RuntimeError(f"No malthusjax/evosax paired result folders found under {results_dir}")

    all_results: list[dict[str, Any]] = []

    for mj_dir, ev_dir, pair_key in pairs:
        mj = load_pipeline_data(mj_dir)
        ev = load_pipeline_data(ev_dir)

        common_seeds = sorted(set(mj.final_best_by_seed) & set(ev.final_best_by_seed))
        if len(common_seeds) < args.min_paired_seeds:
            continue

        mj_raw = np.array([mj.final_best_by_seed[s] for s in common_seeds], dtype=float)
        ev_raw = np.array([ev.final_best_by_seed[s] for s in common_seeds], dtype=float)
        raw_metrics = paired_stats(mj_raw, ev_raw, direction="left_lt_right")

        # Prefer explicit gap_to_optimum when available.
        gap_seeds = sorted(set(mj.gap_by_seed) & set(ev.gap_by_seed))
        distance_metrics: dict[str, Any] | None = None
        distance_seed_source = "summary_gap_to_optimum"

        if len(gap_seeds) >= args.min_paired_seeds:
            mj_dist = np.array([mj.gap_by_seed[s] for s in gap_seeds], dtype=float)
            ev_dist = np.array([ev.gap_by_seed[s] for s in gap_seeds], dtype=float)
            distance_metrics = paired_stats(mj_dist, ev_dist, direction="left_lt_right")
        elif args.optimum is not None and len(common_seeds) >= args.min_paired_seeds:
            mj_dist = np.abs(mj_raw - float(args.optimum))
            ev_dist = np.abs(ev_raw - float(args.optimum))
            distance_metrics = paired_stats(mj_dist, ev_dist, direction="left_lt_right")
            distance_seed_source = "computed_from_raw_with_optimum"

        all_results.append(
            {
                "pair_key": pair_key,
                "pair_name": pair_key.replace("{backend}", "[malthusjax|evosax]"),
                "malthusjax_dir": str(mj_dir),
                "evosax_dir": str(ev_dir),
                "raw_metrics": raw_metrics,
                "distance_metrics": distance_metrics,
                "distance_seed_source": distance_seed_source if distance_metrics is not None else None,
                "n_common_seeds_raw": len(common_seeds),
                "n_common_seeds_distance": (distance_metrics or {}).get("n", 0),
            }
        )

    output_json = output_dir / "parity_postprocess.json"
    output_md = output_dir / "parity_postprocess.md"

    payload = {
        "results_dir": str(results_dir),
        "pair_count": len(all_results),
        "pairs": all_results,
    }

    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown_report(all_results))

    print(f"Pairs analyzed: {len(all_results)}")
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_md}")


if __name__ == "__main__":
    main()
