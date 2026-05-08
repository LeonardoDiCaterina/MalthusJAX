#!/usr/bin/env python3
"""Rank experiment summaries by median result and median runtime.

This script walks result directories, loads each `summary.json`, computes
median `best_fitness` and median `duration_seconds`, then prints rankings.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class SummaryRank:
    name: str
    summary_path: Path
    median_best_fitness: float
    median_duration_s: float
    num_runs: int


def load_summary_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_median(values: Iterable[float]) -> float:
    values_list = [float(v) for v in values if v is not None]
    return median(values_list) if values_list else float("nan")


def _fitness_sign_for_name(name: str) -> float:
    return -1.0 if "evosax" in name.lower() else 1.0


def collect_summary_rank(name: str, summary_path: Path) -> Optional[SummaryRank]:
    summary = load_summary_json(summary_path)
    if summary is None:
        return None

    runs = summary.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return None

    best_vals: List[float] = []
    duration_vals: List[float] = []
    fitness_sign = _fitness_sign_for_name(name)
    for run in runs:
        metrics = run.get("metrics", {})
        if isinstance(metrics, dict):
            best = metrics.get("best_fitness")
            if best is not None:
                try:
                    best_vals.append(float(best) * fitness_sign)
                except ValueError:
                    pass
        duration = run.get("duration_seconds")
        if duration is not None:
            try:
                duration_vals.append(float(duration))
            except ValueError:
                pass

    if not best_vals or not duration_vals:
        return None

    return SummaryRank(
        name=name,
        summary_path=summary_path,
        median_best_fitness=safe_median(best_vals),
        median_duration_s=safe_median(duration_vals),
        num_runs=len(runs),
    )


def find_summary_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and (p / "summary.json").exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank experiment summaries by median fitness and runtime")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("~/Downloads/MalthusJAX_results").expanduser(),
        help="Root folder containing downloaded experiment result directories",
    )
    parser.add_argument(
        "--num-functions",
        type=int,
        default=5,
        help="Number of random experiment directories to sample for ranking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.results_root
    all_dirs = find_summary_dirs(base_dir)
    if not all_dirs:
        print(f"No result directories with summary.json found under: {base_dir}")
        return 1

    sample_count = min(args.num_functions, len(all_dirs))
    random.seed(args.seed)
    selected_dirs = random.sample(all_dirs, sample_count)

    ranks: List[SummaryRank] = []
    for d in selected_dirs:
        rank = collect_summary_rank(d.name, d / "summary.json")
        if rank is not None:
            ranks.append(rank)

    if not ranks:
        print("No valid summaries found in the selected directories.")
        return 1

    ranks.sort(key=lambda x: x.median_best_fitness, reverse=True)
    print("Ranking by median best_fitness (descending):")
    print(
        "{:<28} {:>8} {:>16} {:>14} {:>12}".format(
            "experiment",
            "runs",
            "median_best",
            "median_time(s)",
            "path",
        )
    )
    print("-" * 90)
    for rank in ranks:
        print(
            "{:<28} {:>8d} {:>16.6g} {:>14.6g} {:>12s}".format(
                rank.name,
                rank.num_runs,
                rank.median_best_fitness,
                rank.median_duration_s,
                str(rank.summary_path.parent),
            )
        )

    ranks_by_time = sorted(ranks, key=lambda x: x.median_duration_s)
    print("\nRanking by median runtime (ascending):")
    print(
        "{:<28} {:>8} {:>16} {:>14} {:>12}".format(
            "experiment",
            "runs",
            "median_best",
            "median_time(s)",
            "path",
        )
    )
    print("-" * 90)
    for rank in ranks_by_time:
        print(
            "{:<28} {:>8d} {:>16.6g} {:>14.6g} {:>12s}".format(
                rank.name,
                rank.num_runs,
                rank.median_best_fitness,
                rank.median_duration_s,
                str(rank.summary_path.parent),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
