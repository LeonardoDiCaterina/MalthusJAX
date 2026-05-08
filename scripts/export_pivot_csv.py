#!/usr/bin/env python3
"""Generate clean CSV pivot tables for strategy rankings across fitness functions.

Output:
  - pivot_fitness_by_function.csv: median best_fitness table
  - pivot_timing_by_function.csv: median runtime table
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


def load_summary_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_median(values: List[float]) -> float:
    if not values:
        return float("nan")
    return median(values)


def _fitness_sign_for_name(name: str) -> float:
    return -1.0 if "evosax" in name.lower() else 1.0


def extract_metrics_from_summary(summary_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (median_best_fitness, median_duration_s) or (None, None) on failure."""
    summary = load_summary_json(summary_path)
    if summary is None:
        return None, None

    runs = summary.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return None, None

    best_vals: List[float] = []
    duration_vals: List[float] = []
    
    strategy_name = summary_path.parent.name
    fitness_sign = _fitness_sign_for_name(strategy_name)

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
        return None, None

    return safe_median(best_vals), safe_median(duration_vals)


def collect_results(base_dir: Path) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """Collect {fitness_function: {strategy: (median_best, median_time)}}."""
    results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}

    if not base_dir.exists():
        return results

    all_dirs = list(base_dir.iterdir())
    print(f"Scanning {len(all_dirs)} top-level result directories...", flush=True)
    
    for idx, fitness_func_dir in enumerate(sorted(all_dirs)):
        if idx % 5 == 0:
            print(f"  ... {idx}/{len(all_dirs)}", flush=True)
        
        if not fitness_func_dir.is_dir():
            continue

        fitness_func_name = fitness_func_dir.name
        results[fitness_func_name] = {}

        try:
            strategy_dirs = list(fitness_func_dir.iterdir())
        except PermissionError:
            continue

        for strategy_dir in sorted(strategy_dirs):
            if not strategy_dir.is_dir():
                continue

            summary_json = strategy_dir / "summary.json"
            if not summary_json.exists():
                continue

            strategy_name = strategy_dir.name
            median_best, median_time = extract_metrics_from_summary(summary_json)
            results[fitness_func_name][strategy_name] = (median_best, median_time)

    return results


def extract_strategy_name(full_name: str) -> str:
    """Normalize strategy names from paths like 'sphere_malthusjax_default'."""
    # Extract just the tail part after first underscore (if present)
    parts = full_name.split("_", 1)
    if len(parts) == 2 and "_" in parts[1]:
        # E.g., "sphere_malthusjax_default" → "malthusjax_default"
        return parts[1]
    return full_name


def write_fitness_csv(
    results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]], output_path: Path
) -> None:
    """Write median best_fitness pivot table to CSV (fitness function × strategy)."""
    all_strategies: set[str] = set()
    for fitness_func_data in results.values():
        for strategy in fitness_func_data.keys():
            all_strategies.add(extract_strategy_name(strategy))

    strategies = sorted(all_strategies)
    fitness_functions = sorted(results.keys())

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fitness Function"] + strategies)

        for fitness_func in fitness_functions:
            row = [fitness_func]
            for strategy in strategies:
                found = False
                for full_strategy_name in results[fitness_func].keys():
                    if extract_strategy_name(full_strategy_name) == strategy:
                        best_fit, _ = results[fitness_func][full_strategy_name]
                        if best_fit is None or (isinstance(best_fit, float) and best_fit != best_fit):
                            row.append("")
                        else:
                            row.append(f"{best_fit:.4g}")
                        found = True
                        break
                if not found:
                    row.append("")
            writer.writerow(row)


def write_timing_csv(
    results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]], output_path: Path
) -> None:
    """Write median duration pivot table to CSV (fitness function × strategy)."""
    all_strategies: set[str] = set()
    for fitness_func_data in results.values():
        for strategy in fitness_func_data.keys():
            all_strategies.add(extract_strategy_name(strategy))

    strategies = sorted(all_strategies)
    fitness_functions = sorted(results.keys())

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fitness Function"] + strategies)

        for fitness_func in fitness_functions:
            row = [fitness_func]
            for strategy in strategies:
                found = False
                for full_strategy_name in results[fitness_func].keys():
                    if extract_strategy_name(full_strategy_name) == strategy:
                        _, duration = results[fitness_func][full_strategy_name]
                        if duration is None or (isinstance(duration, float) and duration != duration):
                            row.append("")
                        else:
                            row.append(f"{duration:.4g}")
                        found = True
                        break
                if not found:
                    row.append("")
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV pivot tables for strategy rankings")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("~/Downloads/MalthusJAX_results").expanduser(),
        help="Root folder containing downloaded result directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp"),
        help="Output directory for CSV files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = collect_results(args.results_root)

    if not results:
        print("No results found.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fitness_csv = args.output_dir / "pivot_fitness_by_function.csv"
    timing_csv = args.output_dir / "pivot_timing_by_function.csv"

    print(f"\nWriting fitness CSV: {fitness_csv}")
    write_fitness_csv(results, fitness_csv)

    print(f"Writing timing CSV: {timing_csv}")
    write_timing_csv(results, timing_csv)

    print("\nDone! CSV files ready for import into Excel or analysis tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
