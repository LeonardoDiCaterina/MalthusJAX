from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from .results import ExperimentResult


def write_summary_json(experiment_result: ExperimentResult, path: Path | str) -> None:
    """Write ExperimentResult as summary.json with atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename
    temp_path = path.with_suffix('.json.tmp')
    try:
        with temp_path.open('w') as f:
            json.dump(experiment_result.to_dict(), f, indent=2)
        temp_path.rename(path)
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def read_summary_json(path: Path | str) -> ExperimentResult:
    """Read ExperimentResult from summary.json."""
    path = Path(path)
    with path.open('r') as f:
        data = json.load(f)
    return ExperimentResult.from_dict(data)


def write_histories_csv(experiment_result: ExperimentResult, path: Path | str) -> None:
    """Write combined histories as CSV with seed column (tidy format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    combined = experiment_result.combined_history()
    if not combined:
        # Write empty CSV with just headers
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed'])
        return

    # Get all unique keys across all rows for header
    all_keys: set[str] = set()
    for row in combined:
        all_keys.update(row.keys())

    # Ensure 'seed' comes first
    headers = ['seed'] + sorted(k for k in all_keys if k != 'seed')

    temp_path = path.with_suffix('.csv.tmp')
    try:
        with temp_path.open('w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=headers)
            dict_writer.writeheader()
            dict_writer.writerows(combined)
        temp_path.rename(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_seed_folder(output_dir: Path | str, seed: int) -> Path:
    """Create and return path for seed-specific artifacts."""
    output_dir = Path(output_dir)
    seed_dir = output_dir / f"seed_{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def write_experiment_artifacts(
    experiment_result: ExperimentResult,
    output_dir: Path | str,
    write_csv: bool = True,
    write_json: bool = True,
) -> Dict[str, Path]:
    """Write all experiment artifacts and return paths written."""
    output_dir = Path(output_dir)
    written_paths = {}

    if write_json:
        json_path = output_dir / "summary.json"
        write_summary_json(experiment_result, json_path)
        written_paths["summary_json"] = json_path

    if write_csv:
        csv_path = output_dir / "histories_combined.csv"
        write_histories_csv(experiment_result, csv_path)
        written_paths["histories_csv"] = csv_path

    # Create seed folders for future per-seed artifacts
    for run in experiment_result.runs:
        seed_dir = ensure_seed_folder(output_dir, run.seed)
        written_paths[f"seed_{run.seed:04d}"] = seed_dir

    return written_paths
