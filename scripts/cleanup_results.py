#!/usr/bin/env python3
"""Cleanup and organize the MalthusJAX results/ directory.

Usage:
    # Preview what will be cleaned without making changes (safe):
    python scripts/cleanup_results.py --dry-run

    # Perform the cleanup:
    python scripts/cleanup_results.py --execute

    # Optional: also archive large historical benchmark campaigns into results/_archive/
    python scripts/cleanup_results.py --execute --archive-benchmarks
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Directories and files to NEVER touch
PROTECTED_ITEMS = {
    "empirical_curves.json",
    "tables",
    "logs",
    "showcase",
    "showcases",
    "_archive",
    ".gitkeep",
    ".gitignore",
}

# Known temporary / scratch debug folders to purge
SCRATCH_NAMES = {
    "A", "B", "alg1", "alg2", "p1", "pipe1", "pipe2", "pipe_a", "pipe_b",
    "pipeline_alpha", "roul", "tourn", "simple", "test", "test_binary",
    "test_lightweight", "test_qdax_smoke", "test_run", "test_shapes",
    "toy_gap_parity", "toy_gap_parity_4056", "mjx_test", "quick_experiment",
    "engine_type_test", "es", "evosax_ga", "evosax_manual", "evosax_simple",
    "ga", "native", "rl", "perf", "pure_native_test", "smoke_tensorneat",
    "sphere_experiment", "sphere_bench", "neat_speciation_2",
}

# Large historical campaigns to organize or archive
BENCHMARK_PREFIXES = (
    "h1_", "h2_", "h3_", "mjx_ablate_", "mjx_f", "mjx_bf", "baseline_",
    "evosax_baseline", "malthusjax_wrapper", "malthusjax_batched", "fitness_parity",
)


def get_dir_size_mb(path: Path) -> float:
    """Calculate directory size in megabytes."""
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Clean and organize MalthusJAX results/ folder.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Show planned actions without deleting files.")
    parser.add_argument("--execute", action="store_true", help="Actually execute the cleanup.")
    parser.add_argument("--archive-benchmarks", action="store_true", help="Move large historical benchmark folders into results/_archive/.")
    parser.add_argument("--results-dir", type=str, default="results", help="Target results directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return

    is_dry_run = not args.execute

    print("=" * 75)
    print(f"  MALTHUSJAX RESULTS CLEANUP (Dry-run: {is_dry_run})")
    print(f"  Target: {results_dir}")
    print("=" * 75)

    entries = sorted(list(results_dir.iterdir()))
    to_delete = []
    to_archive = []
    protected = []

    for item in entries:
        name = item.name
        if name in PROTECTED_ITEMS or name.startswith("."):
            protected.append(item)
            continue

        # Scratch or debug throwaway
        if (
            name in SCRATCH_NAMES
            or name.startswith("seed_")
            or name.startswith("pipe")
            or name.startswith("test_")
            or name.startswith("demo_")
        ):
            to_delete.append(item)
            continue

        # Benchmark campaigns
        if any(name.startswith(p) for p in BENCHMARK_PREFIXES) or name in {"profiling", "lsp"}:
            if args.archive_benchmarks:
                to_archive.append(item)
            else:
                protected.append(item)
            continue

        # Other unspecified scratch dirs
        if item.is_dir() and not any(p in name for p in ("showcase", "table", "curve")):
            to_delete.append(item)
        else:
            protected.append(item)

    # 1. Report Protected
    print(f"\n[KEEP] {len(protected)} items preserved:")
    for p in protected[:10]:
        print(f"  + {p.name}")
    if len(protected) > 10:
        print(f"  ... and {len(protected) - 10} more items.")

    # 2. Report & Execute Deletions
    del_size_mb = sum(get_dir_size_mb(item) for item in to_delete)
    print(f"\n[PURGE] {len(to_delete)} temporary/scratch items to delete (~{del_size_mb:.1f} MB):")
    for item in to_delete[:15]:
        print(f"  - {item.name}")
    if len(to_delete) > 15:
        print(f"  ... and {len(to_delete) - 15} more scratch items.")

    if not is_dry_run:
        for item in to_delete:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except OSError:
                    pass
        print(f"  >>> Purged {len(to_delete)} items, freed ~{del_size_mb:.1f} MB.")

    # 3. Report & Execute Archiving
    if args.archive_benchmarks and to_archive:
        archive_dir = results_dir / "_archive"
        arch_size_mb = sum(get_dir_size_mb(item) for item in to_archive)
        print(f"\n[ARCHIVE] {len(to_archive)} benchmark campaigns (~{arch_size_mb:.1f} MB) -> {archive_dir.name}/:")
        for item in to_archive:
            print(f"  -> {item.name}")
        if not is_dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for item in to_archive:
                dest = archive_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(item), str(dest))
            print(f"  >>> Moved {len(to_archive)} campaigns to {archive_dir}.")

    print("\n" + "=" * 75)
    if is_dry_run:
        print("  DRY-RUN COMPLETE. No files were modified.")
        print("  To execute, run:")
        print("    python scripts/cleanup_results.py --execute")
        print("    python scripts/cleanup_results.py --execute --archive-benchmarks")
    else:
        print("  CLEANUP COMPLETE. The results/ directory is now tidy!")
    print("=" * 75)


if __name__ == "__main__":
    main()
