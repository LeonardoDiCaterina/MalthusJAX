"""Run HLO analysis over a directory of HLO text files.

Usage example:
    python run_analysis.py --dir ../ --pattern "*hlo*txt" --outdir ./results

The script finds HLO text files, runs `hlo_toolkit.analyze_files()` and
writes a CSV and JSON summary to the output directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import csv
from typing import List

from hlo_analysis.hlo_toolkit import analyze_files, summarize  # type: ignore


def write_csv(results: List[dict], outpath: str) -> None:
    if not results:
        print("No results to write to CSV.")
        return

    # Determine fieldnames from union of keys
    keys = set()
    for r in results:
        keys.update(r.keys())
    fieldnames = sorted(keys)

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory to search for HLO files")
    parser.add_argument("--pattern", default="*hlo*.txt", help="Glob pattern for HLO files")
    parser.add_argument("--outdir", default="./hlo_analysis_results", help="Directory to write outputs")
    args = parser.parse_args()

    search_path = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        print(f"No files found for pattern: {search_path}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Found {len(files)} HLO files. Running analysis...")
    results = analyze_files(files)

    # Write per-file JSON
    json_out = os.path.join(args.outdir, "hlo_analysis_per_file.json")
    with open(json_out, "w") as jf:
        json.dump(results, jf, indent=2)

    # Write CSV summary
    csv_out = os.path.join(args.outdir, "hlo_analysis_summary.csv")
    write_csv(results, csv_out)

    # Write aggregate summary
    agg = summarize(results)
    agg_out = os.path.join(args.outdir, "hlo_analysis_aggregate.json")
    with open(agg_out, "w") as af:
        json.dump(agg, af, indent=2)

    print("Analysis complete:")
    print(f"  Per-file JSON: {json_out}")
    print(f"  CSV summary : {csv_out}")
    print(f"  Aggregate    : {agg_out}")


if __name__ == "__main__":
    main()
