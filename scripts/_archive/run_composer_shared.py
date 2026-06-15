#!/usr/bin/env python3
"""Run Composer experiments from TOML but enforce shared initial population
and allow strategy-parameter overrides to ensure exact param mapping.

Usage:
  python scripts/run_composer_shared.py path/to/experiment.toml [--pop-seed 123] [--overrides overrides.json]

If `--overrides` is provided and points to a JSON file, its top-level keys
are merged into each pipeline's `strategy_params` table (for evosax) or
into the pipeline dict used for `Composer.compare`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib as toml
except Exception:
    import tomli as toml  # type: ignore

from malthusjax.composer import Composer


def load_toml(path: Path):
    with path.open("rb") as f:
        return toml.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to TOML experiment file")
    p.add_argument("--pop-seed", type=int, default=123)
    p.add_argument("--shared-initial-population", action="store_true", default=True)
    p.add_argument("--no-shared-initial-population", dest="shared_initial_population", action="store_false")
    p.add_argument("--overrides", type=str, default=None, help="JSON file with strategy param overrides")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    tom = load_toml(config_path)
    exp = tom.get("experiment", {})
    shared = exp.get("shared", {})
    # pipelines are stored at top-level under the `pipelines` table
    pipelines_tbl = tom.get("pipelines", {})

    # assemble shared kwargs to pass to Composer.compare
    shared_kwargs = {}
    # pass common quick_run kwargs
    for key in ("pop_size", "generations", "genome_length", "bounds", "use_history_for_final", "track_best"):
        if key in shared:
            shared_kwargs[key] = shared[key]
    # fitness in pipelines may override; let pipeline-specific value remain

    # load overrides if provided
    overrides = {}
    if args.overrides:
        ovp = Path(args.overrides)
        if not ovp.exists():
            print(f"Overrides file not found: {ovp}")
            sys.exit(1)
        with ovp.open("r") as f:
            overrides = json.load(f)

    pipelines = {}
    for name, tbl in pipelines_tbl.items():
        # copy known keys directly; unknown keys are passed through
        pd = {}
        for k, v in tbl.items():
            pd[k] = v
        # if evosax strategy and overrides provided, merge into strategy_params
        if overrides:
            sp = pd.get("strategy_params") or {}
            # ensure we have a dict
            if not isinstance(sp, dict):
                sp = dict(sp)
            # merge overrides
            sp.update(overrides)
            pd["strategy_params"] = sp
        pipelines[name] = pd

    composer = Composer.create_default()
    seeds = tuple(shared.get("seeds", ()))
    if not seeds:
        # fallback to small sample if not provided
        seeds = (0, 1, 2)

    print(f"Running Composer.compare on pipelines: {list(pipelines.keys())}")
    comp = composer.compare(
        pipelines=pipelines,
        seeds=seeds,
        shared_initial_population=args.shared_initial_population,
        pop_seed=args.pop_seed,
        **shared_kwargs,
    )

    print("=== Summary ===")
    try:
        print(comp.summary_table())
    except Exception:
        print("(no summary_table available)")

    # write artifacts if the Composition provided a convenience
    try:
        outdir = Path(exp.get("output_dir", "results/"))
        outdir.mkdir(parents=True, exist_ok=True)
        # minimal report
        report = {
            "pipelines": {},
        }
        for name, res in comp.pipelines.items():
            agg = res.aggregated_summary()
            report["pipelines"][name] = agg
        with (outdir / "composer_shared_report.json").open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote composer_shared_report.json to {outdir}")
    except Exception as e:
        print(f"Failed to write report: {e}")


if __name__ == "__main__":
    main()
