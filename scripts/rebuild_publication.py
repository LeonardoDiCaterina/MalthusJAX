#!/usr/bin/env python3
"""Inventory claims or rebuild review-only tables from explicit raw manifests."""

import argparse
import json
from pathlib import Path

from malthusjax.benchmarking.publication import inventory, rebuild


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-root", type=Path)
    p.add_argument("--left-manifest", type=Path)
    p.add_argument("--right-manifest", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.paper_root:
        data = inventory(a.paper_root)
        if not a.dry_run:
            with a.output.open("x") as f:
                json.dump(data, f, indent=2)
    else:
        if not all((a.left_manifest, a.right_manifest, a.policy)):
            p.error("Rebuilding requires both manifests and an explicit statistical policy")
        data = rebuild(a.left_manifest, a.right_manifest, a.policy, a.output, dry_run=a.dry_run)
    print(json.dumps({"status": data.get("status", "inventory_only"), "dry_run": a.dry_run}))


if __name__ == "__main__":
    main()
