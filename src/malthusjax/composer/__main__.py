"""CLI entry point for malthusjax.composer.

Usage:
    python -m malthusjax.composer configs/lhs_experiments/hyp1_sphere_lhs0.toml
"""

from __future__ import annotations

import sys

from malthusjax.composer import Composer


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m malthusjax.composer <config.toml> [pipeline1 pipeline2 ...]")
        sys.exit(1)

    toml_path = sys.argv[1]
    pipelines = sys.argv[2:] if len(sys.argv) > 2 else None

    print(f"Executing Composer configuration from: {toml_path}")
    result = Composer.from_toml(toml_path, pipelines=pipelines)
    print("\nExecution Completed Successfully!")
    print(result.summary() if hasattr(result, "summary") else str(result))


if __name__ == "__main__":
    main()
