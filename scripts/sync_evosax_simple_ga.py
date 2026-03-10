#!/usr/bin/env python3
"""Utility to snapshot the current evosax SimpleGA implementation.

The benchmark suite depends on ``evosax.algorithms.population_based.simple_ga``
for the ``SimpleGA`` class.  Unfortunately the upstream package does not
expose a stable ``__version__`` and its API may change without warning.  To
help detect such changes we periodically copy the source file into our own
repository so that it can be diffed and reviewed during upgrades.

Run this script from the workspace root:

    python scripts/sync_evosax_simple_ga.py

It will locate the installed module, display its path, and write a local
snapshot to ``scripts/evosax_simple_ga_snapshot.py``.  If the file already
exists you can diff it against the new copy to see if anything changed.
"""

import importlib.util
import inspect
import os
import sys

MODULE_NAME = "evosax.algorithms.population_based.simple_ga"
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "evosax_simple_ga_snapshot.py")


def main():
    try:
        spec = importlib.util.find_spec(MODULE_NAME)
        if spec is None or spec.origin is None:
            raise ImportError(f"cannot locate module {MODULE_NAME}")
        source_path = spec.origin
    except Exception as exc:
        print(f"failed to locate {MODULE_NAME}: {exc}")
        sys.exit(1)

    print(f"found evosax SimpleGA source at: {source_path}")
    try:
        with open(source_path, "r") as f:
            src = f.read()
    except Exception as exc:
        print(f"error reading source file: {exc}")
        sys.exit(1)

    with open(SNAPSHOT_PATH, "w") as f:
        f.write("# snapshot of evosax SimpleGA taken from " + source_path + "\n")
        f.write(src)

    print(f"snapshot written to {SNAPSHOT_PATH}")

    # quick sanity checks on contents
    import evosax.algorithms.population_based.simple_ga as sg
    print("SimpleGA class:", sg.SimpleGA)
    print("has crossover fn:", hasattr(sg, "crossover"))
    print("has mutation fn:", hasattr(sg, "mutation"))
    try:
        import inspect
        sig = inspect.signature(sg.crossover)
        print("crossover signature:", sig)
    except Exception:
        pass
    try:
        sig = inspect.signature(sg.mutation)
        print("mutation signature:", sig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
