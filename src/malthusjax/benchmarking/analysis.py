"""Utilities for parsing and summarising pytest-benchmark output files.

These helpers make it easy to consume the JSON produced by
``pytest --benchmark-save`` and to extract the timing statistics and any
extra information that the tests have recorded (for example the
``start_best_fitness``/``delta_best`` fields).

The module is intentionally lightweight and has no external dependencies
beyond the normal project requirements; pandas is used when available for
convenience but it is not required for the core parsing logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

try:
    import pandas
except ImportError:  # pragma: no cover - pandas is optional
    pandas = None


def load_benchmark_file(path: Path) -> Dict[str, Any]:
    """Load a pytest-benchmark JSON file.

    Parameters
    ----------
    path
        Path to the ``.json`` file stored by ``pytest --benchmark-save``.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    with open(path, "r") as f:
        return cast(Dict[str, Any], json.load(f))


def benchmarks_to_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert raw benchmark JSON into a list of flat records.

    Each element in the returned list corresponds to a single benchmark
    entry.  The ``stats`` dictionary is unpacked and, if ``extra_info`` is
    present, its contents are merged at the top level as well.
    """
    records: List[Dict[str, Any]] = []
    for bench in data.get("benchmarks", []):
        rec: Dict[str, Any] = {}
        rec["group"] = bench.get("group")
        rec["name"] = bench.get("name")
        stats = bench.get("stats", {})
        rec.update(stats)
        extra = bench.get("extra_info", {})
        rec.update(extra)
        records.append(rec)
    return records


def to_dataframe(data: Dict[str, Any]) -> pandas.DataFrame: # type: ignore
    """
    Return a :mod:`pandas` DataFrame containing all benchmarks.

    Requires :mod:`pandas`; raises ``ImportError`` if pandas is not
    installed.
    """
    if pandas is None:
        raise ImportError("pandas is required to convert benchmarks to DataFrame")
    return pandas.DataFrame(benchmarks_to_records(data))


def compute_grouped_kpis(data: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute simple KPI aggregates grouped by ``(group, name)``.

    The returned dictionary maps ``(group, name)`` tuples to a small
    dictionary of metrics such as ``mean`` and ``stddev``.  This helper
    is intended for lightweight use outside of pandas.
    """
    records = benchmarks_to_records(data)
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r.get("group"), r.get("name"))
        groups.setdefault(key, []).append(r)

    kpis: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, recs in groups.items():
        # compute simple aggregates, ignoring missing values
        vals: List[float] = [cast(float, r.get("mean")) for r in recs if r.get("mean") is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        # population stddev
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        kpis[cast(Tuple[str, str], key)] = {"mean": mean, "stddev": std, "count": len(vals)}
    return kpis


def sample_usage(path: Optional[Path] = None) -> None:
    """Example code demonstrating how to load and summarise one file.

    If ``path`` is omitted the function will attempt to locate
    ``.benchmarks/*/*.json`` in the repository root and use the first
    match.  This is purely educational and not used by tests.
    """
    if path is None:
        import glob

        matches = glob.glob(".benchmarks/*/*.json")
        if not matches:
            raise FileNotFoundError("no benchmark files found")
        path = Path(matches[0])

    data = load_benchmark_file(path)
    print(f"loaded {len(data.get('benchmarks', []))} entries from {path}")
    kpis = compute_grouped_kpis(data)
    for (group, name), stats in kpis.items():
        print(f"{group}/{name}: {stats}")


# plotting helpers are optional and only available if matplotlib is
# installed; we import lazily so that the module can still be imported by
# the test suite in environments without visualization dependencies.


def plot_group(group: str, data: Dict[str, Any], ax: Any | None = None) -> Any:
    """Draw a bar chart of mean timings for all benchmark names in ``group``.

    Parameters
    ----------
    group
        The benchmark group name (e.g. ``"single_step/pop100_d10"``).
    data
        Loaded benchmark JSON data.
    ax
        Optional ``matplotlib.axes.Axes`` to plot into; a new figure is
        created if ``None``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("matplotlib required for plotting")

    records = benchmarks_to_records(data)
    names: List[str] = []
    means: List[float] = []
    for r in records:
        if r.get("group") == group:
            name = r.get("name")
            mean = r.get("mean")
            if name is not None and mean is not None:
                names.append(cast(str, name))
                means.append(cast(float, mean))
    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(names, means)
    ax.set_title(group)
    ax.set_ylabel("mean (s)")
    return ax
