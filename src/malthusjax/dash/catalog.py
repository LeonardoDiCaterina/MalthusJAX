import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DataCatalog:
    """Manages the discovery, loading, and unified storage of benchmark results."""

    def __init__(self) -> None:
        self._sources: dict[str, Path] = {}
        self._data: pd.DataFrame | None = None

    def add_source(self, name: str, path: str | Path) -> None:
        """Register a source directory or file."""
        self._sources[name] = Path(path)

    @property
    def data(self) -> pd.DataFrame:
        """Access the loaded dataset. Raises ValueError if not loaded."""
        if self._data is None:
            raise ValueError("DataCatalog is empty. Call load() first.")
        return self._data

    def load(self) -> None:
        """Load all registered sources into a single unified DataFrame."""
        records: list[dict[str, Any]] = []

        search_patterns = [
            "benchmark_results.json",
            "parity_results.json",
            "ablation_results.json",
            "representation_results.json",
        ]

        for source_name, path in self._sources.items():
            if path.is_file():
                records.extend(self._parse_file(path, source_name))
            elif path.is_dir():
                for pattern in search_patterns:
                    for json_path in path.rglob(pattern):
                        records.extend(self._parse_file(json_path, source_name))

        if not records:
            self._data = pd.DataFrame(
                columns=[
                    "source",
                    "experiment",
                    "fn_name",
                    "D",
                    "P",
                    "G",
                    "pipeline",
                    "run_index",
                    "seed",
                    "best_fitness",
                    "execution_time",
                ]
            )
        else:
            self._data = pd.DataFrame(records)

    def _parse_file(self, file_path: Path, source_name: str) -> list[dict[str, Any]]:
        """Parse a single JSON artifact and extract flat records."""
        records = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        exp_name = data.get("experiment", "unknown")

        # Support both new TOML format (in `config`) and legacy format (top-level)
        config = data.get("config", data)

        fn_name = config.get("fn_name", config.get("function", "unknown"))
        D = config.get("D", config.get("dimensions", np.nan))
        P = config.get("P", config.get("population_size", np.nan))
        G = config.get("G", config.get("generations", np.nan))

        # Legacy Fallback: Extract from experiment string if missing
        if pd.isna(D) or pd.isna(P):
            if pd.isna(D):
                m = re.search(r"_d(\d+)_", exp_name)
                if m:
                    D = float(m.group(1))
            if pd.isna(P):
                m = re.search(r"_p(\d+)_", exp_name)
                if m:
                    P = float(m.group(1))
            if pd.isna(G):
                m = re.search(r"_g(\d+)", exp_name)
                if m:
                    G = float(m.group(1))

        pipelines = data.get("pipelines", {})
        for p_name, p_data in pipelines.items():
            # Support old legacy format where runs were directly in a list
            if isinstance(p_data, dict):
                runs = p_data.get("per_seed", p_data)
            else:
                runs = p_data

            if not isinstance(runs, list):
                runs = [runs]

            for run_index, run in enumerate(runs):
                seed = run.get("seed", -1)
                best_fit = run.get("best_fitness", np.nan)
                exec_time = run.get("duration_seconds", np.nan)

                if "timings" in run and "total" in run["timings"]:
                    exec_time = run["timings"]["total"]

                try:
                    best_fit = float(best_fit) if best_fit is not None else np.nan
                    exec_time = float(exec_time) if exec_time is not None else np.nan
                except (ValueError, TypeError):
                    best_fit, exec_time = np.nan, np.nan

                if np.isnan(best_fit) and np.isnan(exec_time):
                    continue

                records.append(
                    {
                        "source": source_name,
                        "experiment": exp_name,
                        "fn_name": fn_name,
                        "D": float(D) if D is not None else np.nan,
                        "P": float(P) if P is not None else np.nan,
                        "G": float(G) if G is not None else np.nan,
                        "pipeline": p_name,
                        "run_index": int(run_index),
                        "seed": int(seed),
                        "best_fitness": best_fit,
                        "execution_time": exec_time,
                    }
                )

        return records
