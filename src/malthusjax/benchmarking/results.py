"""Data structures for storing and manipulating benchmark outcomes.

This module defines :class:`RunResult` and :class:`ExperimentResult` which
record per-seed histories, summary metrics, errors, and timing information.
Utilities for serialization/deserialization and simple aggregation are
provided, along with :class:`ComparisonResult` for aligning multiple
pipelines.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RunResult:
    """Immutable record of a single seeded engine execution.

    Instances capture the seed value, success status, computed numeric
    metrics and generation history, plus optional timing data, artifact paths
    and error messages.  A UTC timestamp is recorded on creation.
    """

    seed: int
    status: str  # e.g., "success", "failure", "timeout", "error"
    metrics: Dict[str, float]
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    duration_seconds: Optional[float] = None
    timings: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the object to a JSON-friendly dictionary.

        The ISO-formatted ``created_at`` timestamp is included directly.
        """
        return {
            "seed": self.seed,
            "status": self.status,
            "metrics": self.metrics,
            "history": self.history,
            "artifacts": self.artifacts,
            "duration_seconds": self.duration_seconds,
            "timings": self.timings,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        """Reconstruct a :class:`RunResult` from a dictionary produced by
        :meth:`to_dict`.

        The *created_at* field is parsed from ISO format if present, or
        defaulted to the current UTC time.
        """
        d = dict(data)
        created = d.get("created_at")
        if isinstance(created, str):
            d["created_at"] = datetime.fromisoformat(created)
        elif created is None:
            d["created_at"] = datetime.now(timezone.utc)
        return cls(
            seed=d["seed"],
            status=d["status"],
            metrics=d.get("metrics", {}),
            history=d.get("history", []),
            artifacts=d.get("artifacts", {}),
            duration_seconds=d.get("duration_seconds"),
            timings=d.get("timings"),
            error=d.get("error"),
            created_at=d["created_at"],
        )

    @classmethod
    def from_json(cls, data: str) -> "RunResult":
        """Create a :class:`RunResult` from a JSON string."""
        return cls.from_dict(json.loads(data))


@dataclass
class ExperimentResult:
    name: str
    runs: List[RunResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "0.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": [r.to_dict() for r in self.runs],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        d = dict(data)
        created = d.get("created_at")
        if isinstance(created, str):
            d["created_at"] = datetime.fromisoformat(created)
        elif created is None:
            d["created_at"] = datetime.now(timezone.utc)
        runs = [RunResult.from_dict(r) for r in d.get("runs", [])]
        return cls(
            name=d["name"],
            runs=runs,
            metadata=d.get("metadata", {}),
            created_at=d["created_at"],
            schema_version=d.get("schema_version", "0.1"),
        )

    def combined_history(self, seed_field: str = "seed") -> List[Dict[str, Any]]:
        """Concatenate all run histories into a single list.

        Each row receives an extra key (default ``"seed"``) identifying the
        originating seed.  This is convenient for writing CSV or pandas
        tables.
        """
        rows: List[Dict[str, Any]] = []
        for run in self.runs:
            for row in run.history:
                combined = {**row, seed_field: run.seed}
                rows.append(combined)
        return rows

    @property
    def canonical_summary(self) -> Dict[str, Any]:
        if not self.runs:
            return {}
        return self.runs[0].metrics

    def aggregated_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute mean/median/stddev for each metric across runs.

        Non-numeric values are ignored.  The result maps each metric name to a
        small dict containing ``mean``, ``median`` and ``stdev`` (zero when
        only one value is available).
        """
        # Collect numeric metrics across runs
        agg: Dict[str, List[float]] = {}
        for r in self.runs:
            for k, v in r.metrics.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                agg.setdefault(k, []).append(val)

        summary: Dict[str, Dict[str, float]] = {}
        for k, vals in agg.items():
            if not vals:
                continue
            mean = statistics.mean(vals)
            med = statistics.median(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[k] = {"mean": mean, "median": med, "stdev": stdev}
        return summary

# ---------------------------------------------------------------------------
# ComparisonResult — aligned multi-pipeline results
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Aligned results from multiple pipelines for direct comparison.

    Returned by :meth:`Composer.compare` and :meth:`Composer.from_toml`.

    Attributes
    ----------
    pipelines : Dict[str, ExperimentResult]
        Mapping of pipeline name → :class:`ExperimentResult`.
    shared_config : Dict[str, Any]
        The shared configuration that was common across all pipelines.
    initial_population : optional
        The shared initial population array, if one was generated.
    """

    pipelines: Dict[str, ExperimentResult]
    shared_config: Dict[str, Any] = field(default_factory=dict)
    initial_population: Optional[Any] = field(default=None, repr=False)
    negate_map: Dict[str, bool] = field(default_factory=dict)
    """Per-pipeline sign-flip flag.

    When ``True`` for a pipeline, fitness values are **negated** before
    display so that all pipelines share a unified "lower is better"
    convention.  Built automatically by :meth:`Composer.compare` based
    on each pipeline's backend (MalthusJAX uses a maximisation
    convention internally, so its values are negated for display).
    """

    # -- internal helper ---------------------------------------------------

    _FITNESS_KEYS = frozenset({"best_fitness", "mean_fitness"})

    def _sign(self, pipeline_name: str) -> float:
        """Return ``-1.0`` if *pipeline_name* should be negated, else ``1.0``."""
        return -1.0 if self.negate_map.get(pipeline_name, False) else 1.0

    # -- Convenience accessors --------------------------------------------

    @property
    def names(self) -> List[str]:
        """Pipeline names in insertion order."""
        return list(self.pipelines.keys())

    def summary_table(self) -> Dict[str, Dict[str, float]]:
        """Per-pipeline aggregated summary.

        Returns ``{pipeline_name: {metric: value, ...}, ...}`` using the
        mean across seeds for each metric.  Fitness values are
        automatically sign-normalised so that **lower is better** across
        all pipelines.
        """
        table: Dict[str, Dict[str, float]] = {}
        for name, exp in self.pipelines.items():
            agg = exp.aggregated_summary()
            s = self._sign(name)
            # Flatten to the mean; negate fitness keys if needed
            table[name] = {
                k: (v["mean"] * s if k in self._FITNESS_KEYS else v["mean"])
                for k, v in agg.items()
            }
        return table

    def convergence_data(
        self, seed_index: int = 0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Per-pipeline convergence history for a single seed.

        Fitness values are sign-normalised so that **lower is better**
        across all pipelines (controlled by :attr:`negate_map`).

        The *seed_index* argument selects which seed's history is extracted
        (defaulting to the first).  The returned mapping contains one list of
        generation/fitness dictionaries per pipeline, with fitness values
        already adjusted according to ``negate_map``.
        """
        data: Dict[str, List[Dict[str, Any]]] = {}
        for name, exp in self.pipelines.items():
            if seed_index < len(exp.runs):
                raw = exp.runs[seed_index].history
                s = self._sign(name)
                if s < 0:
                    data[name] = [
                        {
                            k: (v * s if k in self._FITNESS_KEYS else v)
                            for k, v in row.items()
                        }
                        for row in raw
                    ]
                else:
                    data[name] = raw
            else:
                data[name] = []
        return data

    def plot_convergence(
        self,
        seed_index: int = 0,
        ax: Any = None,
        title: Optional[str] = None,
        negate: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """Overlay convergence curves on a matplotlib axis.

        Sign normalisation from :attr:`negate_map` is applied
        **automatically** (via :meth:`convergence_data`).  The *negate*
        parameter adds an **extra** per-pipeline flip on top, useful for
        switching between "lower is better" and "higher is better"
        display after the automatic normalisation.

        Calling this method produces an overlaid convergence plot for the
        chosen *seed_index* (first seed by default).  If *ax* is omitted a new
        figure and axis are created.  The *title* parameter overrides the
        default ``"Convergence comparison"`` string.  An additional *negate*
        mapping may be supplied to flip any pipeline's curve after the built-in
        normalisation.  The Matplotlib axis object containing the plot is
        returned.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_convergence(). "
                "Install it with: pip install matplotlib"
            ) from e

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        extra_negate = negate or {}
        conv = self.convergence_data(seed_index)  # already sign-normalised

        for name, history in conv.items():
            if not history:
                continue
            gens = [h["generation"] for h in history]
            best = [h["best_fitness"] for h in history]
            if extra_negate.get(name, False):
                best = [-b for b in best]
            ax.plot(gens, best, label=name, linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title(title or "Convergence comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
