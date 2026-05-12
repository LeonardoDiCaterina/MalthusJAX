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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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

    @property
    def summary(self) -> Dict[str, float]:
        """Backward-compatible accessor returning the numeric metrics.

        Some callers expect a ``.summary`` attribute on run objects; expose
        it as a simple mapping to the already-stored ``metrics`` dict.
        """
        return self.metrics

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
    """Result object from a single experiment run with multiple seeds.

    Returned by :meth:`Composer.quick_run`, an :class:`ExperimentResult`
    aggregates multiple independent evolutionary runs (one per seed) and
    provides methods to extract summary statistics and convergence histories.

    Attributes
    ----------
    name : str
        Experiment identifier (usually the ``experiment_name`` param from
        :meth:`Composer.quick_run`).
    runs : List[RunResult]
        Individual run records (one :class:`RunResult` per seed).
    metadata : Dict[str, Any]
        Arbitrary experiment metadata (e.g., config, timestamps).
    created_at : datetime
        UTC timestamp of result creation.
    schema_version : str
        Version string for serialization format (default: "0.1").

    Notes
    -----
    **Typical Workflow**:

    1. Call :meth:`Composer.quick_run(...) <Composer.quick_run>` -> returns
       :class:`ExperimentResult`
    2. Call :meth:`aggregated_summary` to get mean ± stdev fitness across seeds
    3. Call :meth:`combined_history` to extract convergence histories for plotting
    4. Iterate over ``.runs`` to inspect per-seed details (metrics, errors, artifacts)

    **Attributes vs. Methods**:

    - ``canonical_summary`` : metrics from the first seed (quick reference)
    - ``aggregated_summary()`` : mean/median/stdev metrics across all seeds (recommended)
    - ``combined_history()`` : flattened per-generation history with seed labels
    - `.runs[n].history` : per-generation records for a specific seed

    Examples
    --------
    Access aggregated fitness metrics::

        result = composer.quick_run(
            fitness="sphere:dim=10",
            seeds=(42, 43, 44),
        )
        agg = result.aggregated_summary()
        # agg = {
        #   "best_fitness": {"mean": 0.5, "median": 0.48, "stdev": 0.05},
        #   "mean_fitness": {"mean": 2.3, ...},
        # }
        print(
            f"Best fitness: {agg['best_fitness']['mean']:.3f} "
            f"± {agg['best_fitness']['stdev']:.3f}"
        )

    Inspect per-seed runs::

        for i, run in enumerate(result.runs):
            print(f"Seed {run.seed}: status={run.status}, duration={run.duration_seconds}s")
            print(f"  Final best fitness: {run.metrics.get('best_fitness', 'N/A')}")
            # run.history is a list of dicts, one per generation

    Export convergence history to pandas::

        import pandas as pd
        history = result.combined_history(seed_field="seed_id")
        df = pd.DataFrame(history)
        print(df[['generation', 'best_fitness', 'seed_id']])
        # Rows can now be grouped by seed_id and plotted
    """

    name: str
    runs: List[RunResult]
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
        """Flatten all run histories into a single list with seed labels.

        Concatenates every generation record from every seed into one list,
        adding a seed identifier to each row. Useful for exporting to pandas,
        CSV, or custom plotting libraries.

        Parameters
        ----------
        seed_field : str, optional
            Name of the key to add to each row (default: ``"seed"``).
            The value is ``run.seed`` from the originating :class:`RunResult`.

        Returns
        -------
        List[Dict[str, Any]]
            Flattened history with one dict per generation-per-seed.
            Each dict includes original history keys plus the seed identifier.

        Examples
        --------
        Export to pandas DataFrame::

            import pandas as pd
            history = result.combined_history()
            df = pd.DataFrame(history)
            # Columns: 'generation', 'best_fitness', 'seed', ...
            per_seed = df.groupby('seed')['best_fitness'].apply(list)

        Write to CSV::

            import csv
            history = result.combined_history(seed_field="run_seed")
            with open("convergence.csv", "w") as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
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
        """Compute aggregated statistics for each metric across all seeds.

        This is the primary method for summarizing multi-seed experimental
        results. For each numeric metric (e.g., best_fitness, mean_fitness),
        computes mean, median, and standard deviation across all runs.
        Non-numeric values are silently ignored.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping of metric name to statistics dict. Each statistics dict
            contains:

            - ``"mean"`` : arithmetic mean across seeds
            - ``"median"`` : median value across seeds
            - ``"stdev"`` : standard deviation (0.0 if only one seed)

        Notes
        -----
        Only numeric metric values are included. String or object-valued
        metrics are silently skipped.

        If only one seed was run, the standard deviation is 0.0 (by
        definition). For robust statistics, use at least 3 seeds.

        Examples
        --------
        Inspect fitness statistics::

            result = composer.quick_run(..., seeds=(42, 43, 44))
            agg = result.aggregated_summary()
            # Returns: {
            #   'best_fitness': {'mean': 0.123, 'median': 0.110, 'stdev': 0.045},
            #   'mean_fitness': {'mean': 2.345, 'median': 2.310, 'stdev': 0.120},
            # }

        Report results::

            agg = result.aggregated_summary()
            for metric, stats in agg.items():
                mean_val = stats['mean']
                std_val = stats['stdev']
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
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

    def gap_summary(self) -> Dict[str, float]:
        """Convenience summary for the final gap-to-optimum metric.

        This returns the aggregated statistics for ``gap_to_optimum`` when the
        metric is present in one or more run summaries; otherwise it returns an
        empty dictionary.
        """
        return self.aggregated_summary().get("gap_to_optimum", {})


# ---------------------------------------------------------------------------
# ComparisonResult — aligned multi-pipeline results
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Aligned results from multiple algorithm pipelines for direct comparison.

    Returned by :meth:`Composer.compare` and :meth:`Composer.from_toml`, a
    :class:`ComparisonResult` contains multiple :class:`ExperimentResult`
    objects (one per algorithm variant), with convenient methods to summarize,
    visualize, and statistically compare the pipelines.

    Attributes
    ----------
    pipelines : Dict[str, ExperimentResult]
        Mapping of pipeline name (str) to its full :class:`ExperimentResult`.
        Each pipeline contains multi-seed runs and aggregation methods.
    shared_config : Dict[str, Any]
        Configuration parameters that were applied to all pipelines
        (passed as ``**shared_kwargs`` to :meth:`Composer.compare`).
    initial_population : optional
        Shared initial population array, if ``shared_initial_population=True``
        was set during construction. If ``None``, each pipeline initialized
        with its own random population.
    negate_map : Dict[str, bool]
        Per-pipeline fitness sign-flip flag. When ``True``, fitness values
        are negated before display so all pipelines use "lower is better"
        convention (e.g., for Evosax which uses positive fitness values).

    Notes
    -----
    **Typical Workflow**:

    1. Call :meth:`Composer.compare(...) <Composer.compare>` -> returns :class:`ComparisonResult`
    2. Call :meth:`summary_table` to get a dataframe-ready dict of aggregated metrics
    3. Call :meth:`plot_convergence` for matplotlib visualization
    4. Iterate over ``.pipelines.items()`` to access individual :class:`ExperimentResult`

    **Backend Normalization**:

    The ``negate_map`` is automatically populated by :meth:`Composer.compare`
    based on the backend used per pipeline (``"malthusjax"`` vs ``"evosax"``).
    Users typically don't need to set this manually; it's applied transparently
    in :meth:`summary_table` and :meth:`plot_convergence`.

    Examples
    --------
    Compare two algorithms::

        comparison = composer.compare(
            pipelines={
                "GA+Blend": dict(crossover="blend:alpha=0.5"),
                "GA+SBX": dict(crossover="simulated_binary:eta=20"),
            },
            fitness="sphere:dim=10",
            pop_size=50,
            generations=200,
            seeds=(42, 43, 44),
        )

    Extract summary statistics::

        table = comparison.summary_table()
        # table = {
        #   "GA+Blend": {"best_fitness": 0.123, "mean_fitness": 2.345},
        #   "GA+SBX": {"best_fitness": 0.089, "mean_fitness": 1.987},
        # }
        for pipeline_name, metrics in table.items():
            print(f"{pipeline_name}: best={metrics['best_fitness']:.4f}")

    Visualize convergence::

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for seed_idx, ax in enumerate(axes):
            comparison.plot_convergence(seed_index=seed_idx, ax=ax)
            ax.set_title(f"Seed {seed_idx + 1}")
        plt.tight_layout()
        plt.show()
    """

    pipelines: Dict[str, ExperimentResult]
    shared_config: Dict[str, Any] = field(default_factory=dict)
    initial_population: Optional[Any] = field(default=None, repr=False)
    negate_map: Dict[str, bool] = field(default_factory=dict)
    """Per-pipeline sign-flip flag.

    When ``True`` for a pipeline, fitness values are **negated** before
    display so that all pipelines share a unified "lower is better"
    convention (more negative is better).

    This is typically used for pipelines where the raw reported fitness is
    a standard minimisation value (lower is better) but stored as a positive
    quantity (e.g., evosax "fitness" values).  The flag is usually set by
    :meth:`Composer.compare` based on the backend in use.
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

    def summary_table(self, latex: bool = False) -> Union[Dict[str, Dict[str, float]], str]:
        """Compute per-pipeline aggregated metrics across all seeds.

        This is the main method for comparing algorithm performance. Returns
        one row per pipeline, with columns for each metric (best_fitness,
        mean_fitness, etc.). Fitness values are automatically sign-normalized
        so that all pipelines use "lower is better" convention (controlled by
        :attr:`negate_map`).

        Parameters
        ----------
        latex : bool, optional
            If ``True``, return the result as a LaTeX table string.
            Otherwise, return the normal dictionary form.
            Default: ``False``.

        Returns
        -------
        Dict[str, Dict[str, float]] or str
            Either the aggregated metrics mapping or a LaTeX formatted table.

        Notes
        -----
        The result is suitable for pandas DataFrame construction::

            import pandas as pd
            table = comparison.summary_table()
            df = pd.DataFrame(table).T  # Transpose to get pipelines as rows

        Examples
        --------
        View aggregated performance::

            table = comparison.summary_table()
            # table = {
            #   "GA+Blend": {"best_fitness": 0.123, "mean_fitness": 2.34},
            #   "GA+SBX": {"best_fitness": 0.089, "mean_fitness": 1.98},
            #   "Evosax CMA-ES": {"best_fitness": 0.045, "mean_fitness": 1.23},
            # }

        Convert to pandas for ranking::

            import pandas as pd
            table = comparison.summary_table()
            df = pd.DataFrame(table).T
            df = df.sort_values("best_fitness")  # Rank by fitness
            print(df)

        Get a LaTeX table::

            latex_code = comparison.summary_table(latex=True)
            print(latex_code)
        """
        table: Dict[str, Dict[str, float]] = {}
        for name, exp in self.pipelines.items():
            agg = exp.aggregated_summary()
            s = self._sign(name)
            # Flatten to the mean; negate fitness keys if needed
            table[name] = {
                k: (v["mean"] * s if k in self._FITNESS_KEYS else v["mean"]) for k, v in agg.items()
            }

        if not latex:
            return table

        def _latex_escape(value: str) -> str:
            return (
                value.replace("\\", "\\textbackslash{}")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("&", "\\&")
                .replace("~", "\\textasciitilde{}")
                .replace("^", "\\textasciicircum{}")
            )

        metric_names = []
        for metrics in table.values():
            for key in metrics.keys():
                if key not in metric_names:
                    metric_names.append(key)

        header_cols = ["Pipeline"] + metric_names
        lines = ["\\begin{tabular}{l" + "r" * len(metric_names) + "}", "\\hline"]
        lines.append(" & ".join(_latex_escape(col) for col in header_cols) + r" \\")
        lines.append("\\hline")

        for pipeline_name, metrics in table.items():
            row = [ _latex_escape(pipeline_name) ]
            for metric in metric_names:
                value = metrics.get(metric, float("nan"))
                row.append(f"{value:.6g}")
            lines.append(" & ".join(row) + r" \\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def normalized_runs(self, pipeline_name: str) -> List[RunResult]:
        """Return per-seed runs with sign-normalized fitness metrics.

        When comparing pipelines from different backends, raw run metrics may
        use different sign conventions for minimization or maximization.
        This helper returns a copy of the selected pipeline's runs where all
        fitness-related metrics and history entries are normalized according
        to the comparison's ``negate_map``.
        """
        if pipeline_name not in self.pipelines:
            raise KeyError(f"Unknown pipeline '{pipeline_name}'")

        sign = self._sign(pipeline_name)
        if sign == 1.0:
            return self.pipelines[pipeline_name].runs

        normalized_runs: List[RunResult] = []
        for run in self.pipelines[pipeline_name].runs:
            normalized_metrics = {
                k: (v * sign if k in self._FITNESS_KEYS else v)
                for k, v in run.metrics.items()
            }
            normalized_history = [
                {k: (v * sign if k in self._FITNESS_KEYS else v) for k, v in row.items()}
                for row in run.history
            ]
            normalized_runs.append(
                RunResult(
                    seed=run.seed,
                    status=run.status,
                    metrics=normalized_metrics,
                    history=normalized_history,
                    artifacts=run.artifacts,
                    duration_seconds=run.duration_seconds,
                    timings=run.timings,
                    error=run.error,
                    created_at=run.created_at,
                )
            )

        return normalized_runs

    def timing_data(self, timing_key: str = "duration_seconds") -> Dict[str, List[float]]:
        """Collect timing values for each pipeline across all seeds.

        Parameters
        ----------
        timing_key : str, optional
            Which timing channel to collect. ``"duration_seconds"`` gathers
            the per-run wall-clock duration. Other keys may be taken from
            ``RunResult.timings`` if present.
            Default: ``"duration_seconds"``.

        Returns
        -------
        Dict[str, List[float]]
            Mapping of pipeline name to a list of timing values.
        """
        data: Dict[str, List[float]] = {}
        for name, exp in self.pipelines.items():
            values: List[float] = []
            for run in exp.runs:
                if timing_key == "duration_seconds":
                    if run.duration_seconds is not None:
                        values.append(run.duration_seconds)
                else:
                    if run.timings is None:
                        continue
                    if timing_key in run.timings:
                        try:
                            values.append(float(run.timings[timing_key]))
                        except (TypeError, ValueError):
                            continue
            data[name] = values
        return data

    def plot_timing_boxplot(
        self,
        timing_key: str = "duration_seconds",
        ax: Any = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Draw a per-pipeline boxplot for timing values.

        Parameters
        ----------
        timing_key : str, optional
            Timing channel to plot. ``"duration_seconds"`` uses the per-run
            wall-clock runtime. Other values may come from the
            per-run ``RunResult.timings`` dictionary.
            Default: ``"duration_seconds"``.

        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If ``None``, a new figure and axis pair is created.

        title : str, optional
            Plot title. If ``None``, defaults to ``"Timing boxplot"``.

        save_path : str or pathlib.Path, optional
            If provided, save the generated figure to this path.
            The parent directory will be created if it does not exist.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the boxplot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_timing_boxplot(). "
                "Install it with: pip install matplotlib"
            ) from e

        data = self.timing_data(timing_key=timing_key)
        labels: List[str] = []
        values: List[List[float]] = []
        for name, timings in data.items():
            if timings:
                labels.append(name)
                values.append(timings)

        if not values:
            raise ValueError(
                f"No timing values available for timing_key='{timing_key}'"
            )

        fig: Figure | None = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        else:
            fig = getattr(ax, "figure", None)

        try:
            ax.boxplot(values, tick_labels=labels, patch_artist=True)
        except TypeError:
            ax.boxplot(values, labels=labels, patch_artist=True)
        ax.set_title(title or "Timing boxplot")
        ax.set_ylabel(f"{timing_key} (seconds)")
        ax.set_xlabel("Pipeline")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        if save_path is not None and fig is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=300)

        return ax

    def final_metric_data(self, metric_key: str = "best_fitness") -> Dict[str, List[float]]:
        """Collect final metric values for each pipeline across all seeds.

        Parameters
        ----------
        metric_key : str, optional
            Metric value to gather from each run's ``RunResult.metrics``.
            Default: ``"best_fitness"``.

        Returns
        -------
        Dict[str, List[float]]
            Mapping of pipeline name to a list of final metric values.
            Values are sign-normalized for pipelines marked in
            ``negate_map``.
        """
        data: Dict[str, List[float]] = {}
        for name, exp in self.pipelines.items():
            values: List[float] = []
            sign = self._sign(name) if metric_key in self._FITNESS_KEYS else 1.0
            for run in exp.runs:
                if metric_key not in run.metrics:
                    continue
                try:
                    value = float(run.metrics[metric_key])
                except (TypeError, ValueError):
                    continue
                values.append(sign * value)
            data[name] = values
        return data

    def plot_final_metric_boxplot(
        self,
        metric_key: str = "best_fitness",
        ax: Any = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Draw a per-pipeline boxplot for final run metric values.

        Parameters
        ----------
        metric_key : str, optional
            Metric to plot from the final results of each run. By default,
            ``"best_fitness"`` is used.

        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If ``None``, a new figure and axis pair is created.

        title : str, optional
            Plot title. If ``None``, defaults to
            ``"Final {metric_key} distribution"``.

        save_path : str or pathlib.Path, optional
            If provided, save the generated figure to this path.
            The parent directory will be created if it does not exist.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the boxplot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_final_metric_boxplot(). "
                "Install it with: pip install matplotlib"
            ) from e

        data = self.final_metric_data(metric_key=metric_key)
        labels: List[str] = []
        values: List[List[float]] = []
        for name, metric_vals in data.items():
            if metric_vals:
                labels.append(name)
                values.append(metric_vals)

        if not values:
            raise ValueError(
                f"No metric values available for metric_key='{metric_key}'"
            )

        fig: Figure | None = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        else:
            fig = getattr(ax, "figure", None)

        try:
            ax.boxplot(values, tick_labels=labels, patch_artist=True)
        except TypeError:
            ax.boxplot(values, labels=labels, patch_artist=True)

        ax.set_title(title or f"Final {metric_key} distribution")
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_xlabel("Pipeline")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        if save_path is not None and fig is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=300)

        return ax

    def convergence_data(self, seed_index: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        """Extract per-pipeline convergence histories for a single seed.

        Returns generation-by-generation records for the chosen seed across
        all pipelines. Fitness values are automatically sign-normalized
        (controlled by :attr:`negate_map`). Useful for custom plotting or
        statistical analysis beyond built-in :meth:`plot_convergence`.

        Parameters
        ----------
        seed_index : int, optional
            Which seed's history to use (0-indexed). If ``seed_index`` exceeds
            the number of available seeds for a pipeline, that pipeline receives
            an empty list.
            Default: 0 (first seed).

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Mapping of pipeline name -> list of generation dicts.
            Each dict in the list contains:

            - ``"generation"`` : generation number (0, 1, 2, ...)
            - ``"best_fitness"`` : best fitness in population (sign-normalized)
            - ``"mean_fitness"`` : mean population fitness
            - (any other per-generation metrics)

            Fitness values are already sign-normalized (via :attr:`negate_map`).

        Notes
        -----
        Use ``seed_index`` to extract convergence curves for different seeds.
        For example, plot seed 0, 1, 2 separately to inspect variance across
        random initializations.

        Examples
        --------
        Extract and plot custom graph::

            import matplotlib.pyplot as plt
            data = comparison.convergence_data(seed_index=0)
            for pipeline_name, history in data.items():
                gens = [h["generation"] for h in history]
                fitness = [h["best_fitness"] for h in history]
                plt.plot(gens, fitness, label=pipeline_name)
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.legend()
            plt.show()

        Export to pandas for analysis::

            import pandas as pd
            data = comparison.convergence_data(seed_index=0)
            for pipeline_name, history in data.items():
                df = pd.DataFrame(history)
                print(f"{pipeline_name} convergence:")
                print(df[["generation", "best_fitness"]].head(10))
        """
        data: Dict[str, List[Dict[str, Any]]] = {}
        for name, exp in self.pipelines.items():
            if seed_index < len(exp.runs):
                raw = exp.runs[seed_index].history
                s = self._sign(name)
                if s < 0:
                    data[name] = [
                        {k: (v * s if k in self._FITNESS_KEYS else v) for k, v in row.items()}
                        for row in raw
                    ]
                else:
                    data[name] = raw
            else:
                data[name] = []
        return data

    def plot_convergence(
        self,
        seed_index: Union[int, List[int]] = 0,
        ax: Any = None,
        title: Optional[str] = None,
        negate: Optional[Dict[str, bool]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Visualize convergence of all pipelines on a matplotlib axis.

        Creates an overlaid line plot with one curve per pipeline, showing
        best fitness vs. generation for the selected seed. Fitness values are
        automatically sign-normalized (via :attr:`negate_map`) so that all
        pipelines use "lower is better" convention.

        Parameters
        ----------
        seed_index : int or list[int], optional
            Which seed(s) convergence history to plot.
            If an integer, draws a single plot. If a list, draws a subplot for
            each seed in the list and returns a sequence of axes.
            Use ``seed_index=0, 1, 2, ...`` to inspect robustness across
            different random initializations.
            Default: 0 (first seed).

        ax : matplotlib.axes.Axes or sequence of Axes, optional
            Matplotlib axis or axes to draw on. If ``None``, new figure and
            axis/axes are created. When ``seed_index`` is a list, ``ax`` must
            be an iterable of axes with the same length as the seed list.
            Default: ``None``.

        title : str, optional
            Plot title. If ``None``, defaults to ``"Convergence comparison"``.
            When plotting multiple seeds, the seed number is appended to each
            subplot title.
            Default: ``None``.

        negate : Dict[str, bool], optional
            Additional per-pipeline sign-flip after automatic normalization.
            Use this to switch between "lower is better" and "higher is better"
            display. Example: ``negate={"Pipeline A": True}`` flips only
            "Pipeline A"'s curve.
            Default: ``None`` (no additional flips).

        save_path : str or pathlib.Path, optional
            If provided, save the generated figure to this path.
            The parent directory will be created if it does not exist.

        Returns
        -------
        matplotlib.axes.Axes or list[matplotlib.axes.Axes]
            The matplotlib axis object or list of axis objects for multiple
            seed plots.

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Notes
        -----
        **Line Style**:
        Each pipeline gets a different color (matplotlib auto-cycling).
        Lines are 2 points wide for visibility.
        Grid is enabled with alpha=0.3 for readability.

        **Sign Normalization**:
        Fitness values are automatically negated (if ``negate_map[pipeline]``)
        so that all pipelines use "lower is better". The sign map is set by
        :meth:`Composer.compare` and :meth:`Composer.from_toml` based on backend
        conventions.

        **Multiple Seeds**:
        Call with different ``seed_index`` values to create a multi-panel
        figure showing robustness across seeds.

        Examples
        --------
        Simple convergence comparison::

            comparison = composer.compare(
                pipelines={
                    "GA+Blend": {...},
                    "GA+SBX": {...},
                },
                ...
            )
            comparison.plot_convergence(seed_index=0)
            # Shows overlaid curves for both pipelines

        Multi-panel figure (one per seed)::

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for seed_idx, ax in enumerate(axes):
                comparison.plot_convergence(seed_index=seed_idx, ax=ax)
                ax.set_title(f"Seed {seed_idx}")
            plt.tight_layout()
            plt.show()

        Customize plot::

            ax = comparison.plot_convergence(
                seed_index=0,
                title="GA Benchmark on Sphere Function",
            )
            ax.set_ylim(bottom=0)  # Force y-axis to start at 0
            ax.set_yscale("log")    # Log scale for faster-converging algorithms
            plt.show()

        Switch to "higher is better" display (for maximization)::

            # After automatic normalization, flip all curves
            ax = comparison.plot_convergence(
                seed_index=0,
                negate={name: True for name in comparison.names},
            )
            ax.set_ylabel("Best Fitness (higher is better)")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_convergence(). "
                "Install it with: pip install matplotlib"
            ) from e

        if isinstance(seed_index, (list, tuple)):
            seed_list = list(seed_index)
            if ax is None:
                fig, axes_array = plt.subplots(
                    1,
                    len(seed_list),
                    figsize=(5 * len(seed_list), 4),
                    squeeze=False,
                )
                axes: list[Any] = list(axes_array[0])
            elif hasattr(ax, "__iter__") and not isinstance(ax, (str, bytes)):
                axes = list(ax)
                if len(axes) != len(seed_list):
                    raise ValueError(
                        "When seed_index is a list, ax must contain one axis per seed."
                    )
                fig = getattr(axes[0], "figure", None)
            else:
                raise ValueError(
                    "When seed_index is a list, ax must be an iterable of axes."
                )

            for subplot_ax, seed in zip(axes, seed_list):
                subplot_title = (
                    f"{title} (seed {seed})" if title is not None else f"Seed {seed}"
                )
                self.plot_convergence(seed, ax=subplot_ax, title=subplot_title, negate=negate)

            if save_path is not None and fig is not None:
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, bbox_inches="tight")
            return axes

        fig: Figure | None = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = getattr(ax, "figure", None)

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

        if save_path is not None and fig is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
        return ax
