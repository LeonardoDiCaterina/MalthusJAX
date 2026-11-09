from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from malthusjax.stats.core import (
    HypothesisKind,
    MultipleTestingPolicy,
    PairedMetricDataset,
    StatisticalComparisonResult,
    StatisticalComparisonSpec,
    StatisticalSpecError,
    StatisticalSuiteResult,
    TestResult,
    TOSTResult,
    infer_scipy_alternative,
    validate_spec,
)
from malthusjax.stats.correction import adjust_pvalues
from malthusjax.stats.effects import compute_effect_sizes
from malthusjax.stats.tests import compute_standard_tests, compute_tost_paired

if TYPE_CHECKING:
    from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult


def apply_decision_rule(
    *,
    spec: StatisticalComparisonSpec,
    tests: dict[str, TestResult],
    tost: TOSTResult | None,
) -> tuple[bool | None, str, str | None]:
    """Return decision tuple: (pass, basis, error)."""
    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        if (
            tost is None
            or tost.status != "valid"
            or tost.p_value_max is None
            or not np.isfinite(tost.p_value_max)
            or not 0 <= tost.p_value_max <= 1
        ):
            return None, "tost", "TOST result unavailable"
        return bool(tost.p_value_max < spec.alpha), "tost", None

    primary = tests.get(spec.include_tests[0])
    if primary is not None and primary.status == "valid" and primary.p_value is not None:
        return bool(primary.p_value < spec.alpha), f"{primary.name}_{primary.alternative}", None
    return None, "none", "Predeclared primary test unavailable"


def conclusion_for(
    kind, sidedness, decision, orientation="lower_is_better", direction="left_lt_right"
):
    if decision is None:
        return "unavailable"
    if kind == HypothesisKind.EQUIVALENCE:
        return "equivalent" if decision else "equivalence_not_established"
    if sidedness == "one_sided":
        beneficial = (orientation == "lower_is_better") == (direction == "left_lt_right")
        noun = "improvement" if beneficial else "deterioration"
        return f"{noun}_detected" if decision else f"{noun}_not_detected"
    return "difference_detected" if decision else "difference_not_detected"


def primary_p_value(result):
    if result.decision_basis == "tost" and result.tost:
        return result.tost.p_value_max
    for name, test in result.tests.items():
        if result.decision_basis.startswith(name + "_"):
            return test.p_value
    return None


def compare_paired_arrays(
    *,
    label: str,
    left_name: str,
    right_name: str,
    left: np.ndarray,
    right: np.ndarray,
    spec: StatisticalComparisonSpec,
    metadata: dict[str, Any] | None = None,
) -> StatisticalComparisonResult:
    """Core paired comparison engine used by StatisticalComparator."""
    validate_spec(spec)

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if (
        left_arr.ndim != 1
        or right_arr.ndim != 1
        or not np.isfinite(left_arr).all()
        or not np.isfinite(right_arr).all()
    ):
        raise ValueError("Samples must be finite one-dimensional paired arrays")
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left/right shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    if left_arr.size == 0:
        raise StatisticalSpecError("Cannot compare empty samples")
    insufficient = left_arr.size < spec.min_paired_seeds

    alternative = infer_scipy_alternative(spec.sidedness, spec.expected_direction)

    tests = compute_standard_tests(
        left_arr,
        right_arr,
        alternative=alternative,
        include_tests=spec.include_tests,
    )

    tost: TOSTResult | None = None
    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        tost = compute_tost_paired(
            left_arr,
            right_arr,
            margin=float(spec.equivalence_margin),  # type: ignore
            alpha=spec.alpha,
        )

    effects = compute_effect_sizes(left_arr, right_arr)
    decision_pass, decision_basis, decision_error = apply_decision_rule(
        spec=spec,
        tests=tests,
        tost=tost,
    )

    diffs = left_arr - right_arr

    # Normality non-rejection is not proof of assumptions or adequate power.
    shapiro_p = None
    if diffs.size >= 3 and np.var(diffs) > 0:
        import scipy.stats as scipy_stats

        normality = scipy_stats.shapiro(diffs)
        if np.isfinite(normality.pvalue):
            shapiro_p = float(normality.pvalue)
            tests["shapiro_wilk"] = TestResult(
                "shapiro_wilk", float(normality.statistic), shapiro_p, "two-sided"
            )
    decision_reliable = None
    if (
        (decision_basis.startswith("paired_t") or decision_basis == "tost")
        and shapiro_p is not None
        and shapiro_p < spec.alpha
    ):
        decision_reliable = False
    if insufficient:
        decision_pass = None
        decision_error = f"Insufficient paired seeds: {left_arr.size} < {spec.min_paired_seeds}"
    wins_left = int(
        np.sum(
            left_arr < right_arr
            if spec.metric_orientation == "lower_is_better"
            else left_arr > right_arr
        )
    )
    wins_right = int(
        np.sum(
            right_arr < left_arr
            if spec.metric_orientation == "lower_is_better"
            else right_arr > left_arr
        )
    )
    ties = int(np.sum(left_arr == right_arr))

    hypothesis_text = {
        HypothesisKind.LOCATION_SHIFT: "location shift",
        HypothesisKind.CLOSER_TO_OPTIMUM: "closer to optimum",
        HypothesisKind.EQUIVALENCE: "equivalence (TOST)",
    }[spec.hypothesis_kind]

    out_metadata = dict(metadata or {})
    out_metadata.update(
        {
            "left_name": left_name,
            "right_name": right_name,
            "metric_name": spec.metric_name,
            "metric_orientation": spec.metric_orientation,
            "alternative": alternative,
            "hypothesis_kind": spec.hypothesis_kind.value,
            "sidedness": spec.sidedness.value,
            "expected_direction": spec.expected_direction.value,
            "min_paired_seeds": spec.min_paired_seeds,
            "estimand": "mean paired difference"
            if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE
            else "paired signed-rank location"
            if spec.include_tests[0] == "wilcoxon"
            else "paired difference",
            "power_policy": "not certified by a mechanical sample-count minimum",
        }
    )
    if decision_error is not None:
        out_metadata["decision_error"] = decision_error

    left_start_mean = out_metadata.get("left_start_mean")
    right_start_mean = out_metadata.get("right_start_mean")
    left_end_mean = float(np.mean(left_arr))
    right_end_mean = float(np.mean(right_arr))
    out_metadata["left_end_mean"] = left_end_mean
    out_metadata["right_end_mean"] = right_end_mean

    if left_start_mean is not None and right_start_mean is not None:
        left_delta = float(left_end_mean - float(left_start_mean))
        right_delta = float(right_end_mean - float(right_start_mean))
        out_metadata["left_end_minus_start_mean"] = left_delta
        out_metadata["right_end_minus_start_mean"] = right_delta
        out_metadata["delta_end_minus_start_diff_mean"] = float(left_delta - right_delta)

        if spec.optimum_value is not None:
            opt = float(spec.optimum_value)
            left_start_dist = abs(float(left_start_mean) - opt)
            right_start_dist = abs(float(right_start_mean) - opt)
            left_end_dist = abs(left_end_mean - opt)
            right_end_dist = abs(right_end_mean - opt)
            left_closed = float(left_start_dist - left_end_dist)
            right_closed = float(right_start_dist - right_end_dist)
            left_closed_pct = None
            right_closed_pct = None
            if left_start_dist > 0.0:
                left_closed_pct = float(100.0 * left_closed / left_start_dist)
            if right_start_dist > 0.0:
                right_closed_pct = float(100.0 * right_closed / right_start_dist)

            out_metadata.update(
                {
                    "optimum_value": opt,
                    "left_start_distance_to_optimum": float(left_start_dist),
                    "right_start_distance_to_optimum": float(right_start_dist),
                    "left_end_distance_to_optimum": float(left_end_dist),
                    "right_end_distance_to_optimum": float(right_end_dist),
                    "left_distance_closed_pct": left_closed_pct,
                    "right_distance_closed_pct": right_closed_pct,
                    "distance_closed_pct_point_diff_left_minus_right": (
                        float(left_closed_pct - right_closed_pct)
                        if left_closed_pct is not None and right_closed_pct is not None
                        else None
                    ),
                }
            )

    if tost is not None:
        out_metadata.update(
            {
                "equivalence_margin": tost.margin,
                "equivalence_bounds": [tost.lower_bound, tost.upper_bound],
                "tost_t_lower": tost.t_stat_lower,
                "tost_t_upper": tost.t_stat_upper,
                "tost_p_lower": tost.p_value_lower,
                "tost_p_upper": tost.p_value_upper,
                "tost_p_max": tost.p_value_max,
                "tost_equivalent_raw": tost.equivalent,
                "tost_equivalent": decision_pass,
            }
        )

        tests = dict(tests)
        tests["tost"] = TestResult(
            name="tost",
            statistic=None,
            p_value=tost.p_value_max,
            alternative="equivalence",
        )

    if spec.include_value_lists:
        out_metadata["include_value_lists"] = True
        out_metadata["left_end_values"] = [float(v) for v in left_arr.tolist()]
        out_metadata["right_end_values"] = [float(v) for v in right_arr.tolist()]

    return StatisticalComparisonResult(
        label=label,
        hypothesis_text=hypothesis_text,
        n_paired=int(left_arr.size),
        wins_left=wins_left,
        wins_right=wins_right,
        ties=ties,
        left_mean=float(np.mean(left_arr)),
        right_mean=float(np.mean(right_arr)),
        mean_diff_left_minus_right=float(np.mean(diffs)),
        median_diff_left_minus_right=float(np.median(diffs)),
        tests=tests,
        tost=tost,
        effects=effects,
        alpha=spec.alpha,
        decision_pass=decision_pass,
        decision_reliable=decision_reliable,
        decision_basis=decision_basis,
        metadata=out_metadata,
        status="insufficient"
        if insufficient
        else "valid"
        if decision_pass is not None
        else "unavailable",
        reason=decision_error,
        conclusion=conclusion_for(
            spec.hypothesis_kind,
            spec.sidedness,
            decision_pass,
            spec.metric_orientation,
            spec.expected_direction,
        ),
        raw_decision_pass=decision_pass,
        raw_conclusion=conclusion_for(
            spec.hypothesis_kind,
            spec.sidedness,
            decision_pass,
            spec.metric_orientation,
            spec.expected_direction,
        ),
    )


class StatisticalComparator:
    """High-level interface for paired and suite-level statistical comparison."""

    def compare_paired(
        self,
        dataset: PairedMetricDataset,
        spec: StatisticalComparisonSpec,
    ) -> StatisticalComparisonResult:
        """Run a single paired comparison for one aligned dataset."""
        if len(dataset.seeds) != len(set(dataset.seeds)) or len(dataset.seeds) != len(
            dataset.left_values
        ):
            raise StatisticalSpecError(
                "Paired seed IDs must be unique and aligned with observations"
            )
        return compare_paired_arrays(
            label=dataset.label,
            left_name=dataset.left_name,
            right_name=dataset.right_name,
            left=dataset.left_values,
            right=dataset.right_values,
            spec=spec,
            metadata={
                **dataset.metadata,
                "metric_name": dataset.metric_name,
                "metric_source": dataset.metric_source,
                "seeds": list(dataset.seeds),
            },
        )

    def compare_suite(
        self,
        datasets: list[PairedMetricDataset],
        spec: StatisticalComparisonSpec,
    ) -> StatisticalSuiteResult:
        """Run a suite of paired comparisons under one common spec."""
        if len({ds.label for ds in datasets}) != len(datasets):
            raise StatisticalSpecError("Comparison labels must be unique within a family")
        results: list[StatisticalComparisonResult] = [
            self.compare_paired(ds, spec) for ds in datasets
        ]
        suite = StatisticalSuiteResult(spec=spec, results=results)
        if spec.multiple_testing != MultipleTestingPolicy.NONE:
            return self.adjust_suite_pvalues(suite)
        return suite

    def adjust_suite_pvalues(self, suite: StatisticalSuiteResult) -> StatisticalSuiteResult:
        """Apply suite-level p-value correction to an existing suite result."""
        if suite.spec.multiple_testing == MultipleTestingPolicy.NONE:
            return suite

        if len({r.label for r in suite.results}) != len(suite.results):
            raise StatisticalSpecError("Comparison labels must be unique within a family")
        pvals = [primary_p_value(r) if r.status == "valid" else None for r in suite.results]
        # Unavailable planned hypotheses retain their place in the family.
        adjusted = adjust_pvalues(
            [p if p is not None else np.nan for p in pvals], suite.spec.multiple_testing
        )
        suite.adjusted_p_values = {}
        for result, raw, p in zip(suite.results, pvals, adjusted):
            result.decision_scope = "family_corrected"
            result.adjusted_p_value = float(p) if raw is not None and np.isfinite(p) else None
            result.decision_pass = (
                bool(p < suite.spec.alpha) if result.adjusted_p_value is not None else None
            )
            result.conclusion = conclusion_for(
                result.metadata.get("hypothesis_kind", suite.spec.hypothesis_kind),
                result.metadata.get("sidedness", suite.spec.sidedness),
                result.decision_pass,
                result.metadata.get("metric_orientation", suite.spec.metric_orientation),
                result.metadata.get("expected_direction", suite.spec.expected_direction),
            )
            if result.decision_basis == "tost":
                result.metadata["tost_equivalent_raw"] = result.tost.equivalent
                result.metadata["tost_equivalent"] = result.decision_pass
            result.metadata.update(
                comparison_family_size=len(suite.results),
                multiplicity_policy=suite.spec.multiple_testing.value,
            )
            if result.adjusted_p_value is not None:
                suite.adjusted_p_values[result.label] = {"primary": result.adjusted_p_value}
        return suite


def attach_adjusted_pvalues(
    results: list[StatisticalComparisonResult],
    adjusted: list[float],
    *,
    key: str,
) -> dict[str, dict[str, float]]:
    """Build label-indexed adjusted p-value mapping for suite exports."""
    if len(results) != len(adjusted):
        raise ValueError("results and adjusted lengths must match")
    out: dict[str, dict[str, float]] = {}
    for result, value in zip(results, adjusted):
        if np.isfinite(value):
            out[result.label] = {key: float(value)}
    return out


# We need to port the `_extract_*` and `paired_dataset_from_*` from benchmarking.statistics
# over here since they are tightly coupled to PairedMetricDataset creation from results


def _describe_values(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def _paired_timing_stats(left: list[float], right: list[float]) -> dict[str, float | None]:
    import scipy.stats as stats

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0 or left_arr.size != right_arr.size:
        return {
            "paired_diff_mean_left_minus_right": None,
            "paired_diff_median_left_minus_right": None,
            "paired_cohen_dz": None,
            "paired_wilcoxon_p_value": None,
        }

    diffs = left_arr - right_arr
    if left_arr.size >= 2:
        std = float(np.std(diffs, ddof=1))
        cohen_dz = 0.0 if np.isclose(std, 0.0) else float(np.mean(diffs) / std)
        try:
            w_res = stats.wilcoxon(
                left_arr,
                right_arr,
                alternative="two-sided",
                zero_method="wilcox",
            )
            w_p = float(w_res.pvalue)
        except ValueError:
            w_p = None
    else:
        cohen_dz = None
        w_p = None

    return {
        "paired_diff_mean_left_minus_right": float(np.mean(diffs)),
        "paired_diff_median_left_minus_right": float(np.median(diffs)),
        "paired_cohen_dz": cohen_dz,
        "paired_wilcoxon_p_value": w_p,
    }


def _build_timing_summary(
    left_total: list[float],
    right_total: list[float],
    left_components: dict[str, list[float]],
    right_components: dict[str, list[float]],
) -> dict[str, Any]:
    total_summary = {
        "left_mean": _describe_values(left_total)["mean"],
        "right_mean": _describe_values(right_total)["mean"],
        "left_median": _describe_values(left_total)["median"],
        "right_median": _describe_values(right_total)["median"],
        "left_min": _describe_values(left_total)["min"],
        "right_min": _describe_values(right_total)["min"],
        "left_max": _describe_values(left_total)["max"],
        "right_max": _describe_values(right_total)["max"],
        **_paired_timing_stats(left_total, right_total),
    }

    components: dict[str, Any] = {}
    common_keys = sorted(set(left_components) & set(right_components))
    for key in common_keys:
        lv = left_components[key]
        rv = right_components[key]
        components[key] = {
            "left_mean": _describe_values(lv)["mean"],
            "right_mean": _describe_values(rv)["mean"],
            "left_median": _describe_values(lv)["median"],
            "right_median": _describe_values(rv)["median"],
            "left_min": _describe_values(lv)["min"],
            "right_min": _describe_values(rv)["min"],
            "left_max": _describe_values(lv)["max"],
            "right_max": _describe_values(rv)["max"],
            **_paired_timing_stats(lv, rv),
        }

    return {
        "duration_seconds": total_summary,
        "components": components,
    }


def _extract_metric_from_run(run: "RunResult", metric_name: str) -> float:
    from ..core.timing_records import ALIASES, select_timing

    if metric_name in ALIASES:
        return select_timing(run, metric_name)[0]
    if metric_name in run.metrics:
        return float(run.metrics[metric_name])
    if run.history and metric_name in run.history[-1]:
        return float(run.history[-1][metric_name])
    raise StatisticalSpecError(f"metric '{metric_name}' not found for seed {run.seed}")


def _extract_initial_metric_from_run(run: "RunResult", metric_name: str) -> float | None:
    if metric_name == "best_fitness":
        initial = run.metrics.get("initial_fitness")
        if initial is not None:
            return float(initial)
    if run.history and metric_name in run.history[0]:
        return float(run.history[0][metric_name])
    return None


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return None
    return float(np.mean(np.asarray(valid, dtype=float)))


def paired_dataset_from_experiments(
    left: "ExperimentResult",
    right: "ExperimentResult",
    left_name: str,
    right_name: str,
    spec: StatisticalComparisonSpec,
) -> PairedMetricDataset:
    validate_spec(spec)

    left_by_seed = {int(run.seed): run for run in left.runs}
    right_by_seed = {int(run.seed): run for run in right.runs}
    if len(left_by_seed) != len(left.runs) or len(right_by_seed) != len(right.runs):
        raise StatisticalSpecError("Duplicate seeds in timing/objective records")
    common = [int(run.seed) for run in left.runs if int(run.seed) in right_by_seed]

    if not common:
        raise StatisticalSpecError("No paired seeds available")

    from ..core.timing_records import ALIASES, timing_values

    if spec.metric_name in ALIASES:
        _, lc = timing_values([left_by_seed[s] for s in common], spec.metric_name)
        _, rc = timing_values([right_by_seed[s] for s in common], spec.metric_name)
        if lc != rc:
            raise StatisticalSpecError("Incompatible timing contracts")

    left_values: list[float] = []
    right_values: list[float] = []
    left_start_values: list[float | None] = []
    right_start_values: list[float | None] = []

    for seed in common:
        lrun = left_by_seed[seed]
        rrun = right_by_seed[seed]

        if spec.hypothesis_kind == HypothesisKind.CLOSER_TO_OPTIMUM:
            l_gap = lrun.metrics.get("gap_to_optimum")
            r_gap = rrun.metrics.get("gap_to_optimum")
            if l_gap is not None and r_gap is not None:
                left_values.append(float(l_gap))
                right_values.append(float(r_gap))
                continue
            if spec.optimum_value is None:
                raise StatisticalSpecError(
                    "optimum_value required when gap_to_optimum is missing for CLOSER_TO_OPTIMUM"
                )
            lv = _extract_metric_from_run(lrun, spec.metric_name)
            rv = _extract_metric_from_run(rrun, spec.metric_name)
            left_values.append(abs(lv - float(spec.optimum_value)))
            right_values.append(abs(rv - float(spec.optimum_value)))
        else:
            left_values.append(_extract_metric_from_run(lrun, spec.metric_name))
            right_values.append(_extract_metric_from_run(rrun, spec.metric_name))

        left_start_values.append(_extract_initial_metric_from_run(lrun, spec.metric_name))
        right_start_values.append(_extract_initial_metric_from_run(rrun, spec.metric_name))

    source = "experiment_runs"
    if spec.hypothesis_kind == HypothesisKind.CLOSER_TO_OPTIMUM:
        source = "gap_to_optimum_or_derived_distance"

    left_end_mean = float(np.mean(np.asarray(left_values, dtype=float)))
    right_end_mean = float(np.mean(np.asarray(right_values, dtype=float)))
    left_start_mean = _mean_or_none(left_start_values)
    right_start_mean = _mean_or_none(right_start_values)

    left_delta = None
    right_delta = None
    delta_diff = None
    if left_start_mean is not None:
        left_delta = float(left_end_mean - left_start_mean)
    if right_start_mean is not None:
        right_delta = float(right_end_mean - right_start_mean)
    if left_delta is not None and right_delta is not None:
        delta_diff = float(left_delta - right_delta)

    metadata: dict[str, Any] = {
        "left_experiment": left.name,
        "right_experiment": right.name,
        "left_start_mean": left_start_mean,
        "right_start_mean": right_start_mean,
        "left_end_mean": left_end_mean,
        "right_end_mean": right_end_mean,
        "left_end_minus_start_mean": left_delta,
        "right_end_minus_start_mean": right_delta,
        "delta_end_minus_start_diff_mean": delta_diff,
    }

    if spec.include_value_lists:
        metadata["include_value_lists"] = True
        metadata["left_end_values"] = [float(v) for v in left_values]
        metadata["right_end_values"] = [float(v) for v in right_values]
        metadata["left_start_values"] = [
            float(v) if v is not None else None for v in left_start_values
        ]
        metadata["right_start_values"] = [
            float(v) if v is not None else None for v in right_start_values
        ]

    if spec.include_timing_stats:
        from ..core.timing_records import paired_timing_summary

        metadata["timing_summary"] = paired_timing_summary(
            [left_by_seed[s] for s in common], [right_by_seed[s] for s in common], spec.timing_phase
        )
        metadata["timing_summary"].update(
            seeds=common,
            left_unpaired_seeds=[s for s in left_by_seed if s not in right_by_seed],
            right_unpaired_seeds=[s for s in right_by_seed if s not in left_by_seed],
        )

    return PairedMetricDataset(
        label=f"{left_name}_vs_{right_name}",
        left_name=left_name,
        right_name=right_name,
        seeds=common,
        left_values=np.asarray(left_values, dtype=float),
        right_values=np.asarray(right_values, dtype=float),
        metric_name=spec.metric_name,
        metric_source=source,
        metadata=metadata,
    )


def paired_dataset_from_comparison(
    comparison: "ComparisonResult",
    left_pipeline: str,
    right_pipeline: str,
    spec: StatisticalComparisonSpec,
) -> PairedMetricDataset:
    if left_pipeline not in comparison.pipelines:
        raise KeyError(f"Unknown left pipeline '{left_pipeline}'")
    if right_pipeline not in comparison.pipelines:
        raise KeyError(f"Unknown right pipeline '{right_pipeline}'")

    return paired_dataset_from_experiments(
        left=comparison.pipelines[left_pipeline],
        right=comparison.pipelines[right_pipeline],
        left_name=left_pipeline,
        right_name=right_pipeline,
        spec=spec,
    )


def _load_start_and_final_by_seed_from_histories(
    csv_path: Path,
    metric_name: str,
) -> tuple[dict[int, float], dict[int, float]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if (
            not reader.fieldnames
            or "seed" not in reader.fieldnames
            or metric_name not in reader.fieldnames
        ):
            raise StatisticalSpecError(
                f"{csv_path} missing required columns: seed and {metric_name}"
            )

        generation_col = None
        for candidate in ("generation", "generation_counter", "final_generation"):
            if candidate in reader.fieldnames:
                generation_col = candidate
                break

        by_seed_first: dict[int, tuple[float, float]] = {}
        by_seed_last: dict[int, tuple[float, float]] = {}
        row_index = 0.0
        for row in reader:
            row_index += 1.0
            seed = int(row["seed"])
            value = float(row[metric_name])

            if generation_col is not None:
                try:
                    key = float(row[generation_col])
                except (TypeError, ValueError):
                    key = row_index
            else:
                key = row_index

            prev_first = by_seed_first.get(seed)
            if prev_first is None or key <= prev_first[0]:
                by_seed_first[seed] = (key, value)

            prev_last = by_seed_last.get(seed)
            if prev_last is None or key >= prev_last[0]:
                by_seed_last[seed] = (key, value)

    first = {seed: val for seed, (_, val) in by_seed_first.items()}
    last = {seed: val for seed, (_, val) in by_seed_last.items()}
    return first, last


def _load_gap_by_seed_from_summary(summary_path: Path) -> dict[int, float]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text())
    runs = payload.get("runs", [])
    out: dict[int, float] = {}
    for run in runs:
        seed = run.get("seed")
        metrics = run.get("metrics", {}) or {}
        if seed is None:
            continue
        gap = metrics.get("gap_to_optimum")
        if gap is None:
            continue
        out[int(seed)] = float(gap)
    return out


def _load_timings_by_seed_from_summary(summary_path: Path) -> dict[int, dict[str, float]]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text())
    runs = payload.get("runs", [])
    out: dict[int, dict[str, float]] = {}
    for run in runs:
        seed = run.get("seed")
        if seed is None:
            continue
        timings = run.get("timings") or {}
        if not isinstance(timings, dict):
            continue
        out[int(seed)] = {
            str(k): float(v) for k, v in timings.items() if v is not None and np.isfinite(float(v))
        }
    return out


def paired_dataset_from_artifacts(
    left_dir: Path,
    right_dir: Path,
    left_name: str,
    right_name: str,
    spec: StatisticalComparisonSpec,
) -> PairedMetricDataset:
    validate_spec(spec)

    from ..core.timing_records import ALIASES

    if spec.metric_name in ALIASES:
        from ..benchmarking.io import read_summary_json

        return paired_dataset_from_experiments(
            read_summary_json(left_dir / "summary.json"),
            read_summary_json(right_dir / "summary.json"),
            left_name,
            right_name,
            spec,
        )

    left_csv = left_dir / "histories_combined.csv"
    right_csv = right_dir / "histories_combined.csv"
    if not left_csv.exists() or not right_csv.exists():
        raise StatisticalSpecError("Both artifact dirs must contain histories_combined.csv")

    left_start, left_final = _load_start_and_final_by_seed_from_histories(
        left_csv, spec.metric_name
    )
    right_start, right_final = _load_start_and_final_by_seed_from_histories(
        right_csv, spec.metric_name
    )
    common = sorted(set(left_final) & set(right_final))

    if not common:
        raise StatisticalSpecError("No paired seeds available")

    source = "histories_combined"
    if spec.hypothesis_kind == HypothesisKind.CLOSER_TO_OPTIMUM:
        left_gap = _load_gap_by_seed_from_summary(left_dir / "summary.json")
        right_gap = _load_gap_by_seed_from_summary(right_dir / "summary.json")
        gap_common = sorted(set(left_gap) & set(right_gap) & set(common))
        if gap_common:
            seeds = gap_common
            left_values = np.asarray([left_gap[s] for s in seeds], dtype=float)
            right_values = np.asarray([right_gap[s] for s in seeds], dtype=float)
            source = "summary_gap_to_optimum"
        else:
            if spec.optimum_value is None:
                raise StatisticalSpecError(
                    "optimum_value required when summary gap_to_optimum is unavailable"
                )
            seeds = common
            opt = float(spec.optimum_value)
            left_values = np.asarray([abs(left_final[s] - opt) for s in seeds], dtype=float)
            right_values = np.asarray([abs(right_final[s] - opt) for s in seeds], dtype=float)
            source = "derived_distance_from_final_metric"
    else:
        seeds = common
        left_values = np.asarray([left_final[s] for s in seeds], dtype=float)
        right_values = np.asarray([right_final[s] for s in seeds], dtype=float)

    left_start_common = [left_start.get(s) for s in seeds]
    right_start_common = [right_start.get(s) for s in seeds]
    left_start_mean = _mean_or_none(left_start_common)
    right_start_mean = _mean_or_none(right_start_common)
    left_end_mean = float(np.mean(left_values))
    right_end_mean = float(np.mean(right_values))

    left_delta = None
    right_delta = None
    delta_diff = None
    if left_start_mean is not None:
        left_delta = float(left_end_mean - left_start_mean)
    if right_start_mean is not None:
        right_delta = float(right_end_mean - right_start_mean)
    if left_delta is not None and right_delta is not None:
        delta_diff = float(left_delta - right_delta)

    timing_summary: dict[str, Any] | None = None
    if spec.include_timing_stats:
        from ..core.timing_records import paired_timing_summary

        def load_timing_rows(path):
            with path.open() as stream:
                rows = json.load(stream).get("runs", [])
            if len({row["seed"] for row in rows}) != len(rows):
                raise ValueError("Duplicate timing seeds in artifact")
            return {row["seed"]: row for row in rows}

        try:
            lrows = load_timing_rows(left_dir / "summary.json")
            rrows = load_timing_rows(right_dir / "summary.json")
            timing_summary = paired_timing_summary(
                [lrows[s] for s in seeds], [rrows[s] for s in seeds], spec.timing_phase
            )
        except (OSError, ValueError, KeyError) as exc:
            timing_summary = {"status": "unavailable", "reason": str(exc), "count": 0}

    metadata: dict[str, Any] = {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "left_start_mean": left_start_mean,
        "right_start_mean": right_start_mean,
        "left_end_mean": left_end_mean,
        "right_end_mean": right_end_mean,
        "left_end_minus_start_mean": left_delta,
        "right_end_minus_start_mean": right_delta,
        "delta_end_minus_start_diff_mean": delta_diff,
    }

    if spec.include_value_lists:
        metadata["include_value_lists"] = True
        metadata["left_end_values"] = [float(v) for v in left_values.tolist()]
        metadata["right_end_values"] = [float(v) for v in right_values.tolist()]
        metadata["left_start_values"] = [
            float(v) if v is not None else None for v in left_start_common
        ]
        metadata["right_start_values"] = [
            float(v) if v is not None else None for v in right_start_common
        ]

    if timing_summary is not None:
        metadata["timing_summary"] = timing_summary

    return PairedMetricDataset(
        label=f"{left_name}_vs_{right_name}",
        left_name=left_name,
        right_name=right_name,
        seeds=seeds,
        left_values=left_values,
        right_values=right_values,
        metric_name=spec.metric_name,
        metric_source=source,
        metadata=metadata,
    )
