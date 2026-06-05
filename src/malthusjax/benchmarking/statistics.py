"""Statistical comparison layer for parity and benchmark analysis.

This module defines the abstraction scaffold for hypothesis-driven paired
comparisons across pipelines. It intentionally starts as an API-first skeleton
with concrete dataclasses, enums, and method signatures.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult


class StatisticalSpecError(ValueError):
    """Raised when a StatisticalComparisonSpec is invalid."""


class HypothesisKind(str, Enum):
    """Top-level hypothesis family."""

    LOCATION_SHIFT = "location_shift"
    CLOSER_TO_OPTIMUM = "closer_to_optimum"
    EQUIVALENCE = "equivalence"


class Sidedness(str, Enum):
    """Directional mode for non-equivalence tests."""

    TWO_SIDED = "two_sided"
    ONE_SIDED = "one_sided"


class ExpectedDirection(str, Enum):
    """Expected direction when one-sided mode is used."""

    LEFT_LT_RIGHT = "left_lt_right"
    LEFT_GT_RIGHT = "left_gt_right"


class MultipleTestingPolicy(str, Enum):
    """Policy for suite-level p-value correction."""

    NONE = "none"
    HOLM = "holm"
    FDR_BH = "fdr_bh"


@dataclass(frozen=True)
class StatisticalComparisonSpec:
    """Configuration for one paired statistical comparison run."""

    metric_name: str = "best_fitness"
    hypothesis_kind: HypothesisKind = HypothesisKind.LOCATION_SHIFT
    sidedness: Sidedness = Sidedness.TWO_SIDED
    expected_direction: ExpectedDirection = ExpectedDirection.LEFT_LT_RIGHT
    optimum_value: float | None = None
    equivalence_margin: float | None = None
    min_paired_seeds: int = 10
    alpha: float = 0.05
    multiple_testing: MultipleTestingPolicy = MultipleTestingPolicy.NONE
    include_tests: tuple[str, ...] = ("wilcoxon", "paired_t", "sign")
    include_value_lists: bool = False
    include_timing_stats: bool = True
    include_mean_summary: bool = False


@dataclass
class PairedMetricDataset:
    """Paired metric arrays aligned by common seed set."""

    label: str
    left_name: str
    right_name: str
    seeds: list[int]
    left_values: np.ndarray
    right_values: np.ndarray
    metric_name: str
    metric_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result for one statistical test."""

    name: str
    statistic: float | None
    p_value: float | None
    alternative: str


@dataclass
class TOSTResult:
    """Structured output for paired TOST equivalence testing."""

    margin: float
    lower_bound: float
    upper_bound: float
    t_stat_lower: float | None
    t_stat_upper: float | None
    p_value_lower: float | None
    p_value_upper: float | None
    p_value_max: float | None
    equivalent: bool | None


@dataclass
class EffectSizeResult:
    """Effect size outputs for paired differences."""

    cohen_dz: float | None
    rank_biserial: float | None


@dataclass
class StatisticalComparisonResult:
    """One fully evaluated paired-comparison result."""

    label: str
    hypothesis_text: str
    n_paired: int
    wins_left: int
    wins_right: int
    ties: int
    left_mean: float
    right_mean: float
    mean_diff_left_minus_right: float
    median_diff_left_minus_right: float
    tests: dict[str, TestResult]
    tost: TOSTResult | None
    effects: EffectSizeResult
    alpha: float
    decision_pass: bool | None
    decision_basis: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalSuiteResult:
    """Collection of comparison results under a shared spec."""

    spec: StatisticalComparisonSpec
    results: list[StatisticalComparisonResult]
    adjusted_p_values: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the suite."""
        def _test_to_dict(test: TestResult) -> dict[str, Any]:
            return {
                "name": test.name,
                "statistic": test.statistic,
                "p_value": test.p_value,
                "alternative": test.alternative,
            }

        def _tost_to_dict(tost: TOSTResult | None) -> dict[str, Any] | None:
            if tost is None:
                return None
            return {
                "margin": tost.margin,
                "lower_bound": tost.lower_bound,
                "upper_bound": tost.upper_bound,
                "t_stat_lower": tost.t_stat_lower,
                "t_stat_upper": tost.t_stat_upper,
                "p_value_lower": tost.p_value_lower,
                "p_value_upper": tost.p_value_upper,
                "p_value_max": tost.p_value_max,
                "equivalent": tost.equivalent,
            }

        def _result_to_dict(result: StatisticalComparisonResult) -> dict[str, Any]:
            return {
                "label": result.label,
                "hypothesis_text": result.hypothesis_text,
                "n_paired": result.n_paired,
                "wins_left": result.wins_left,
                "wins_right": result.wins_right,
                "ties": result.ties,
                "left_mean": result.left_mean,
                "right_mean": result.right_mean,
                "mean_diff_left_minus_right": result.mean_diff_left_minus_right,
                "median_diff_left_minus_right": result.median_diff_left_minus_right,
                "tests": {k: _test_to_dict(v) for k, v in result.tests.items()},
                "tost": _tost_to_dict(result.tost),
                "effects": {
                    "cohen_dz": result.effects.cohen_dz,
                    "rank_biserial": result.effects.rank_biserial,
                },
                "alpha": result.alpha,
                "decision_pass": result.decision_pass,
                "decision_basis": result.decision_basis,
                "metadata": result.metadata,
            }

        return {
            "spec": {
                "metric_name": self.spec.metric_name,
                "hypothesis_kind": self.spec.hypothesis_kind.value,
                "sidedness": self.spec.sidedness.value,
                "expected_direction": self.spec.expected_direction.value,
                "optimum_value": self.spec.optimum_value,
                "equivalence_margin": self.spec.equivalence_margin,
                "min_paired_seeds": self.spec.min_paired_seeds,
                "alpha": self.spec.alpha,
                "multiple_testing": self.spec.multiple_testing.value,
                "include_tests": list(self.spec.include_tests),
                "include_value_lists": self.spec.include_value_lists,
                "include_timing_stats": self.spec.include_timing_stats,
                "include_mean_summary": self.spec.include_mean_summary,
            },
            "results": [_result_to_dict(r) for r in self.results],
            "adjusted_p_values": self.adjusted_p_values,
        }

    def to_markdown(self) -> str:
        """Render suite results as a markdown summary."""
        def _fmt(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"{value:.6g}"

        lines: list[str] = []
        lines.append("# Statistical Suite Summary")
        lines.append("")
        if self.spec.include_mean_summary:
            lines.append(
                "| Label | n | Left Start | Right Start | Left End | Right End | "
                "Diff End (L-R) | Left Delta | Right Delta | Delta Diff (L-R) | "
                "Cohen dz | Primary p | Decision | Basis |"
            )
            lines.append(
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
            )
        else:
            lines.append("| Label | n | Cohen dz | Primary p | Decision | Basis |")
            lines.append("|---|---:|---:|---:|---|---|")

        for r in self.results:
            primary_p = None
            if r.decision_basis == "tost" and r.tost is not None:
                primary_p = r.tost.p_value_max
            elif r.decision_basis.startswith("wilcoxon") and "wilcoxon" in r.tests:
                primary_p = r.tests["wilcoxon"].p_value
            elif r.decision_basis.startswith("paired_t") and "paired_t" in r.tests:
                primary_p = r.tests["paired_t"].p_value

            p_text = "n/a" if primary_p is None else f"{primary_p:.6g}"
            if r.decision_pass is True:
                decision = "pass"
            elif r.decision_pass is False:
                decision = "fail"
            else:
                decision = "n/a"

            left_start = r.metadata.get("left_start_mean")
            right_start = r.metadata.get("right_start_mean")
            left_delta = r.metadata.get("left_end_minus_start_mean")
            right_delta = r.metadata.get("right_end_minus_start_mean")
            delta_diff = r.metadata.get("delta_end_minus_start_diff_mean")

            if self.spec.include_mean_summary:
                lines.append(
                    f"| {r.label} | {r.n_paired} | {_fmt(left_start)} | {_fmt(right_start)} | "
                    f"{r.left_mean:.6g} | {r.right_mean:.6g} | "
                    f"{r.mean_diff_left_minus_right:.6g} | "
                    f"{_fmt(left_delta)} | {_fmt(right_delta)} | {_fmt(delta_diff)} | "
                    f"{_fmt(r.effects.cohen_dz)} | {p_text} | {decision} | {r.decision_basis} |"
                )
            else:
                lines.append(
                    f"| {r.label} | {r.n_paired} | {_fmt(r.effects.cohen_dz)} | "
                    f"{p_text} | {decision} | {r.decision_basis} |"
                )

        for r in self.results:
            left_start = r.metadata.get("left_start_mean")
            right_start = r.metadata.get("right_start_mean")
            left_delta = r.metadata.get("left_end_minus_start_mean")
            right_delta = r.metadata.get("right_end_minus_start_mean")
            delta_diff = r.metadata.get("delta_end_minus_start_diff_mean")
            if (
                left_start is None
                or right_start is None
                or left_delta is None
                or right_delta is None
                or delta_diff is None
            ):
                continue

            lines.append("")
            lines.append(f"## Progress Context: {r.label}")
            lines.append("")
            lines.append(
                "| Left Start | Right Start | Left Delta | Right Delta | "
                "Delta Diff (L-R) |"
            )
            lines.append("|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {_fmt(left_start)} | {_fmt(right_start)} | {_fmt(left_delta)} | "
                f"{_fmt(right_delta)} | {_fmt(delta_diff)} |"
            )

            if r.metadata.get("optimum_value") is not None:
                lines.append("")
                lines.append(
                    "| Optimum | Left Start Dist | Right Start Dist | Left End Dist | "
                    "Right End Dist | Left Closed % | Right Closed % | "
                    "Closed %-pt Diff (L-R) |"
                )
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
                lines.append(
                    f"| {_fmt(r.metadata.get('optimum_value'))} | "
                    f"{_fmt(r.metadata.get('left_start_distance_to_optimum'))} | "
                    f"{_fmt(r.metadata.get('right_start_distance_to_optimum'))} | "
                    f"{_fmt(r.metadata.get('left_end_distance_to_optimum'))} | "
                    f"{_fmt(r.metadata.get('right_end_distance_to_optimum'))} | "
                    f"{_fmt(r.metadata.get('left_distance_closed_pct'))} | "
                    f"{_fmt(r.metadata.get('right_distance_closed_pct'))} | "
                    f"{_fmt(r.metadata.get('distance_closed_pct_point_diff_left_minus_right'))} |"
                )

        if self.adjusted_p_values:
            lines.append("")
            lines.append("## Adjusted P-values")
            lines.append("")
            lines.append("| Label | Key | Adjusted p |")
            lines.append("|---|---|---:|")
            for label, mapping in self.adjusted_p_values.items():
                for key, value in mapping.items():
                    lines.append(f"| {label} | {key} | {value:.6g} |")

        for r in self.results:
            timing_summary = r.metadata.get("timing_summary")
            if isinstance(timing_summary, dict):
                total = timing_summary.get("duration_seconds", {})
                lines.append("")
                lines.append(f"## Timing Summary: {r.label}")
                lines.append("")
                lines.append(
                    "| Timing | Left Mean | Right Mean | Left Median | Right Median | "
                    "Left Min | Right Min | Left Max | Right Max | "
                    "Paired Diff Mean (L-R) | Wilcoxon p |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                if isinstance(total, dict):
                    lines.append(
                        f"| duration_seconds | {_fmt(total.get('left_mean'))} | "
                        f"{_fmt(total.get('right_mean'))} | "
                        f"{_fmt(total.get('left_median'))} | "
                        f"{_fmt(total.get('right_median'))} | "
                        f"{_fmt(total.get('left_min'))} | {_fmt(total.get('right_min'))} | "
                        f"{_fmt(total.get('left_max'))} | {_fmt(total.get('right_max'))} | "
                        f"{_fmt(total.get('paired_diff_mean_left_minus_right'))} | "
                        f"{_fmt(total.get('paired_wilcoxon_p_value'))} |"
                    )
                components = timing_summary.get("components", {})
                if isinstance(components, dict):
                    for key, comp in components.items():
                        if not isinstance(comp, dict):
                            continue
                        lines.append(
                            f"| {key} | {_fmt(comp.get('left_mean'))} | "
                            f"{_fmt(comp.get('right_mean'))} | "
                            f"{_fmt(comp.get('left_median'))} | "
                            f"{_fmt(comp.get('right_median'))} | "
                            f"{_fmt(comp.get('left_min'))} | {_fmt(comp.get('right_min'))} | "
                            f"{_fmt(comp.get('left_max'))} | {_fmt(comp.get('right_max'))} | "
                            f"{_fmt(comp.get('paired_diff_mean_left_minus_right'))} | "
                            f"{_fmt(comp.get('paired_wilcoxon_p_value'))} |"
                        )

        for r in self.results:
            if not bool(r.metadata.get("include_value_lists", False)):
                continue
            left_vals = r.metadata.get("left_end_values")
            right_vals = r.metadata.get("right_end_values")
            if left_vals is None or right_vals is None:
                continue
            lines.append("")
            lines.append(f"## Raw Values: {r.label}")
            lines.append("")
            lines.append(f"- left_end_values: {left_vals}")
            lines.append(f"- right_end_values: {right_vals}")
            left_start_vals = r.metadata.get("left_start_values")
            right_start_vals = r.metadata.get("right_start_values")
            if left_start_vals is not None and right_start_vals is not None:
                lines.append(f"- left_start_values: {left_start_vals}")
                lines.append(f"- right_start_values: {right_start_vals}")

        return "\n".join(lines)


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


def validate_spec(spec: StatisticalComparisonSpec) -> None:
    """Validate cross-field constraints for a comparison specification.

    Raises
    ------
    StatisticalSpecError
        If required fields are missing or inconsistent.
    """
    if not spec.metric_name:
        raise StatisticalSpecError("metric_name must be non-empty")

    if spec.min_paired_seeds < 1:
        raise StatisticalSpecError("min_paired_seeds must be >= 1")

    if not (0.0 < spec.alpha < 1.0):
        raise StatisticalSpecError("alpha must be in (0, 1)")

    if not spec.include_tests:
        raise StatisticalSpecError("include_tests must contain at least one test")

    if spec.sidedness == Sidedness.ONE_SIDED and spec.expected_direction is None:
        raise StatisticalSpecError("expected_direction is required for one-sided tests")

    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        if spec.equivalence_margin is None:
            raise StatisticalSpecError("equivalence_margin is required for EQUIVALENCE")
        if spec.equivalence_margin <= 0.0:
            raise StatisticalSpecError("equivalence_margin must be > 0 for EQUIVALENCE")


def infer_scipy_alternative(
    sidedness: Sidedness,
    direction: ExpectedDirection,
) -> str:
    """Map enum options to scipy-compatible alternatives.

    Returns one of: "two-sided", "less", "greater".
    """
    if sidedness == Sidedness.TWO_SIDED:
        return "two-sided"

    if direction == ExpectedDirection.LEFT_LT_RIGHT:
        return "less"
    return "greater"


def compute_tost_paired(
    left: np.ndarray,
    right: np.ndarray,
    margin: float,
    alpha: float,
) -> TOSTResult:
    """Compute paired TOST for diff = left - right using +/- margin bounds."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)

    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left/right shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    if margin <= 0.0:
        raise ValueError("margin must be > 0")

    diffs = left_arr - right_arr
    n = int(diffs.size)
    if n < 2:
        return TOSTResult(
            margin=float(margin),
            lower_bound=float(-margin),
            upper_bound=float(margin),
            t_stat_lower=None,
            t_stat_upper=None,
            p_value_lower=None,
            p_value_upper=None,
            p_value_max=None,
            equivalent=None,
        )

    lower_test = stats.ttest_1samp(diffs, popmean=-margin, alternative="greater")
    upper_test = stats.ttest_1samp(diffs, popmean=margin, alternative="less")

    p_lower = float(lower_test.pvalue)
    p_upper = float(upper_test.pvalue)
    p_max = max(p_lower, p_upper)

    return TOSTResult(
        margin=float(margin),
        lower_bound=float(-margin),
        upper_bound=float(margin),
        t_stat_lower=float(lower_test.statistic),
        t_stat_upper=float(upper_test.statistic),
        p_value_lower=p_lower,
        p_value_upper=p_upper,
        p_value_max=p_max,
        equivalent=bool(p_max < alpha),
    )


def compute_standard_tests(
    left: np.ndarray,
    right: np.ndarray,
    alternative: str,
    include_tests: tuple[str, ...],
) -> dict[str, TestResult]:
    """Compute standard requested tests (wilcoxon, paired_t, sign)."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)

    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left/right shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {sorted(valid_alternatives)}")

    out: dict[str, TestResult] = {}
    tests_set = set(include_tests)

    if "wilcoxon" in tests_set:
        try:
            res = stats.wilcoxon(left_arr, right_arr, alternative=alternative, zero_method="wilcox")
            out["wilcoxon"] = TestResult(
                name="wilcoxon",
                statistic=float(res.statistic),
                p_value=float(res.pvalue),
                alternative=alternative,
            )
        except ValueError:
            out["wilcoxon"] = TestResult(
                name="wilcoxon",
                statistic=None,
                p_value=None,
                alternative=alternative,
            )

    if "paired_t" in tests_set:
        if left_arr.size >= 2:
            t_res = stats.ttest_rel(left_arr, right_arr, alternative=alternative)
            out["paired_t"] = TestResult(
                name="paired_t",
                statistic=float(t_res.statistic),
                p_value=float(t_res.pvalue),
                alternative=alternative,
            )
        else:
            out["paired_t"] = TestResult(
                name="paired_t",
                statistic=None,
                p_value=None,
                alternative=alternative,
            )

    if "sign" in tests_set:
        diffs = left_arr - right_arr
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n = n_pos + n_neg

        if n == 0:
            out["sign"] = TestResult(
                name="sign",
                statistic=None,
                p_value=None,
                alternative=alternative,
            )
        else:
            if alternative == "two-sided":
                k = n_pos
                sign_alt = "two-sided"
            elif alternative == "greater":
                k = n_pos
                sign_alt = "greater"
            else:
                k = n_neg
                sign_alt = "greater"

            s_res = stats.binomtest(k, n, 0.5, alternative=sign_alt)
            out["sign"] = TestResult(
                name="sign",
                statistic=float(k),
                p_value=float(s_res.pvalue),
                alternative=alternative,
            )

    return out


def compute_effect_sizes(left: np.ndarray, right: np.ndarray) -> EffectSizeResult:
    """Compute paired effect sizes for left-right differences."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)

    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left/right shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    diffs = left_arr - right_arr

    if diffs.size < 2:
        dz = float("nan")
    else:
        std = float(np.std(diffs, ddof=1))
        dz = 0.0 if np.isclose(std, 0.0) else float(np.mean(diffs) / std)

    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        rank_biserial = None
    else:
        n_pos = int(np.sum(nonzero > 0))
        n_neg = int(np.sum(nonzero < 0))
        rank_biserial = float((n_pos - n_neg) / (n_pos + n_neg))

    return EffectSizeResult(cohen_dz=dz, rank_biserial=rank_biserial)


def apply_decision_rule(
    *,
    spec: StatisticalComparisonSpec,
    tests: dict[str, TestResult],
    tost: TOSTResult | None,
) -> tuple[bool | None, str, str | None]:
    """Return decision tuple: (pass, basis, error)."""
    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        if tost is None or tost.p_value_max is None:
            return None, "tost", "TOST result unavailable"
        return bool(tost.p_value_max < spec.alpha), "tost", None

    wilcoxon = tests.get("wilcoxon")
    if wilcoxon is not None and wilcoxon.p_value is not None:
        return bool(wilcoxon.p_value > spec.alpha), f"wilcoxon_{wilcoxon.alternative}", None

    paired_t = tests.get("paired_t")
    if paired_t is not None and paired_t.p_value is not None:
        return bool(paired_t.p_value > spec.alpha), f"paired_t_{paired_t.alternative}", None

    return None, "none", "No decision-eligible p-value available"


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
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left/right shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    if left_arr.size < spec.min_paired_seeds:
        raise StatisticalSpecError(
            f"paired sample size {left_arr.size} is below min_paired_seeds={spec.min_paired_seeds}"
        )

    alternative = infer_scipy_alternative(spec.sidedness, spec.expected_direction)

    tests = compute_standard_tests(
        left_arr,
        right_arr,
        alternative=alternative,
        include_tests=spec.include_tests,
    )

    tost: TOSTResult | None = None
    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        # validate_spec guarantees margin is present for equivalence.
        tost = compute_tost_paired(
            left_arr,
            right_arr,
            margin=float(spec.equivalence_margin),
            alpha=spec.alpha,
        )

    effects = compute_effect_sizes(left_arr, right_arr)
    decision_pass, decision_basis, decision_error = apply_decision_rule(
        spec=spec,
        tests=tests,
        tost=tost,
    )

    diffs = left_arr - right_arr
    wins_left = int(np.sum(left_arr < right_arr))
    wins_right = int(np.sum(right_arr < left_arr))
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
            "alternative": alternative,
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
                "tost_equivalent": tost.equivalent,
            }
        )

    if tost is not None:
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
        decision_basis=decision_basis,
        metadata=out_metadata,
    )


def adjust_pvalues(
    p_values: list[float],
    policy: MultipleTestingPolicy,
) -> list[float]:
    """Adjust p-values according to the requested multiple-testing policy."""
    if policy == MultipleTestingPolicy.NONE:
        return list(p_values)

    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    if n == 0:
        return []

    finite_mask = np.isfinite(p)
    adjusted = np.full(n, np.nan, dtype=float)
    idx = np.where(finite_mask)[0]
    if idx.size == 0:
        return adjusted.tolist()

    pf = p[idx]
    m = int(pf.size)

    order = np.argsort(pf)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    p_sorted = pf[order]

    if policy == MultipleTestingPolicy.HOLM:
        raw = np.array([(m - i) * p_sorted[i] for i in range(m)], dtype=float)
        adj_sorted = np.maximum.accumulate(raw)
        adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    elif policy == MultipleTestingPolicy.FDR_BH:
        raw = np.array([m * p_sorted[i] / (i + 1) for i in range(m)], dtype=float)
        adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    else:
        raise NotImplementedError(f"Unsupported multiple-testing policy: {policy.value}")

    adj_f = adj_sorted[inv_order]
    adjusted[idx] = adj_f
    return adjusted.tolist()


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


def _extract_metric_from_run(run: "RunResult", metric_name: str) -> float:
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
    """Build a paired metric dataset from two ExperimentResult objects."""
    validate_spec(spec)

    left_by_seed = {int(run.seed): run for run in left.runs}
    right_by_seed = {int(run.seed): run for run in right.runs}
    common = sorted(set(left_by_seed) & set(right_by_seed))

    if len(common) < spec.min_paired_seeds:
        raise StatisticalSpecError(
            f"paired seed count {len(common)} below min_paired_seeds={spec.min_paired_seeds}"
        )

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

    left_total_timing: list[float] = []
    right_total_timing: list[float] = []
    left_component_timing: dict[str, list[float]] = {}
    right_component_timing: dict[str, list[float]] = {}

    if spec.include_timing_stats:
        for seed in common:
            lrun = left_by_seed[seed]
            rrun = right_by_seed[seed]

            if lrun.duration_seconds is not None and rrun.duration_seconds is not None:
                left_total_timing.append(float(lrun.duration_seconds))
                right_total_timing.append(float(rrun.duration_seconds))

            if lrun.timings and rrun.timings:
                common_timing_keys = sorted(set(lrun.timings) & set(rrun.timings))
                for key in common_timing_keys:
                    lv = lrun.timings.get(key)
                    rv = rrun.timings.get(key)
                    if lv is None or rv is None:
                        continue
                    left_component_timing.setdefault(key, []).append(float(lv))
                    right_component_timing.setdefault(key, []).append(float(rv))

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

    metadata = {
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
        metadata["timing_summary"] = _build_timing_summary(
            left_total=left_total_timing,
            right_total=right_total_timing,
            left_components=left_component_timing,
            right_components=right_component_timing,
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
    """Build a paired dataset from two named pipelines in a ComparisonResult."""
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


def _load_final_by_seed_from_histories(csv_path: Path, metric_name: str) -> dict[int, float]:
    _, final = _load_start_and_final_by_seed_from_histories(csv_path, metric_name)
    return final


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
            str(k): float(v)
            for k, v in timings.items()
            if v is not None and np.isfinite(float(v))
        }
    return out


def paired_dataset_from_artifacts(
    left_dir: Path,
    right_dir: Path,
    left_name: str,
    right_name: str,
    spec: StatisticalComparisonSpec,
) -> PairedMetricDataset:
    """Build a paired dataset from two artifact directories.

    Directories are expected to contain at least histories_combined.csv.
    summary.json is optional but used for gap_to_optimum extraction.
    """
    validate_spec(spec)

    left_csv = left_dir / "histories_combined.csv"
    right_csv = right_dir / "histories_combined.csv"
    if not left_csv.exists() or not right_csv.exists():
        raise StatisticalSpecError("Both artifact dirs must contain histories_combined.csv")

    left_start, left_final = _load_start_and_final_by_seed_from_histories(
        left_csv,
        spec.metric_name,
    )
    right_start, right_final = _load_start_and_final_by_seed_from_histories(
        right_csv,
        spec.metric_name,
    )
    common = sorted(set(left_final) & set(right_final))

    if len(common) < spec.min_paired_seeds:
        raise StatisticalSpecError(
            f"paired seed count {len(common)} below min_paired_seeds={spec.min_paired_seeds}"
        )

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
        left_timings = _load_timings_by_seed_from_summary(left_dir / "summary.json")
        right_timings = _load_timings_by_seed_from_summary(right_dir / "summary.json")

        left_total_timing: list[float] = []
        right_total_timing: list[float] = []
        left_component_timing: dict[str, list[float]] = {}
        right_component_timing: dict[str, list[float]] = {}

        for seed in seeds:
            lt = left_timings.get(seed)
            rt = right_timings.get(seed)
            if not lt or not rt:
                continue

            left_total_timing.append(float(sum(lt.values())))
            right_total_timing.append(float(sum(rt.values())))

            common_keys = sorted(set(lt) & set(rt))
            for key in common_keys:
                left_component_timing.setdefault(key, []).append(float(lt[key]))
                right_component_timing.setdefault(key, []).append(float(rt[key]))

        timing_summary = _build_timing_summary(
            left_total=left_total_timing,
            right_total=right_total_timing,
            left_components=left_component_timing,
            right_components=right_component_timing,
        )

    metadata = {
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


class StatisticalComparator:
    """High-level interface for paired and suite-level statistical comparison."""

    def compare_paired(
        self,
        dataset: PairedMetricDataset,
        spec: StatisticalComparisonSpec,
    ) -> StatisticalComparisonResult:
        """Run a single paired comparison for one aligned dataset."""
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
        results: list[StatisticalComparisonResult] = [
            self.compare_paired(ds, spec) for ds in datasets
        ]
        suite = StatisticalSuiteResult(spec=spec, results=results)
        if spec.multiple_testing != MultipleTestingPolicy.NONE:
            return self.adjust_suite_pvalues(suite)
        return suite

    def adjust_suite_pvalues(self, suite: StatisticalSuiteResult) -> StatisticalSuiteResult:
        """Apply suite-level p-value correction to an existing suite result."""
        # v1 default: no-op for NONE.
        if suite.spec.multiple_testing == MultipleTestingPolicy.NONE:
            return suite

        primary_pvals: list[float] = []
        labels: list[str] = []
        for result in suite.results:
            pval: float | None = None
            if result.decision_basis == "tost" and result.tost is not None:
                pval = result.tost.p_value_max
            elif result.decision_basis.startswith("wilcoxon") and "wilcoxon" in result.tests:
                pval = result.tests["wilcoxon"].p_value
            elif result.decision_basis.startswith("paired_t") and "paired_t" in result.tests:
                pval = result.tests["paired_t"].p_value

            if pval is None:
                continue
            primary_pvals.append(float(pval))
            labels.append(result.label)

        adjusted = adjust_pvalues(primary_pvals, suite.spec.multiple_testing)
        suite.adjusted_p_values = {
            label: {"primary": float(adj)}
            for label, adj in zip(labels, adjusted)
            if np.isfinite(adj)
        }
        return suite
