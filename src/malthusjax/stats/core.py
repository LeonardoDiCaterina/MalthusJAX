from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, cast

import numpy as np


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
class MetricVector:
    """A named 1D array of scalar observations."""

    name: str
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Metric samples must be finite one-dimensional arrays")
        object.__setattr__(self, "values", values)

    @property
    def n(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class PairedSample:
    """Two aligned MetricVectors from matched observations (e.g., same seed)."""

    left: MetricVector
    right: MetricVector
    label: str = ""

    def __post_init__(self) -> None:
        if self.left.n != self.right.n:
            raise ValueError(f"Paired sample size mismatch: {self.left.n} vs {self.right.n}")

    @property
    def n(self) -> int:
        return self.left.n

    @property
    def diffs(self) -> np.ndarray:
        return cast(np.ndarray, self.left.values - self.right.values)


@dataclass(frozen=True)
class RegressionDataset:
    """Labeled arrays for OLS modeling."""

    y: np.ndarray
    X: dict[str, np.ndarray]
    treatment_col: str = "is_treatment"
    interaction_col: str | None = None
    label: str = ""


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
    multiple_testing: MultipleTestingPolicy = MultipleTestingPolicy.HOLM
    include_tests: tuple[str, ...] = ("wilcoxon", "paired_t", "sign")
    include_value_lists: bool = False
    include_timing_stats: bool = True
    include_mean_summary: bool = False
    timing_phase: str = "prepared_first_execution"
    metric_orientation: str = "lower_is_better"


@dataclass
class PairedMetricDataset:
    """Paired metric arrays aligned by common seed set. (Legacy structure)"""

    label: str
    left_name: str
    right_name: str
    seeds: list[int]
    left_values: np.ndarray
    right_values: np.ndarray
    metric_name: str
    metric_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestResult:
    """Result for one statistical test."""

    __test__: ClassVar[bool] = False

    name: str
    statistic: float | None
    p_value: float | None
    alternative: str

    status: str = "valid"
    reason: str | None = None

    def __post_init__(self):
        if (
            self.p_value is None
            or not np.isfinite(self.p_value)
            or not 0 <= self.p_value <= 1
            or (self.statistic is not None and not np.isfinite(self.statistic))
        ):
            object.__setattr__(self, "statistic", None)
            object.__setattr__(self, "p_value", None)
            if self.status == "valid":
                object.__setattr__(self, "status", "unavailable")
            object.__setattr__(self, "reason", self.reason or "No finite valid p-value")

    def rejects_null(self, alpha: float = 0.05) -> bool | None:
        """Whether a valid p-value is below alpha; says nothing about equivalence."""
        if self.p_value is None:
            return None
        return self.p_value < alpha

    def passes(self, alpha: float = 0.05) -> bool | None:
        """Legacy alias for rejects_null, with schema-2 rejection semantics."""
        return self.rejects_null(alpha)


@dataclass(frozen=True)
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
    status: str = "valid"
    reason: str | None = None


@dataclass(frozen=True)
class EffectSizeResult:
    """Effect size outputs for paired differences."""

    cohen_dz: float | None
    rank_biserial: float | None


@dataclass(frozen=True)
class DiagnosticResult:
    """Output of a regression diagnostic."""

    name: str
    statistic: float | None
    p_value: float | None

    def passes(self, alpha: float = 0.05) -> bool | None:
        if self.p_value is None:
            return None
        return self.p_value > alpha


@dataclass
class OLSResult:
    """Output of an OLS regression fit."""

    coefficients: dict[str, float]
    p_values: dict[str, float]
    robust_p_values: dict[str, dict[str, float]]
    r_squared: float
    diagnostics: list[DiagnosticResult]
    label: str = ""


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
    decision_reliable: bool | None
    decision_basis: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "valid"
    reason: str | None = None
    conclusion: str = "unavailable"
    raw_decision_pass: bool | None = None
    raw_conclusion: str = "unavailable"
    adjusted_p_value: float | None = None
    decision_scope: str = "raw"


@dataclass
class StatisticalSuiteResult:
    """Collection of comparison results under a shared spec."""

    spec: StatisticalComparisonSpec
    results: list[StatisticalComparisonResult]
    adjusted_p_values: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from malthusjax.stats.io import suite_to_dict

        return suite_to_dict(self)

    def to_markdown(self) -> str:
        from malthusjax.stats.io import suite_to_markdown

        return suite_to_markdown(self)


def validate_spec(spec: StatisticalComparisonSpec) -> None:
    """Validate cross-field constraints for a comparison specification.

    Raises
    ------
    StatisticalSpecError
        If required fields are missing or inconsistent.
    """
    if spec.metric_orientation not in {"lower_is_better", "higher_is_better"}:
        raise StatisticalSpecError("Unknown metric orientation")
    if not spec.metric_name:
        raise StatisticalSpecError("metric_name must be non-empty")

    if type(spec.min_paired_seeds) is not int or spec.min_paired_seeds < 2:
        raise StatisticalSpecError("min_paired_seeds must be >= 2 (not a power guarantee)")

    if not (0.0 < spec.alpha < 1.0):
        raise StatisticalSpecError("alpha must be in (0, 1)")

    if set(spec.include_tests) - {"wilcoxon", "paired_t", "sign"}:
        raise StatisticalSpecError("Unknown statistical test")
    if not spec.include_tests:
        raise StatisticalSpecError("include_tests must contain at least one test")

    if spec.sidedness == Sidedness.ONE_SIDED and spec.expected_direction is None:
        raise StatisticalSpecError("expected_direction is required for one-sided tests")

    if spec.hypothesis_kind == HypothesisKind.EQUIVALENCE:
        if spec.equivalence_margin is None:
            raise StatisticalSpecError("equivalence_margin is required for EQUIVALENCE")
        if not np.isfinite(spec.equivalence_margin) or spec.equivalence_margin <= 0.0:
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
