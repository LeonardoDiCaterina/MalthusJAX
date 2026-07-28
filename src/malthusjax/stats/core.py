from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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

    @property
    def n(self) -> int:
        return self.values.shape[0]


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
        return self.left.values - self.right.values


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
    multiple_testing: MultipleTestingPolicy = MultipleTestingPolicy.NONE
    include_tests: tuple[str, ...] = ("wilcoxon", "paired_t", "sign")
    include_value_lists: bool = False
    include_timing_stats: bool = True
    include_mean_summary: bool = False


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

    name: str
    statistic: float | None
    p_value: float | None
    alternative: str

    def passes(self, alpha: float = 0.05) -> bool | None:
        if self.p_value is None:
            return None
        return self.p_value > alpha


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
    decision_basis: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
