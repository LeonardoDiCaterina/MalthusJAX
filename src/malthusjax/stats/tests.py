import numpy as np
from scipy import stats

from malthusjax.stats.core import PairedSample, TestResult, TOSTResult


def _unavailable(name, alternative, reason, status="unavailable"):
    return TestResult(name, None, None, alternative, status, reason)


def _validate(sample, alternative):
    if (
        sample.left.values.ndim != 1
        or sample.right.values.ndim != 1
        or sample.left.values.shape != sample.right.values.shape
        or not np.isfinite(sample.left.values).all()
        or not np.isfinite(sample.right.values).all()
    ):
        raise ValueError("Inputs must be finite aligned one-dimensional samples")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("Invalid test alternative")
    if sample.n == 0:
        raise ValueError("Cannot test an empty sample")


def wilcoxon(sample: PairedSample, alternative="two-sided") -> TestResult:
    _validate(sample, alternative)
    if sample.n < 2:
        return _unavailable("wilcoxon", alternative, "At least two pairs required", "insufficient")
    if np.all(sample.diffs == 0):
        return _unavailable("wilcoxon", alternative, "All paired differences are exactly zero")
    try:
        result = stats.wilcoxon(sample.diffs, alternative=alternative, zero_method="pratt")
        return TestResult("wilcoxon", float(result.statistic), float(result.pvalue), alternative)
    except ValueError as exc:
        return _unavailable("wilcoxon", alternative, str(exc))


def paired_t(sample: PairedSample, alternative="two-sided") -> TestResult:
    _validate(sample, alternative)
    if sample.n < 2:
        return _unavailable("paired_t", alternative, "At least two pairs required", "insufficient")
    if np.var(sample.diffs) == 0:
        return _unavailable("paired_t", alternative, "Zero variance of paired differences")
    result = stats.ttest_rel(sample.left.values, sample.right.values, alternative=alternative)
    return TestResult("paired_t", float(result.statistic), float(result.pvalue), alternative)


def sign_test(sample: PairedSample, alternative="two-sided") -> TestResult:
    _validate(sample, alternative)
    if sample.n < 2:
        return _unavailable("sign", alternative, "At least two pairs required", "insufficient")
    positive = int(np.sum(sample.diffs > 0))
    negative = int(np.sum(sample.diffs < 0))
    n = positive + negative
    if n == 0:
        return _unavailable("sign", alternative, "All paired differences are exactly zero")
    result = stats.binomtest(positive, n, 0.5, alternative=alternative)
    return TestResult("sign", float(positive), float(result.pvalue), alternative)


def tost(sample: PairedSample, margin: float, alpha=0.05) -> TOSTResult:
    if not np.isfinite(margin) or margin <= 0 or not 0 < alpha < 1:
        raise ValueError("margin must be > 0 and finite; alpha must be in (0,1)")
    _validate(sample, "two-sided")
    reason = None
    status = "valid"
    if sample.n < 2:
        reason, status = "At least two pairs required", "insufficient"
    elif np.var(sample.diffs) == 0:
        reason, status = (
            "Zero variance of paired differences; parametric TOST unavailable",
            "unavailable",
        )
    if reason:
        return TOSTResult(
            margin, -margin, margin, None, None, None, None, None, None, status, reason
        )
    lower = stats.ttest_1samp(sample.diffs, popmean=-margin, alternative="greater")
    upper = stats.ttest_1samp(sample.diffs, popmean=margin, alternative="less")
    if not all(
        np.isfinite(x) for x in (lower.pvalue, upper.pvalue, lower.statistic, upper.statistic)
    ):
        return TOSTResult(
            margin,
            -margin,
            margin,
            None,
            None,
            None,
            None,
            None,
            None,
            "unavailable",
            "Nonfinite TOST output",
        )
    p = max(float(lower.pvalue), float(upper.pvalue))
    return TOSTResult(
        margin,
        -margin,
        margin,
        float(lower.statistic),
        float(upper.statistic),
        float(lower.pvalue),
        float(upper.pvalue),
        p,
        bool(p < alpha),
    )


# Compatibility function for existing comparison logic
def compute_standard_tests(
    left: np.ndarray,
    right: np.ndarray,
    alternative: str,
    include_tests: tuple[str, ...],
) -> dict[str, TestResult]:
    from malthusjax.stats.core import MetricVector

    sample = PairedSample(
        left=MetricVector("left", np.asarray(left, dtype=float)),
        right=MetricVector("right", np.asarray(right, dtype=float)),
    )
    out = {}
    if "wilcoxon" in include_tests:
        out["wilcoxon"] = wilcoxon(sample, alternative)
    if "paired_t" in include_tests:
        out["paired_t"] = paired_t(sample, alternative)
    if "sign" in include_tests:
        out["sign"] = sign_test(sample, alternative)
    return out


# Compatibility function for TOST
def compute_tost_paired(
    left: np.ndarray,
    right: np.ndarray,
    margin: float,
    alpha: float,
) -> TOSTResult:
    from malthusjax.stats.core import MetricVector

    sample = PairedSample(
        left=MetricVector("left", np.asarray(left, dtype=float)),
        right=MetricVector("right", np.asarray(right, dtype=float)),
    )
    return tost(sample, margin, alpha)
