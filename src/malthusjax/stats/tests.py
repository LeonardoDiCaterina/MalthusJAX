import numpy as np
from scipy import stats

from malthusjax.stats.core import PairedSample, TestResult, TOSTResult


def wilcoxon(sample: PairedSample, alternative: str = "two-sided") -> TestResult:
    """Compute Wilcoxon signed-rank test for paired sample."""
    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {sorted(valid_alternatives)}")

    if sample.n == 0:
        raise ValueError("Cannot compute Wilcoxon test on empty sample")

    if not np.isfinite(sample.left.values).all() or not np.isfinite(sample.right.values).all():
        raise ValueError("Inputs must be finite")

    try:
        res = stats.wilcoxon(
            sample.left.values, sample.right.values, alternative=alternative, zero_method="wilcox"
        )
        return TestResult(
            name="wilcoxon",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            alternative=alternative,
        )
    except ValueError:
        return TestResult(
            name="wilcoxon",
            statistic=None,
            p_value=None,
            alternative=alternative,
        )


def paired_t(sample: PairedSample, alternative: str = "two-sided") -> TestResult:
    """Compute paired t-test."""
    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {sorted(valid_alternatives)}")

    if sample.n == 0:
        raise ValueError("Cannot compute paired t-test on empty sample")

    if sample.n < 2:
        return TestResult(
            name="paired_t",
            statistic=None,
            p_value=None,
            alternative=alternative,
        )

    t_res = stats.ttest_rel(sample.left.values, sample.right.values, alternative=alternative)
    return TestResult(
        name="paired_t",
        statistic=float(t_res.statistic) if np.isfinite(t_res.statistic) else None,
        p_value=float(t_res.pvalue) if np.isfinite(t_res.pvalue) else None,
        alternative=alternative,
    )


def sign_test(sample: PairedSample, alternative: str = "two-sided") -> TestResult:
    """Compute simple sign test."""
    valid_alternatives = {"two-sided", "less", "greater"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {sorted(valid_alternatives)}")

    if sample.n == 0:
        raise ValueError("Cannot compute sign test on empty sample")

    diffs = sample.diffs
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n = n_pos + n_neg

    if n == 0:
        return TestResult(
            name="sign",
            statistic=None,
            p_value=None,
            alternative=alternative,
        )

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
    return TestResult(
        name="sign",
        statistic=float(k),
        p_value=float(s_res.pvalue),
        alternative=alternative,
    )


def tost(sample: PairedSample, margin: float, alpha: float = 0.05) -> TOSTResult:
    """Compute paired TOST for diff = left - right using +/- margin bounds."""
    if margin <= 0.0:
        raise ValueError("margin must be > 0")

    if sample.n == 0:
        raise ValueError("Cannot compute TOST on empty sample")

    if sample.n < 2:
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

    diffs = sample.diffs
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
