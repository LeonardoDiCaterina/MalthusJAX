import numpy as np
import pytest

from malthusjax.stats.core import TestResult, TOSTResult
from malthusjax.stats.tests import paired_t, sign_test, tost, wilcoxon


def test_wilcoxon_null_not_rejected(paired_equal):
    res = wilcoxon(paired_equal)
    assert isinstance(res, TestResult)
    assert res.name == "wilcoxon"
    assert res.p_value > 0.05


def test_wilcoxon_shift_detected(paired_shifted):
    res = wilcoxon(paired_shifted, alternative="less")
    assert res.p_value < 0.05


def test_wilcoxon_identical_arrays(paired_identical):
    res = wilcoxon(paired_identical)
    assert res.p_value == 1.0
    assert res.statistic == 0.0


def test_wilcoxon_n1(paired_single):
    res = wilcoxon(paired_single)
    assert res.p_value == 1.0


def test_wilcoxon_empty(paired_empty):
    with pytest.raises(ValueError, match="empty sample"):
        wilcoxon(paired_empty)


@pytest.mark.parametrize("test_fn", [wilcoxon, paired_t, lambda s: tost(s, margin=1.0)])
def test_tests_with_nans(paired_equal, test_fn):
    arr = paired_equal.left.values.copy()
    arr[0] = np.nan
    from malthusjax.stats.core import MetricVector, PairedSample

    ps = PairedSample(MetricVector("l", arr), paired_equal.right)
    with pytest.raises(ValueError, match="finite"):
        test_fn(ps)


def test_wilcoxon_ties_pratt_method():
    """Verify wilcoxon correctly uses zero_method='pratt' to penalize exact ties."""
    from malthusjax.stats.core import MetricVector, PairedSample
    from scipy import stats

    # 10 ties, 5 differing elements to ensure pratt and wilcox diverge
    left_vals = np.zeros(15)
    right_vals = np.array([0.0]*10 + [1.1, 2.1, 3.1, 4.1, 5.1])
    ps = PairedSample(
        left=MetricVector("left", left_vals),
        right=MetricVector("right", right_vals)
    )

    # Calculate truth using direct scipy call with pratt
    expected_res = stats.wilcoxon(left_vals, right_vals, alternative="two-sided", zero_method="pratt")
    
    # Calculate truth using direct scipy call with wilcox for comparison (not used in assertion)
    wilcox_res = stats.wilcoxon(left_vals, right_vals, alternative="two-sided", zero_method="wilcox")

    res = wilcoxon(ps, alternative="two-sided")
    
    assert res.statistic == float(expected_res.statistic)
    assert res.p_value == float(expected_res.pvalue)
    # Confirm it actively differs from the 'wilcox' method's result
    assert res.p_value != float(wilcox_res.pvalue)


def test_paired_t_null_not_rejected(paired_equal):
    res = paired_t(paired_equal)
    assert res.p_value > 0.05


def test_paired_t_shift_detected(paired_shifted):
    res = paired_t(paired_shifted, alternative="less")
    assert res.p_value < 0.05


def test_paired_t_n1(paired_single):
    res = paired_t(paired_single)
    assert res.p_value is None


def test_paired_t_constant_diffs(paired_identical):
    res = paired_t(paired_identical)
    assert res.p_value is None or np.isnan(res.p_value)


def test_sign_all_positive_diffs():
    from malthusjax.stats.core import MetricVector, PairedSample

    ps = PairedSample(
        MetricVector("l", np.array([2.0, 3.0, 4.0])), MetricVector("r", np.array([1.0, 1.0, 1.0]))
    )
    res = sign_test(ps, alternative="greater")
    assert res.p_value < 0.2  # 1/8


def test_sign_all_ties(paired_identical):
    res = sign_test(paired_identical)
    assert res.statistic is None
    assert res.p_value is None


def test_sign_n1(paired_single):
    res = sign_test(paired_single)
    assert res.p_value == 1.0


def test_tost_equivalent_small_diff(paired_equal):
    res = tost(paired_equal, margin=1.0)
    assert isinstance(res, TOSTResult)
    assert res.equivalent is True


def test_tost_not_equivalent_large_diff(paired_shifted):
    res = tost(paired_shifted, margin=0.5)
    assert res.equivalent is False


def test_tost_margin_zero_raises(paired_equal):
    with pytest.raises(ValueError, match="margin must be > 0"):
        tost(paired_equal, margin=0.0)


def test_tost_n1(paired_single):
    res = tost(paired_single, margin=1.0)
    assert res.p_value_max is None
    assert res.equivalent is None


def test_tost_identical_arrays(paired_identical):
    res = tost(paired_identical, margin=1.0)
    assert res.equivalent is True
