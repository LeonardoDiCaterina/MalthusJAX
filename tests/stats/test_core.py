import numpy as np
import pytest

from malthusjax.stats.core import MetricVector, PairedSample, TestResult


def test_metric_vector_construction():
    mv = MetricVector("test", np.array([1.0, 2.0]))
    assert mv.name == "test"
    assert mv.n == 2
    assert np.array_equal(mv.values, [1.0, 2.0])


def test_metric_vector_is_frozen():
    mv = MetricVector("test", np.array([1.0, 2.0]))
    with pytest.raises(Exception):
        mv.name = "changed"


def test_metric_vector_empty():
    mv = MetricVector("empty", np.array([]))
    assert mv.n == 0


def test_paired_sample_construction():
    left = MetricVector("left", np.array([1.0, 2.0]))
    right = MetricVector("right", np.array([1.5, 2.5]))
    ps = PairedSample(left, right, "label")
    assert ps.n == 2
    assert ps.label == "label"
    assert np.array_equal(ps.diffs, [-0.5, -0.5])


def test_paired_sample_size_mismatch_raises():
    left = MetricVector("left", np.array([1.0, 2.0]))
    right = MetricVector("right", np.array([1.0]))
    with pytest.raises(ValueError, match="mismatch"):
        PairedSample(left, right)


def test_paired_sample_n1():
    left = MetricVector("left", np.array([1.0]))
    right = MetricVector("right", np.array([1.0]))
    ps = PairedSample(left, right)
    assert ps.n == 1


def test_test_result_passes_alpha():
    res = TestResult("test", statistic=1.0, p_value=0.04, alternative="two-sided")
    assert res.passes(0.05) is False
    assert res.passes(0.01) is True


def test_test_result_passes_none_pvalue():
    res = TestResult("test", statistic=None, p_value=None, alternative="two-sided")
    assert res.passes(0.05) is None
