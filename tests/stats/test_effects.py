import numpy as np

from malthusjax.stats.effects import cohens_dz, glass_delta, rank_biserial


def test_cohens_dz_positive_shift(paired_shifted):
    dz = cohens_dz(paired_shifted)
    assert dz < 0.0  # left - right where right is shifted +2


def test_cohens_dz_no_shift(paired_equal):
    dz = cohens_dz(paired_equal)
    assert abs(dz) < 0.5


def test_cohens_dz_zero_variance(paired_identical):
    dz = cohens_dz(paired_identical)
    assert dz == 0.0


def test_cohens_dz_n1(paired_single):
    dz = cohens_dz(paired_single)
    assert np.isnan(dz)


def test_rank_biserial_all_positive():
    from malthusjax.stats.core import MetricVector, PairedSample

    ps = PairedSample(
        MetricVector("l", np.array([2.0, 3.0, 4.0])), MetricVector("r", np.array([1.0, 1.0, 1.0]))
    )
    rb = rank_biserial(ps)
    assert rb == 1.0


def test_rank_biserial_all_negative():
    from malthusjax.stats.core import MetricVector, PairedSample

    ps = PairedSample(
        MetricVector("l", np.array([1.0, 1.0, 1.0])), MetricVector("r", np.array([2.0, 3.0, 4.0]))
    )
    rb = rank_biserial(ps)
    assert rb == -1.0


def test_rank_biserial_all_ties(paired_identical):
    rb = rank_biserial(paired_identical)
    assert rb is None


def test_glass_delta_zero_variance():
    from malthusjax.stats.core import MetricVector, PairedSample

    ps = PairedSample(
        MetricVector("l", np.array([2.0, 3.0, 4.0])),
        MetricVector("r", np.array([1.0, 1.0, 1.0])),  # std is 0
    )
    gd = glass_delta(ps)
    assert gd == 0.0
