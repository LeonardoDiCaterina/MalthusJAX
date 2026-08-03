import numpy as np

from malthusjax.stats.correction import fdr_bh, holm_bonferroni


def test_holm_basic():
    pvals = [0.001, 0.01, 0.04, 0.05, 0.1]
    adj = holm_bonferroni(pvals)
    assert adj[0] == 0.005
    assert adj[1] == 0.04
    assert np.isclose(adj[2], 0.12)
    assert np.isclose(adj[3], 0.12)
    assert np.isclose(adj[4], 0.12)


def test_holm_single_pvalue():
    assert holm_bonferroni([0.05]) == [0.05]


def test_holm_empty_list():
    assert holm_bonferroni([]) == []


def test_holm_clipped_to_one():
    pvals = [0.5, 0.6, 0.7]
    adj = holm_bonferroni(pvals)
    assert all(a <= 1.0 for a in adj)


def test_holm_with_nan():
    pvals = [0.01, np.nan, 0.05]
    adj = holm_bonferroni(pvals)
    assert adj[0] == 0.02
    assert np.isnan(adj[1])
    assert adj[2] == 0.05


def test_fdr_bh_basic():
    pvals = [0.001, 0.01, 0.04, 0.05, 0.1]
    adj = fdr_bh(pvals)
    assert adj[0] <= 0.005
    assert adj[1] <= 0.025
    assert np.isclose(adj[2], 0.0625)
    assert np.isclose(adj[3], 0.0625)
    assert np.isclose(adj[4], 0.1)


def test_correction_holm_vs_fdr():
    pvals = [0.01, 0.02, 0.03, 0.04, 0.05]
    holm = holm_bonferroni(pvals)
    fdr = fdr_bh(pvals)
    for h, f in zip(holm, fdr):
        assert h >= f
