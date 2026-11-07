"""Statistical harness calibration and pre-declared TOST margin validation.

Verifies:
1. A/A null test: False Positive Rate (FPR) <= 0.08 at alpha=0.05 over 200 synthetic paired runs.
2. A/B power test: Statistical power >= 0.85 when detecting a 5% shift (N=30).
3. TOST equivalence margin decision rule: 0% shift is equivalent, 5% shift fails equivalence,
   and under-powered samples (N=3) fail equivalence rather than silently passing via non-significance.
"""

from __future__ import annotations

import numpy as np

from malthusjax.stats.core import MetricVector, PairedSample
from malthusjax.stats.correction import holm_bonferroni
from malthusjax.stats.tests import tost, wilcoxon


def test_aa_null_false_positive_rate_calibrated():
    """Verify empirical Type I error rate (FPR) does not exceed nominal alpha (+ tolerance)."""
    rng = np.random.default_rng(2024)
    p_vals = []
    n_trials = 200
    sample_size = 30

    for _ in range(n_trials):
        a = rng.normal(10.0, 1.0, size=sample_size)
        b = rng.normal(10.0, 1.0, size=sample_size)
        sample = PairedSample(
            left=MetricVector("left_a", values=a),
            right=MetricVector("right_a", values=b),
        )
        res = wilcoxon(sample)
        p_vals.append(res.p_value)

    raw_fpr = float(np.mean(np.array(p_vals) < 0.05))
    adj_p = holm_bonferroni(p_vals)
    adj_fpr = float(np.mean(np.array(adj_p) < 0.05))

    # Nominal alpha = 0.05, allow empirical Monte Carlo tolerance <= 0.08
    assert raw_fpr <= 0.08, f"Unadjusted FPR {raw_fpr:.3f} exceeded 0.08 threshold"
    assert adj_fpr <= 0.05, f"FWER-adjusted FPR {adj_fpr:.3f} exceeded 0.05 threshold"


def test_ab_power_detects_five_percent_shift():
    """Verify detection power >= 0.85 when detecting a 5% shift with N=30."""
    rng = np.random.default_rng(42)
    p_vals = []
    n_trials = 100
    sample_size = 30

    for _ in range(n_trials):
        # Baseline mean = 10.0, shift delta = +0.5 (5%), noise sigma = 0.3
        a = rng.normal(10.0, 0.3, size=sample_size)
        b = rng.normal(10.5, 0.3, size=sample_size)
        sample = PairedSample(
            left=MetricVector("baseline", values=a),
            right=MetricVector("shifted", values=b),
        )
        res = wilcoxon(sample)
        p_vals.append(res.p_value)

    power = float(np.mean(np.array(p_vals) < 0.05))
    assert power >= 0.85, f"Statistical power {power:.3f} fell below 0.85 threshold"


def test_tost_equivalence_margin_decision_rule():
    """Verify pre-declared TOST margin Delta = 0.02 correctly validates equivalence.

    1. Identical distributions pass equivalence (p_TOST < 0.05).
    2. A 5% shift fails equivalence (p_TOST >= 0.05).
    3. Underpowered small-sample / high-variance tests fail equivalence (absence of evidence != evidence of absence).
    """
    rng = np.random.default_rng(999)
    sample_size = 30
    baseline_mean = 10.0
    relative_margin = 0.02
    absolute_margin = relative_margin * baseline_mean  # 0.2

    # 1. Zero-shift test
    base = np.full(sample_size, baseline_mean) + rng.normal(0, 0.02, size=sample_size)
    target_zero = base + rng.normal(0, 0.02, size=sample_size)
    s_zero = PairedSample(
        left=MetricVector("base", values=base),
        right=MetricVector("zero_shift", values=target_zero),
    )
    tost_zero = tost(s_zero, margin=absolute_margin, alpha=0.05)
    assert tost_zero.equivalent is True, (
        f"Zero shift failed TOST equivalence: p_max={tost_zero.p_value_max}"
    )
    assert tost_zero.p_value_max < 0.05

    # 2. 5% shift test (+0.5 exceeds 0.2 margin)
    target_5pct = base + 0.5 + rng.normal(0, 0.02, size=sample_size)
    s_5pct = PairedSample(
        left=MetricVector("base", values=base),
        right=MetricVector("shifted_5pct", values=target_5pct),
    )
    tost_5pct = tost(s_5pct, margin=absolute_margin, alpha=0.05)
    assert tost_5pct.equivalent is False, "5% shift erroneously declared equivalent"
    assert tost_5pct.p_value_max >= 0.05

    # 3. Underpowered small sample (N=3, wide variance)
    base_under = np.array([10.0, 10.0, 10.0])
    target_under = np.array([9.5, 10.5, 10.1])
    s_under = PairedSample(
        left=MetricVector("base_under", values=base_under),
        right=MetricVector("target_under", values=target_under),
    )
    tost_under = tost(s_under, margin=absolute_margin, alpha=0.05)
    assert tost_under.equivalent is False, "Under-powered test must not pass equivalence"
