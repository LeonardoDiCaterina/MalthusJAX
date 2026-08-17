from malthusjax.stats.core import (
    EffectSizeResult,
    StatisticalComparisonResult,
    StatisticalComparisonSpec,
    StatisticalSuiteResult,
    TestResult,
    TOSTResult,
)


def test_suite_to_dict_roundtrip():
    spec = StatisticalComparisonSpec(metric_name="score")

    test_res = TestResult(name="wilcoxon", statistic=10.0, p_value=0.04, alternative="two-sided")
    tost_res = TOSTResult(
        margin=1.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        t_stat_lower=2.0,
        t_stat_upper=-2.0,
        p_value_lower=0.01,
        p_value_upper=0.01,
        p_value_max=0.01,
        equivalent=True,
    )
    eff = EffectSizeResult(cohen_dz=0.8, rank_biserial=0.9)

    comp = StatisticalComparisonResult(
        label="test_vs_ref",
        hypothesis_text="location shift",
        n_paired=30,
        wins_left=20,
        wins_right=10,
        ties=0,
        left_mean=10.0,
        right_mean=8.0,
        mean_diff_left_minus_right=2.0,
        median_diff_left_minus_right=1.8,
        tests={"wilcoxon": test_res},
        tost=tost_res,
        effects=eff,
        alpha=0.05,
        decision_pass=True,
        decision_reliable=True,
        decision_basis="wilcoxon_two_sided",
        metadata={"extra": "data"},
    )

    suite = StatisticalSuiteResult(spec=spec, results=[comp])
    suite.adjusted_p_values = {"test_vs_ref": {"primary": 0.045}}

    d = suite.to_dict()
    assert d["spec"]["metric_name"] == "score"
    assert d["results"][0]["label"] == "test_vs_ref"
    assert d["results"][0]["tests"]["wilcoxon"]["p_value"] == 0.04
    assert d["results"][0]["effects"]["cohen_dz"] == 0.8
    assert d["adjusted_p_values"]["test_vs_ref"]["primary"] == 0.045


def test_none_values_in_results():
    spec = StatisticalComparisonSpec(metric_name="score")
    comp = StatisticalComparisonResult(
        label="test",
        hypothesis_text="shift",
        n_paired=0,
        wins_left=0,
        wins_right=0,
        ties=0,
        left_mean=0.0,
        right_mean=0.0,
        mean_diff_left_minus_right=0.0,
        median_diff_left_minus_right=0.0,
        tests={"t": TestResult("t", None, None, "two-sided")},
        tost=None,
        effects=EffectSizeResult(None, None),
        alpha=0.05,
        decision_pass=None,
        decision_reliable=None,
        decision_basis="none",
        metadata={},
    )
    suite = StatisticalSuiteResult(spec=spec, results=[comp])
    d = suite.to_dict()
    assert d["results"][0]["tests"]["t"]["p_value"] is None
    assert d["results"][0]["tost"] is None
