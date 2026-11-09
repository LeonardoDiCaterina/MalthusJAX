"""P4 validity, hypothesis meaning, and family-corrected export regressions."""

import json

import numpy as np
import pytest

from malthusjax.stats.comparator import StatisticalComparator, compare_paired_arrays
from malthusjax.stats.core import (
    HypothesisKind,
    MetricVector,
    MultipleTestingPolicy,
    PairedSample,
    Sidedness,
    StatisticalComparisonSpec,
    StatisticalSuiteResult,
    TestResult,
    TOSTResult,
)
from malthusjax.stats.paper import analyze_archive_records, inference_text
from malthusjax.stats.tests import paired_t, tost, wilcoxon


def compare(left, right, **kwargs):
    return compare_paired_arrays(
        label="fixed",
        left_name="native",
        right_name="reference",
        left=np.asarray(left),
        right=np.asarray(right),
        spec=StatisticalComparisonSpec(min_paired_seeds=2, **kwargs),
    )


@pytest.mark.parametrize("bad", [[np.nan, 1], [np.inf, 1], [[1, 2]], 1.0])
def test_invalid_samples_are_not_inferential_evidence(bad):
    with pytest.raises(ValueError):
        MetricVector("invalid", bad)


def test_one_seed_cannot_establish_equivalence():
    sample = PairedSample(MetricVector("l", [1.0]), MetricVector("r", [1.0]))
    result = tost(sample, 1.0)
    assert result.status == "insufficient"
    assert result.p_value_max is None and result.equivalent is None
    result = compare(
        [1.0], [1.0], hypothesis_kind=HypothesisKind.EQUIVALENCE, equivalence_margin=1.0
    )
    assert result.status == "insufficient" and result.decision_pass is None


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_constant_differences_make_parametric_inference_unavailable(delta):
    sample = PairedSample(
        MetricVector("l", np.arange(10.0) + delta), MetricVector("r", np.arange(10.0))
    )
    assert tost(sample, 2.0).p_value_max is None
    assert paired_t(sample).p_value is None
    if delta == 0:
        assert wilcoxon(sample).p_value is None


def test_directional_significance_means_improvement_not_failure():
    result = compare(
        np.arange(12.0) * 0.01,
        np.arange(12.0) * 0.01 + np.linspace(1.0, 2.0, 12),
        sidedness=Sidedness.ONE_SIDED,
    )
    assert result.decision_pass is True
    assert result.conclusion == "improvement_detected"
    assert result.raw_conclusion == result.conclusion


def test_non_rejection_does_not_mean_equivalence():
    result = compare(np.array([-0.2, 0.1, 0.2, -0.1]), np.zeros(4))
    assert result.conclusion == "difference_not_detected"
    assert result.decision_pass is False


def test_two_raw_equivalence_p03_become_corrected_p06(monkeypatch):
    import malthusjax.stats.comparator as module

    monkeypatch.setattr(
        module,
        "compute_tost_paired",
        lambda *args, **kwargs: TOSTResult(1.0, -1.0, 1.0, 2.0, -2.0, 0.03, 0.03, 0.03, True),
    )
    results = [
        compare(
            np.linspace(-0.1, 0.1, 12),
            np.zeros(12),
            hypothesis_kind=HypothesisKind.EQUIVALENCE,
            equivalence_margin=1.0,
        )
        for _ in range(2)
    ]
    for index, result in enumerate(results):
        result.label = f"h{index}"
    spec = StatisticalComparisonSpec(
        hypothesis_kind=HypothesisKind.EQUIVALENCE, equivalence_margin=1.0, min_paired_seeds=2
    )
    suite = StatisticalComparator().adjust_suite_pvalues(StatisticalSuiteResult(spec, results))
    assert all(r.raw_decision_pass is True for r in results)
    assert all(
        r.decision_pass is False and r.conclusion == "equivalence_not_established" for r in results
    )
    assert all(r.adjusted_p_value == 0.06 for r in results)
    payload = suite.to_dict()
    assert all(
        r["tost"]["equivalent"] is False and r["tost"]["raw_equivalent"] is True
        for r in payload["results"]
    )
    text = suite.to_markdown()
    assert "0.06" in text and "equivalence_not_established" in text
    assert "| pass |" not in text
    json.dumps(payload, allow_nan=False)


def test_unavailable_hypotheses_do_not_shrink_declared_family():
    from malthusjax.stats.correction import adjust_pvalues

    adjusted = adjust_pvalues([0.03, np.nan], MultipleTestingPolicy.HOLM)
    assert adjusted[0] == 0.06
    assert np.isnan(adjusted[1])


def test_unknown_test_and_invalid_p_values_are_rejected():
    assert TestResult("bad", 1.0, np.nan, "two-sided").p_value is None
    with pytest.raises(ValueError):
        compare([1, 2], [2, 3], include_tests=("unknown",))


def test_paper_without_predeclared_policy_is_descriptive():
    result = analyze_archive_records([], [], None)
    assert result["status"] == "descriptive_only"
    assert "Descriptive only" in inference_text(result, "best_fitness")


def test_paper_policy_declares_family_estimand_margin_and_sample_minimum():
    rows = [{"seed": seed, "best_fitness": [seed * 0.01]} for seed in range(12)]
    other = [{"seed": seed, "best_fitness": [0.0]} for seed in range(12)]
    policy = {
        "family": "fixed family",
        "sample_size_rationale": "Synthetic regression only; no publication power claim",
        "alpha": 0.05,
        "minimum_pairs": 10,
        "multiplicity": "holm",
        "hypotheses": [
            {
                "label": "fitness_equivalence",
                "metric": "best_fitness",
                "kind": "equivalence",
                "primary_test": "paired_t",
                "metric_orientation": "lower_is_better",
                "estimand": "mean_paired_difference",
                "margin": 1.0,
                "sidedness": "two_sided",
                "expected_direction": "left_lt_right",
            }
        ],
    }
    analysis = analyze_archive_records(rows, other, policy)
    assert analysis["status"] == "valid"
    assert analysis["results"][0]["conclusion"] == "equivalent"
    assert "family corrected" in inference_text(analysis, "best_fitness")
    policy["hypotheses"][0].pop("margin")
    with pytest.raises(ValueError):
        analyze_archive_records(rows, other, policy)


def test_mutated_nonfinite_sample_cannot_silently_drop_pairs():
    from malthusjax.stats.tests import sign_test

    sample = PairedSample(MetricVector("l", [1.0, 2.0]), MetricVector("r", [0.0, 0.0]))
    sample.left.values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sign_test(sample)


def test_higher_is_better_metrics_have_correct_win_counts():
    result = compare([3.0, 4.0, 5.0], [1.0, 2.0, 3.0], metric_orientation="higher_is_better")
    assert result.wins_left == 3 and result.wins_right == 0


def test_rank_biserial_uses_ranks_instead_of_sign_counts():
    from malthusjax.stats.effects import rank_biserial

    sample = PairedSample(MetricVector("l", [10.0, -1.0, -2.0]), MetricVector("r", [0.0, 0.0, 0.0]))
    assert rank_biserial(sample) == 0.0


def test_directional_worsening_is_not_labeled_improvement():
    from malthusjax.stats.core import ExpectedDirection

    result = compare(
        np.arange(12.0) + np.linspace(1.0, 2.0, 12),
        np.arange(12.0),
        sidedness=Sidedness.ONE_SIDED,
        expected_direction=ExpectedDirection.LEFT_GT_RIGHT,
    )
    assert result.decision_pass is True
    assert result.conclusion == "deterioration_detected"
