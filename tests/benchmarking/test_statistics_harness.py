import numpy as np
import pytest

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult
from malthusjax.benchmarking.statistics import (
    EffectSizeResult,
    ExpectedDirection,
    HypothesisKind,
    MultipleTestingPolicy,
    PairedMetricDataset,
    Sidedness,
    StatisticalComparator,
    StatisticalComparisonResult,
    StatisticalComparisonSpec,
    StatisticalSpecError,
    StatisticalSuiteResult,
    TOSTResult,
    adjust_pvalues,
    apply_decision_rule,
    attach_adjusted_pvalues,
    compare_paired_arrays,
    compute_effect_sizes,
    compute_standard_tests,
    compute_tost_paired,
    infer_scipy_alternative,
    paired_dataset_from_artifacts,
    paired_dataset_from_comparison,
    paired_dataset_from_experiments,
    validate_spec,
)
from malthusjax.benchmarking.statistics import (
    TestResult as StatTestResult,
)


@pytest.fixture
def sample_dataset() -> PairedMetricDataset:
    seeds = [0, 1, 2, 3, 4]
    left = np.array([1.0, 0.9, 1.1, 0.95, 1.05], dtype=float)
    right = np.array([1.1, 1.0, 1.2, 1.05, 1.15], dtype=float)
    return PairedMetricDataset(
        label="toy_pair",
        left_name="malthusjax",
        right_name="evosax",
        seeds=seeds,
        left_values=left,
        right_values=right,
        metric_name="best_fitness",
        metric_source="fixture",
    )


@pytest.fixture
def raw_spec() -> StatisticalComparisonSpec:
    return StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.LOCATION_SHIFT,
        sidedness=Sidedness.ONE_SIDED,
        expected_direction=ExpectedDirection.LEFT_LT_RIGHT,
        min_paired_seeds=3,
        alpha=0.05,
        multiple_testing=MultipleTestingPolicy.NONE,
    )


@pytest.fixture
def eq_spec() -> StatisticalComparisonSpec:
    return StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.EQUIVALENCE,
        equivalence_margin=0.25,
        min_paired_seeds=3,
        alpha=0.05,
    )


# --- immediate scaffold sanity checks (should pass now) ---

def test_dataclass_shapes(raw_spec: StatisticalComparisonSpec, sample_dataset: PairedMetricDataset):
    assert raw_spec.metric_name == "best_fitness"
    assert sample_dataset.left_values.shape == sample_dataset.right_values.shape
    assert len(sample_dataset.seeds) == sample_dataset.left_values.size


def test_result_types_constructible():
    test = StatTestResult(name="wilcoxon", statistic=1.0, p_value=0.2, alternative="two-sided")
    tost = TOSTResult(
        margin=0.1,
        lower_bound=-0.1,
        upper_bound=0.1,
        t_stat_lower=1.0,
        t_stat_upper=2.0,
        p_value_lower=0.1,
        p_value_upper=0.2,
        p_value_max=0.2,
        equivalent=False,
    )
    effects = EffectSizeResult(cohen_dz=0.1, rank_biserial=0.2)
    result = StatisticalComparisonResult(
        label="x",
        hypothesis_text="h",
        n_paired=5,
        wins_left=3,
        wins_right=2,
        ties=0,
        left_mean=1.0,
        right_mean=1.1,
        mean_diff_left_minus_right=-0.1,
        median_diff_left_minus_right=-0.1,
        tests={"wilcoxon": test},
        tost=tost,
        effects=effects,
        alpha=0.05,
        decision_pass=True,
        decision_basis="wilcoxon",
    )
    suite = StatisticalSuiteResult(spec=StatisticalComparisonSpec(), results=[result])
    assert suite.results[0].label == "x"


# --- contract tests for future logic (xfail until implementation) ---

def test_validate_spec_rejects_missing_equivalence_margin():
    spec = StatisticalComparisonSpec(hypothesis_kind=HypothesisKind.EQUIVALENCE,
                                     equivalence_margin=None)
    with pytest.raises(StatisticalSpecError):
        validate_spec(spec)


def test_infer_scipy_alternative_mapping():
    assert infer_scipy_alternative(Sidedness.TWO_SIDED,
                                   ExpectedDirection.LEFT_LT_RIGHT) == "two-sided"
    assert infer_scipy_alternative(Sidedness.ONE_SIDED,
                                   ExpectedDirection.LEFT_LT_RIGHT) == "less"
    assert infer_scipy_alternative(Sidedness.ONE_SIDED,
                                   ExpectedDirection.LEFT_GT_RIGHT) == "greater"


def test_compute_tost_paired_contract(sample_dataset: PairedMetricDataset):
    out = compute_tost_paired(sample_dataset.left_values,
                              sample_dataset.right_values,
                              margin=0.2, alpha=0.05)
    assert isinstance(out, TOSTResult)
    assert out.margin == 0.2
    assert out.lower_bound == -0.2
    assert out.upper_bound == 0.2
    assert out.p_value_max is not None


def test_compute_standard_tests_contract(sample_dataset: PairedMetricDataset):
    tests = compute_standard_tests(
        sample_dataset.left_values,
        sample_dataset.right_values,
        alternative="less",
        include_tests=("wilcoxon", "paired_t", "sign"),
    )
    assert set(tests.keys()) == {"wilcoxon", "paired_t", "sign"}
    assert all(isinstance(v, StatTestResult) for v in tests.values())


def test_compute_effect_sizes_contract(sample_dataset: PairedMetricDataset):
    effects = compute_effect_sizes(sample_dataset.left_values, sample_dataset.right_values)
    assert isinstance(effects, EffectSizeResult)


def test_apply_decision_rule_prefers_tost(eq_spec: StatisticalComparisonSpec):
    tests = {
        "wilcoxon": StatTestResult(name="wilcoxon",
                                   statistic=1.0,
                                   p_value=0.001,
                                   alternative="two-sided"),
        "paired_t": StatTestResult(name="paired_t",
                                   statistic=1.0,
                                   p_value=0.001,
                                   alternative="two-sided"),
    }
    tost = TOSTResult(
        margin=0.2,
        lower_bound=-0.2,
        upper_bound=0.2,
        t_stat_lower=1.0,
        t_stat_upper=1.0,
        p_value_lower=0.2,
        p_value_upper=0.2,
        p_value_max=0.2,
        equivalent=False,
    )
    passed, basis, error = apply_decision_rule(spec=eq_spec, tests=tests, tost=tost)
    assert passed is False
    assert basis == "tost"
    assert error is None


def test_compare_paired_arrays_contract(sample_dataset: PairedMetricDataset,
                                        raw_spec: StatisticalComparisonSpec):
    result = compare_paired_arrays(
        label=sample_dataset.label,
        left_name=sample_dataset.left_name,
        right_name=sample_dataset.right_name,
        left=sample_dataset.left_values,
        right=sample_dataset.right_values,
        spec=raw_spec,
        metadata={"source": "fixture"},
    )
    assert isinstance(result, StatisticalComparisonResult)
    assert result.n_paired == len(sample_dataset.seeds)
    assert "source" in result.metadata


def test_adjust_pvalues_none_is_identity():
    pvals = [0.01, 0.1, 0.5]
    adjusted = adjust_pvalues(pvals, MultipleTestingPolicy.NONE)
    assert adjusted == pvals


def test_comparator_compare_suite_contract(sample_dataset: PairedMetricDataset,
                                           raw_spec: StatisticalComparisonSpec):
    comp = StatisticalComparator()
    suite = comp.compare_suite([sample_dataset], raw_spec)
    assert isinstance(suite, StatisticalSuiteResult)
    assert len(suite.results) == 1


def test_adjust_pvalues_holm_and_fdr_bh():
    pvals = [0.01, 0.04, 0.03]

    holm = adjust_pvalues(pvals, MultipleTestingPolicy.HOLM)
    fdr = adjust_pvalues(pvals, MultipleTestingPolicy.FDR_BH)

    assert len(holm) == len(pvals)
    assert len(fdr) == len(pvals)
    assert all(0.0 <= p <= 1.0 for p in holm)
    assert all(0.0 <= p <= 1.0 for p in fdr)


def test_suite_to_dict_and_markdown(sample_dataset: PairedMetricDataset,
                                    raw_spec: StatisticalComparisonSpec):
    comp = StatisticalComparator()
    suite = comp.compare_suite([sample_dataset], raw_spec)

    payload = suite.to_dict()
    assert "spec" in payload
    assert "results" in payload
    assert payload["results"][0]["label"] == sample_dataset.label

    md = suite.to_markdown()
    assert "# Statistical Suite Summary" in md
    assert sample_dataset.label in md
    assert "Cohen dz" in md


def test_attach_adjusted_pvalues_contract(sample_dataset: PairedMetricDataset,
                                          raw_spec: StatisticalComparisonSpec):
    result = compare_paired_arrays(
        label=sample_dataset.label,
        left_name=sample_dataset.left_name,
        right_name=sample_dataset.right_name,
        left=sample_dataset.left_values,
        right=sample_dataset.right_values,
        spec=raw_spec,
    )
    mapping = attach_adjusted_pvalues([result], [0.123], key="primary")
    assert mapping[result.label]["primary"] == pytest.approx(0.123)


@pytest.fixture
def tiny_experiments_pair() -> tuple[ExperimentResult, ExperimentResult]:
    left_runs = [
        RunResult(
            seed=0,
            status="success",
            metrics={"best_fitness": 1.0, "gap_to_optimum": 1.0, "initial_fitness": 2.0},
            history=[{"best_fitness": 2.0}, {"best_fitness": 1.0}],
        ),
        RunResult(
            seed=1,
            status="success",
            metrics={"best_fitness": 0.8, "gap_to_optimum": 0.8, "initial_fitness": 1.8},
            history=[{"best_fitness": 1.8}, {"best_fitness": 0.8}],
        ),
    ]
    right_runs = [
        RunResult(
            seed=0,
            status="success",
            metrics={"best_fitness": 1.2, "gap_to_optimum": 1.2, "initial_fitness": 2.2},
            history=[{"best_fitness": 2.2}, {"best_fitness": 1.2}],
        ),
        RunResult(
            seed=1,
            status="success",
            metrics={"best_fitness": 0.9, "gap_to_optimum": 0.9, "initial_fitness": 1.9},
            history=[{"best_fitness": 1.9}, {"best_fitness": 0.9}],
        ),
    ]
    return (
        ExperimentResult(name="left_exp", runs=left_runs),
        ExperimentResult(name="right_exp", runs=right_runs),
    )


def test_paired_dataset_from_experiments_location(tiny_experiments_pair):
    left, right = tiny_experiments_pair
    spec = StatisticalComparisonSpec(metric_name="best_fitness", min_paired_seeds=2)

    ds = paired_dataset_from_experiments(left, right, "left", "right", spec)
    assert ds.seeds == [0, 1]
    assert np.allclose(ds.left_values, [1.0, 0.8])
    assert np.allclose(ds.right_values, [1.2, 0.9])
    assert ds.metadata["left_start_mean"] == pytest.approx(1.9)
    assert ds.metadata["right_start_mean"] == pytest.approx(2.05)
    assert ds.metadata["left_end_mean"] == pytest.approx(0.9)
    assert ds.metadata["right_end_mean"] == pytest.approx(1.05)


def test_paired_dataset_from_experiments_closer_uses_gap(tiny_experiments_pair):
    left, right = tiny_experiments_pair
    spec = StatisticalComparisonSpec(
        metric_name="best_fitness",
        hypothesis_kind=HypothesisKind.CLOSER_TO_OPTIMUM,
        min_paired_seeds=2,
    )

    ds = paired_dataset_from_experiments(left, right, "left", "right", spec)
    assert ds.metric_source == "gap_to_optimum_or_derived_distance"
    assert np.allclose(ds.left_values, [1.0, 0.8])
    assert np.allclose(ds.right_values, [1.2, 0.9])


def test_paired_dataset_from_comparison(tiny_experiments_pair):
    left, right = tiny_experiments_pair
    comp = ComparisonResult(pipelines={"A": left, "B": right})
    spec = StatisticalComparisonSpec(metric_name="best_fitness", min_paired_seeds=2)

    ds = paired_dataset_from_comparison(comp, "A", "B", spec)
    assert ds.left_name == "A"
    assert ds.right_name == "B"


def test_paired_dataset_from_artifacts(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    left_csv = (
        "seed,best_fitness,generation\n"
        "0,1.0,1\n"
        "0,0.8,2\n"
        "1,0.9,1\n"
        "1,0.7,2\n"
    )
    right_csv = (
        "seed,best_fitness,generation\n"
        "0,1.1,1\n"
        "0,1.0,2\n"
        "1,1.0,1\n"
        "1,0.95,2\n"
    )
    (left_dir / "histories_combined.csv").write_text(left_csv)
    (right_dir / "histories_combined.csv").write_text(right_csv)

    spec = StatisticalComparisonSpec(metric_name="best_fitness", min_paired_seeds=2)
    ds = paired_dataset_from_artifacts(left_dir, right_dir, "L", "R", spec)

    assert ds.seeds == [0, 1]
    assert np.allclose(ds.left_values, [0.8, 0.7])
    assert np.allclose(ds.right_values, [1.0, 0.95])
