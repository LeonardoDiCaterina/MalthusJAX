import pytest
import numpy as np
from malthusjax.stats.comparator import compare_paired_arrays

def test_comparator_identical_arrays():
    """Test comparator handling of identical arrays."""
    
    # Identical arrays have zero variance and zero difference
    arr1 = np.ones(50)
    arr2 = np.ones(50)
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec()
    
    # The Mann-Whitney U test or t-test might raise warnings or handle 
    # zero variance with fallbacks. We just ensure it doesn't crash.
    metrics = compare_paired_arrays(
        label="test",
        left_name="left",
        right_name="right",
        left=arr1,
        right=arr2, 
        spec=spec
    )
    
    assert metrics is not None

def test_comparator_mismatched_lengths():
    """Test comparator handling arrays with different numbers of seeds."""
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec()
    
    arr1 = np.random.randn(5, 10) # 5 seeds
    arr2 = np.random.randn(4, 10) # 4 seeds
    
    with pytest.raises(ValueError):
        compare_paired_arrays(
            label="test",
            left_name="left",
            right_name="right",
            left=arr1,
            right=arr2, 
            spec=spec
        )

def test_comparator_other_tests():
    """Test other statistical methods."""
    arr1 = np.random.randn(50)
    arr2 = np.random.randn(50) + 1.0
    
    from malthusjax.stats.core import StatisticalComparisonSpec
    spec = StatisticalComparisonSpec(include_tests=("wilcoxon", "paired_t", "tost", "sign"))
    metrics = compare_paired_arrays(
        label="test",
        left_name="left",
        right_name="right",
        left=arr1,
        right=arr2,
        spec=spec
    )
    assert metrics is not None

def test_comparator_equivalence_and_metadata():
    """Test equivalence hypothesis, start/end means, and include_value_lists."""
    arr1 = np.random.randn(20)
    arr2 = np.random.randn(20) + 0.1
    
    from malthusjax.stats.core import (
        StatisticalComparisonSpec,
        HypothesisKind,
        MultipleTestingPolicy,
        PairedMetricDataset,
    )
    from malthusjax.stats.comparator import StatisticalComparator
    
    spec = StatisticalComparisonSpec(
        hypothesis_kind=HypothesisKind.EQUIVALENCE,
        equivalence_margin=0.5,
        optimum_value=0.0,
        include_value_lists=True,
        multiple_testing=MultipleTestingPolicy.HOLM,
        min_paired_seeds=5,
    )
    
    ds = PairedMetricDataset(
        label="ds1",
        left_name="L",
        right_name="R",
        seeds=list(range(20)),
        left_values=arr1,
        right_values=arr2,
        metric_name="fitness",
        metric_source="eval",
        metadata={"left_start_mean": 1.0, "right_start_mean": 1.5},
    )
    
    comparator = StatisticalComparator()
    suite_res = comparator.compare_suite([ds], spec)
    assert suite_res is not None
    assert len(suite_res.results) == 1
    assert "include_value_lists" in suite_res.results[0].metadata

def test_comparator_small_sample_error():
    """Test min_paired_seeds error path."""
    arr1 = np.random.randn(3)
    arr2 = np.random.randn(3)
    
    from malthusjax.stats.core import StatisticalComparisonSpec, StatisticalSpecError
    spec = StatisticalComparisonSpec(min_paired_seeds=10)
    
    with pytest.raises(StatisticalSpecError):
        compare_paired_arrays(
            label="test",
            left_name="left",
            right_name="right",
            left=arr1,
            right=arr2,
            spec=spec,
        )

def test_comparator_internal_helpers():
    from malthusjax.stats.comparator import (
        _describe_values,
        _paired_timing_stats,
        _build_timing_summary,
        _mean_or_none,
    )
    
    # Empty & populated describe
    assert _describe_values([])["mean"] is None
    d = _describe_values([1.0, 2.0, 3.0])
    assert d["mean"] == 2.0
    assert _describe_values([1.0])["std"] == 0.0

    # Timing stats empty & mismatched
    ts_empty = _paired_timing_stats([], [])
    assert ts_empty["paired_diff_mean_left_minus_right"] is None
    
    ts_pop = _paired_timing_stats([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
    assert ts_pop["paired_diff_mean_left_minus_right"] == 0.5

    # Timing summary
    summary = _build_timing_summary(
        left_total=[1.0, 2.0],
        right_total=[0.5, 1.0],
        left_components={"c1": [0.5, 1.0]},
        right_components={"c1": [0.2, 0.5]},
    )
    assert "duration_seconds" in summary
    assert "c1" in summary["components"]

    # Mean or none
    assert _mean_or_none([None, float("nan")]) is None
    assert _mean_or_none([1.0, None, 3.0]) == 2.0

def test_paired_dataset_from_experiments():
    from malthusjax.benchmarking.results import ExperimentResult, RunResult
    from malthusjax.stats.comparator import paired_dataset_from_experiments
    from malthusjax.stats.core import StatisticalComparisonSpec, HypothesisKind
    
    runs_left = [
        RunResult(seed=s, status="success", metrics={"best_fitness": float(s), "initial_fitness": 10.0}, history=[{"best_fitness": 10.0}], duration_seconds=1.0)
        for s in range(5)
    ]
    runs_right = [
        RunResult(seed=s, status="success", metrics={"best_fitness": float(s + 1), "initial_fitness": 12.0}, history=[{"best_fitness": 12.0}], duration_seconds=0.8)
        for s in range(5)
    ]
    
    exp_left = ExperimentResult(name="left_exp", runs=runs_left)
    exp_right = ExperimentResult(name="right_exp", runs=runs_right)
    
    spec = StatisticalComparisonSpec(min_paired_seeds=3, include_value_lists=True, include_timing_stats=True)
    ds = paired_dataset_from_experiments(exp_left, exp_right, "left", "right", spec)
    assert ds is not None
    assert len(ds.left_values) == 5

    # Test CLOSER_TO_OPTIMUM path
    spec_opt = StatisticalComparisonSpec(min_paired_seeds=3, hypothesis_kind=HypothesisKind.CLOSER_TO_OPTIMUM, optimum_value=0.0)
    ds_opt = paired_dataset_from_experiments(exp_left, exp_right, "left", "right", spec_opt)
    assert ds_opt is not None

def test_paired_dataset_from_artifacts(tmp_path):
    import json
    from malthusjax.stats.comparator import paired_dataset_from_artifacts
    from malthusjax.stats.core import StatisticalComparisonSpec
    
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    
    csv_header = "seed,generation,best_fitness\n"
    csv_rows_left = csv_header + "\n".join([f"{s},1,{float(s)}" for s in range(5)])
    csv_rows_right = csv_header + "\n".join([f"{s},1,{float(s+1)}" for s in range(5)])
    
    (left_dir / "histories_combined.csv").write_text(csv_rows_left)
    (right_dir / "histories_combined.csv").write_text(csv_rows_right)
    
    summary_left = {"runs": [{"seed": s, "metrics": {"gap_to_optimum": 0.1}, "timings": {"step": 0.05}} for s in range(5)]}
    summary_right = {"runs": [{"seed": s, "metrics": {"gap_to_optimum": 0.2}, "timings": {"step": 0.05}} for s in range(5)]}
    
    (left_dir / "summary.json").write_text(json.dumps(summary_left))
    (right_dir / "summary.json").write_text(json.dumps(summary_right))
    
    spec = StatisticalComparisonSpec(min_paired_seeds=3, include_timing_stats=True)
    ds = paired_dataset_from_artifacts(left_dir, right_dir, "L", "R", spec)
    assert ds is not None
    assert len(ds.left_values) == 5

def test_comparator_extra_branches():
    from malthusjax.benchmarking.results import RunResult, ComparisonResult, ExperimentResult
    from malthusjax.stats.comparator import (
        _extract_metric_from_run,
        paired_dataset_from_comparison,
        StatisticalComparator,
    )
    from malthusjax.stats.core import (
        StatisticalComparisonSpec,
        StatisticalSpecError,
        MultipleTestingPolicy,
        PairedMetricDataset,
    )
    
    # 1. Metric not found in run
    run = RunResult(seed=0, status="success", metrics={"other": 1.0})
    with pytest.raises(StatisticalSpecError):
        _extract_metric_from_run(run, "missing_metric")

    # 2. paired_dataset_from_comparison
    exp_l = ExperimentResult(name="p1", runs=[RunResult(seed=0, status="success", metrics={"best_fitness": 1.0})])
    exp_r = ExperimentResult(name="p2", runs=[RunResult(seed=0, status="success", metrics={"best_fitness": 2.0})])
    comp_res = ComparisonResult(pipelines={"p1": exp_l, "p2": exp_r})
    
    spec = StatisticalComparisonSpec(min_paired_seeds=1)
    ds = paired_dataset_from_comparison(comp_res, "p1", "p2", spec)
    assert ds is not None

    # Invalid pipeline keys
    with pytest.raises(KeyError):
        paired_dataset_from_comparison(comp_res, "unknown", "p2", spec)

    # 3. adjust_suite_pvalues with FDR_BH
    spec_bonf = StatisticalComparisonSpec(min_paired_seeds=1, multiple_testing=MultipleTestingPolicy.FDR_BH)
    comparator = StatisticalComparator()
    ds1 = PairedMetricDataset("ds1", "p1", "p2", [0], np.array([1.0]), np.array([2.0]), "fitness", "source")
    suite = comparator.compare_suite([ds1], spec_bonf)
    assert suite is not None

def test_comparator_decision_error():
    from malthusjax.stats.comparator import compare_paired_arrays
    from malthusjax.stats.core import StatisticalComparisonSpec
    
    spec = StatisticalComparisonSpec(min_paired_seeds=1, include_tests=("unknown_test",))
    res = compare_paired_arrays(
        label="err_test",
        left_name="L",
        right_name="R",
        left=np.array([1.0]),
        right=np.array([2.0]),
        spec=spec,
    )
    assert "decision_error" in res.metadata

def test_comparator_shapiro_normality():
    from malthusjax.stats.comparator import compare_paired_arrays
    from malthusjax.stats.core import StatisticalComparisonSpec

    # Create a non-normal distribution of differences
    # E.g., one huge outlier
    left = np.zeros(20)
    right = np.zeros(20)
    left[0] = 100.0  # Outlier to break normality
    
    spec_parametric = StatisticalComparisonSpec(min_paired_seeds=10, include_tests=("paired_t",), alpha=0.05)
    # The default decision_basis for this spec would be "paired_t_two-sided"
    
    res = compare_paired_arrays(
        label="test_normality",
        left_name="L",
        right_name="R",
        left=left,
        right=right,
        spec=spec_parametric,
    )
    
    assert "shapiro_wilk" in res.tests
    shapiro_p = res.tests["shapiro_wilk"].p_value
    assert shapiro_p < 0.05
    assert res.decision_reliable is False
    
    # Non-parametric decision basis should still be reliable even if non-normal
    spec_nonparametric = StatisticalComparisonSpec(min_paired_seeds=10, include_tests=("wilcoxon",), alpha=0.05)
    res_np = compare_paired_arrays(
        label="test_normality_np",
        left_name="L",
        right_name="R",
        left=left,
        right=right,
        spec=spec_nonparametric,
    )
    assert res_np.decision_reliable is True

def test_comparator_shapiro_zero_variance():
    from malthusjax.stats.comparator import compare_paired_arrays
    from malthusjax.stats.core import StatisticalComparisonSpec

    # Create an identical distribution of differences
    # E.g., zero variance
    left = np.zeros(20)
    right = np.zeros(20)
    
    spec_parametric = StatisticalComparisonSpec(min_paired_seeds=10, include_tests=("paired_t",), alpha=0.05)
    
    res = compare_paired_arrays(
        label="test_zero_variance",
        left_name="L",
        right_name="R",
        left=left,
        right=right,
        spec=spec_parametric,
    )
    
    # Shapiro-Wilk should not crash on zero variance; p-value should be forced to 1.0
    assert "shapiro_wilk" in res.tests
    shapiro_p = res.tests["shapiro_wilk"].p_value
    assert shapiro_p == 1.0
    # Because shapiro_p >= 0.05, the decision basis remains reliable
    assert res.decision_reliable is True
