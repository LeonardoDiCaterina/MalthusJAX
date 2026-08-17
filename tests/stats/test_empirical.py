import numpy as np

from malthusjax.stats.comparator import StatisticalComparator, compare_paired_arrays
from malthusjax.stats.core import (
    MultipleTestingPolicy,
    PairedMetricDataset,
    StatisticalComparisonSpec,
)


def test_empirical_per_test_fpr():
    """Verify that individual tests hit roughly the nominal alpha rate on identical populations."""
    n_iters = 500
    n_seeds = 30
    alpha = 0.05

    spec = StatisticalComparisonSpec(
        alpha=alpha,
        include_tests=("wilcoxon", "paired_t", "sign"),
        multiple_testing=MultipleTestingPolicy.NONE,
    )

    rejects = {"wilcoxon": 0, "paired_t": 0, "sign": 0}

    rng = np.random.default_rng(42)
    for _ in range(n_iters):
        # Draw two identical populations (H0 is true)
        # We add some normal noise to both, but the mean difference is exactly 0
        left = rng.normal(0, 1, n_seeds)
        right = left + rng.normal(0, 0.1, n_seeds)  # slight jitter so not exact ties everywhere

        metrics = compare_paired_arrays(
            label="test", left_name="left", right_name="right", left=left, right=right, spec=spec
        )

        # Check decisions
        for test_name, res in metrics.tests.items():
            if test_name not in rejects:
                continue
            if res.p_value is not None and res.p_value < alpha:
                rejects[test_name] += 1

    fpr_wilcoxon = rejects["wilcoxon"] / n_iters
    fpr_paired_t = rejects["paired_t"] / n_iters
    fpr_sign = rejects["sign"] / n_iters

    print(f"\nEmpirical per-test FPR (n={n_iters}):")
    print(f"Wilcoxon: {fpr_wilcoxon:.3f}")
    print(f"Paired T: {fpr_paired_t:.3f}")
    print(f"Sign:     {fpr_sign:.3f}")

    # 99% binomial confidence interval for n=500, p=0.05 is roughly [0.025, 0.08]
    assert 0.02 <= fpr_wilcoxon <= 0.09
    assert 0.02 <= fpr_paired_t <= 0.09
    assert 0.02 <= fpr_sign <= 0.09


def test_empirical_family_wise_error_rate():
    """Simulate a 10-problem suite and measure the Family-Wise Error Rate (FWER)."""
    n_iters = 200
    n_problems = 10
    n_seeds = 30
    alpha = 0.05

    spec = StatisticalComparisonSpec(alpha=alpha, include_tests=("wilcoxon", "paired_t", "sign"))

    comparator = StatisticalComparator()
    rng = np.random.default_rng(1337)

    suite_false_positives = 0

    for _ in range(n_iters):
        datasets = []
        for p in range(n_problems):
            left = rng.normal(0, 1, n_seeds)
            right = left + rng.normal(0, 0.1, n_seeds)
            ds = PairedMetricDataset(
                label=f"prob_{p}",
                left_name="L",
                right_name="R",
                seeds=list(range(n_seeds)),
                left_values=left,
                right_values=right,
                metric_name="fitness",
                metric_source="eval",
            )
            datasets.append(ds)

        suite_res = comparator.compare_suite(datasets, spec)

        # A suite has a false positive if ANY problem in the suite falsely rejects H0
        suite_rejected = False

        if suite_res.adjusted_p_values:
            for prob_label, pvals in suite_res.adjusted_p_values.items():
                for test_name, p in pvals.items():
                    if p < alpha:
                        suite_rejected = True
                        break
                if suite_rejected:
                    break
        else:
            for result in suite_res.results:
                if not result.decision_pass:
                    suite_rejected = True
                    break

        if suite_rejected:
            suite_false_positives += 1

    fwer = suite_false_positives / n_iters
    print(f"\nEmpirical Family-Wise Error Rate (n={n_iters} suites of {n_problems} problems):")
    print(f"FWER: {fwer:.3f}")

    # After Fix 4, this should be bounded properly near alpha
    assert fwer <= 0.08
