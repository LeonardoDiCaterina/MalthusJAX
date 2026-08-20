# `malthusjax.stats`

Seed-aligned statistical comparison between two runs — e.g. MalthusJAX vs. evosax on
the same problem and seeds, or two configurations of the same engine. Runs paired
hypothesis tests (Wilcoxon, paired t-test, sign test, TOST), applies suite-level
multiple-testing correction, checks whether parametric test assumptions actually
hold, and reports effect sizes alongside every p-value.

> Standalone: works on any two aligned arrays of per-seed metrics. It does not
> require MalthusJAX's engines, the Composer, or TOML configs — if you have two
> seed-aligned numpy arrays from anywhere, you can compare them here.

## Package map

| File | What's in it |
|---|---|
| `core.py` | Data types — `MetricVector`, `PairedSample`, `StatisticalComparisonSpec`, `StatisticalComparisonResult`. Start here to see what goes in and comes out. |
| `tests.py` | The individual paired tests: `wilcoxon`, `paired_t`, `sign_test`, `tost`. |
| `comparator.py` | Orchestration — runs the configured tests, checks normality, decides pass/fail and reliability. |
| `correction.py` | Suite-level multiple-testing correction (Holm, FDR-BH). |
| `effects.py` | Effect size computation (Cohen's dz and friends). |
| `io.py` | Renders a result as Markdown (for `mjax parity` / reports) or JSON. |

## Quick start

```python
from malthusjax.stats.core import MetricVector, PairedSample, StatisticalComparisonSpec
from malthusjax.stats.comparator import StatisticalComparator

left = MetricVector(name="malthusjax", values=my_left_array)
right = MetricVector(name="evosax", values=my_right_array)
sample = PairedSample(left=left, right=right, label="sphere_d10")

spec = StatisticalComparisonSpec(metric_name="best_fitness", alpha=0.05)
result = StatisticalComparator().compare(sample, spec)

print(result.decision_pass, result.decision_reliable)
```

`left` and `right` must be the same length and matched by seed (index `i` in each
array should come from the same seed). `min_paired_seeds` on the spec (default 10)
is enforced before any test runs.

## Reading a result

| Field | Meaning |
|---|---|
| `decision_basis` | Which test (`"wilcoxon"`, `"paired_t"`, `"sign"`, `"tost"`) drives the formal decision. Default is `"wilcoxon"`. |
| `decision_pass` | Pass/fail per the hypothesis in `spec`, based on the `decision_basis` test's (corrected) p-value. |
| `decision_reliable` | `False` when `decision_basis` is a parametric test (`paired_t`/`tost`) and the data fails a normality check — i.e. `decision_pass` is computed but shouldn't be trusted as-is. Always `True` for non-parametric bases. Check this before acting on `decision_pass`. |
| `tests` | Dict of every test in `spec.include_tests`, each with its own statistic/p-value. Only the one named by `decision_basis` feeds the formal decision — the rest are diagnostic. The rendered Markdown/JSON marks which one is primary. |
| `effects` | Effect size for the comparison — always computed and surfaced, not just significance. |
| `alpha` | Significance threshold used (default 0.05). |
| `mean_diff_left_minus_right`, `median_diff_left_minus_right`, `wins_left`/`wins_right`/`ties` | Descriptive summary, independent of any test. |

At the suite level (`StatisticalSuiteResult`, multiple problems run together), each
problem's `decision_basis` p-value is what gets pooled and corrected — the other
tests in each problem's `tests` dict are not part of the correction pool.

## Methodology notes

- **Paired, not independent.** All tests assume `left[i]` and `right[i]` share seed
  `i`; this is a paired-design comparison, not a two-sample test.
- **Multiple-testing correction is on by default.** `StatisticalComparisonSpec.multiple_testing`
  defaults to `MultipleTestingPolicy.HOLM`, applied across a suite's `decision_basis`
  p-values. Set it to `NONE` explicitly if you want raw, uncorrected p-values (not
  recommended for anything you plan to report — see Limitations).
- **Wilcoxon uses Pratt's method for zeros.** Exact ties (identical values on both
  sides — common when two engines both hit the same optimum) are rank-penalized,
  not dropped from the sample.
- **Parametric tests are normality-gated.** Before trusting `paired_t`/`tost`, a
  Shapiro-Wilk test runs on the paired differences. If it fails, `decision_reliable`
  is set to `False` and the Markdown report shows an explicit warning — the p-value
  itself is still reported (nothing is silently substituted or hidden), it's just
  flagged as not to be trusted at face value.
- **Shapiro-Wilk is a weak check at the sample sizes typical here.** With the usual
  handful of seeds, it has limited power — passing it is not strong evidence of
  normality, only an absence of a detected red flag.

## Limitations

- Only the `decision_basis` test determines `decision_pass`. The other tests in
  `include_tests` are computed and shown for context but don't independently
  affect the formal decision or get corrected for multiple comparisons.
- `decision_reliable` currently only checks normality for parametric bases; it
  doesn't check other test assumptions (e.g. Wilcoxon's symmetry assumption).
