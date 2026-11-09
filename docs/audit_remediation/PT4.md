# PT4 — Versioned timing observations and consistent consumers

Implemented on `codex/audit-remediation`, after PT1–PT3. Local CPU validation only; PT5 hardware validation is still required. Job 1303 and historical paper measurements have not been modified or regenerated.

## Record and measurement contract

`core/timing_records.py` introduces `TimingPolicy`, `TimingTrial`, and `TimingRecord` (timing schema version 1, independent of the experiment schema and objective status). Runner records bind each observation to its seed and execution order. They retain phase durations, executable class/static options, input shapes/dtypes/sharding, cache/reset policy, backend mode, source revision/dirty diff identity/import path, Python/JAX/jaxlib/NumPy/platform/device identity, and validity diagnostics.

Adapters separately mark initialization and readiness, executable preparation/compilation, and host postprocessing around the existing fully synchronized execution boundaries. “Compilation” is the elapsed preparation call, including a cached executable lookup when applicable; it is not a claim that XLA compiled anew. Input-signature collection is outside the compilation and execution intervals. The runner records its monotonic end-to-end duration separately. Device/effect finalization belongs to execution; history conversion belongs to host postprocessing. These named intervals need not sum to wall time because signature capture, instrumentation, and runner bookkeeping also take time.

Numeric compatibility aliases remain explicit: `execution`/`execution_time`/`time_execution` mean `prepared_first_execution`; `warmup` means initialization plus preparation; adapter `total` stops after execution; runner `duration_seconds` is a different phase. The latter phases remain available as raw descriptive values and cannot be substituted for execution in performance comparisons.

The default target is one prepared first execution per seed. A frozen policy can instead request a predetermined number of preconditioning and measured repeat trials, each initialized from a fresh key with the same seed while retaining the executable cache. Every first execution, preconditioning trial, repeat, and repeat result is preserved. Repeated values reduce to a within-seed median, never additional independent seeds. Seeded objective/history disagreement, changed signatures, or any invalid trial makes comparative timing unavailable while retaining the first objective result. Repeats are named `prepared_execution_repeat`; they are not automatically claimed to establish hardware steady state.

Example explicit policy (not launched):

```python
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.core.timing_records import TimingPolicy

runner = BenchmarkRunner(
    engine,
    timing_policy=TimingPolicy(
        target_phase="prepared_execution_repeat", preconditioning=1, repetitions=3
    ),
)
result = runner.run([0, 42, 43])
# Select this phase explicitly; the first-execution alias cannot select it.
comparison.timing_data("prepared_execution_repeat")
```

## Consumer and compatibility changes

- Both comparator paths retain seed 0 and the first seed. Auxiliary timing summaries are descriptive medians with counts and explicit unavailable reasons; they no longer infer timing significance or sum overlapping timing buckets. In-memory comparisons retain input seed order and disclose unmatched seeds.
- Plotting no longer discards the first sample or the slowest baseline pair. Shared selection rejects duplicate seeds, unknown schemas, unavailable observations, inconsistent trial order/counts/phases, invalid durations, and incompatible policies/scopes/modes/environments.
- Explicit timing comparisons and analyzer extraction reject legacy timing and phase substitution. Objective-only legacy analysis remains usable. Aggregate CLI reports retain objective output if the timing grid is unavailable.
- Empirical extraction retains complete per-seed timing records; the generator checks contracts across backends. JSON, timing-trial CSV, Markdown timing summaries, tables, and plots select the same observations. Descriptive summary tables display the median and record the estimator/count. CSV retains every raw trial and its complete record, including unavailable observations.
- Legacy and future-schema records remain readable and preserved, but are not upgraded to corrected timings by the presence of a numeric execution alias. Nonfinite typed trial durations retain an explicit raw representation and an unavailable status.

The empirical generator compares first execution explicitly. Repeated-phase collection is exposed through the runner API and explicit consumer phase selection; the cluster entrypoint/configuration belongs to PT5. Statistical decision policies and publication inference remain P4 work; complete manifests and timeout handling remain P9 work.

## Regression evidence

Environment: local CPU, `/opt/anaconda3/bin/python3`, Python 3.12.5, JAX 0.11.0. No timing-speed threshold is used as an isolation assertion.

```bash
/opt/anaconda3/bin/python3 -m pytest --no-cov -q \
  tests/core/test_timing_records.py tests/core/test_timing.py \
  tests/engine/test_prepared_run.py tests/composer/test_timing_contract.py
# 60 passed, 2 warnings, 29.88 s

/opt/anaconda3/bin/python3 -m pytest --no-cov -q \
  tests/core/test_timing_records.py tests/benchmarking tests/stats \
  tests/test_timing_isolation.py
# 228 passed, 13 warnings, 25.63 s (final strict-schema verification)
```

Selections overlap and counts are not additive. The first command verifies retained executable adoption, first/reordered seeds, complete barriers/effects, preparation failures, and actual optional upstream adapters. The second covers record round-trips, repeat policies and invalid trials, phase/provenance incompatibility, seed 0/order, both comparator paths, empirical generator/table exports, analyzer fallback/duplicate rejection, legacy objective compatibility, and CLI reports. Initial broader verification exposed stale fixtures expecting unversioned timing to be accepted; those fixtures now declare synthetic versioned observations, with separate legacy rejection tests.

Changed timing core/runner/results/comparator and new tests pass Ruff undefined/unused-symbol checks; new timing-record code/tests also pass the configured Ruff rules. `git diff --check` passes. Targeted test commands use `--no-cov`: they do not claim the repository-wide coverage threshold or a full repository test pass. CPU timing gate A is satisfied within the audited paths. GPU gate B, throughput claims, and publication-sized runs remain gated on PT5 and scientific remediation.
