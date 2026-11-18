# Release test failure report

Date: 18 November 2026  
Affected branch: `codex/clean-install-smoke`  
Initial release revision: `45519e01`  
Fix revision: `36f57105ea1ecf6b8fc5b921275b961c7391552b`

## What happened

The full DAH2 test run exposed several independent problems. An incompatible
GPU environment caused broad failures before individual framework behavior could
be assessed. After repairing that environment, the suite exposed two framework
bugs and several test fixtures that had not been updated with the release's
output contract. The remote staging method also omitted Git metadata required
by the publication provenance tests.

The failure count therefore overstated the number of independent defects:
one environment mismatch affected many GPU operations, and one shared test
assertion failed across several adapters.

## 1. JAX and its CUDA plugin had different versions

**Classification:** environment installation problem.

Installing the development and integration extras into the existing GPU
environment changed `jax` and `jaxlib` to `0.11.2`, while the previously installed
CUDA 13 plugin and PJRT package remained at `0.12.0.1`.

These components cooperate at runtime. A working Python import and a successful
`pip check` did not establish that their native interfaces were compatible.
The resulting GPU failure included:

```text
NOT_FOUND: No FFI handler registered for cusolver_geqrf_ffi on a platform CUDA
```

A representative failing CLI test passed on CPU in the same environment. The
CUDA QR smoke check passed after all four JAX packages were aligned at `0.11.2`.
This isolated the initial GPU failure from the experiment configuration and CLI.

**Resolution:** align the existing stack with `jax[cuda13]==0.11.2`. Contributor
instructions now install integration dependencies and the selected backend
together. `make test` and `make test-fast` run
`scripts/check_test_environment.py` before pytest to catch incompatible installed
GPU package versions.

**Limit:** this preflight checks package versions; it does not replace an actual
GPU smoke test or validate every driver and CUDA-library combination.

## 2. GPU-only backend selection excluded the callback backend

**Classification:** test-run configuration problem.

The retry initially used `JAX_PLATFORMS=cuda`. Some logging tests exercise
`jax.debug.callback`, and the tested JAX setup needed an available CPU backend
for those host callbacks. Excluding CPU caused callback-related failures even
after the GPU packages were aligned.

**Resolution:** use `JAX_PLATFORMS=cuda,cpu` for the GPU suite. CUDA remains the
default execution backend; enabling CPU also makes the callback backend
available. CPU-only runs use `JAX_PLATFORMS=cpu`.

## 3. Categorical cardinality was incorrectly treated as dynamic JAX data

**Classification:** framework bug.

`CategoricalGenomeConfig.num_categories` was a dynamic PyTree field. When a
configuration was passed through JIT, JAX could turn that integer into a tracer.
Permutation initialization and crossover validation need the category count as
a concrete configuration value: it determines array shapes and participates in
Python configuration checks. The compiled compliance path exposed a
`TracerBoolConversionError`.

**Resolution:** declare `num_categories` with
`struct.field(pytree_node=False)`, like the other structural configuration
fields. Different category counts can appropriately produce different compiled
specializations.

The catalog compliance fixture also assumed every nonbinary crossover used real
genomes. That assumption became invalid when `OrderCrossover` was registered.
It now supplies categorical permutation parents for that operator.

**Verification:** categorical-genome tests, compiled catalog compliance, release
crossover tests, and an installed-wheel GPU JIT initialization smoke test passed.

## 4. JSON serialization could recurse through mock objects

**Classification:** framework bug, exposed by mocked tests.

`standard_json()` previously treated any object exposing `.tolist()` as an
array. `MagicMock` can manufacture that attribute, and calling it returns another
mock. Recursive conversion therefore kept following mock objects instead of
reaching a JSON value or the unsupported-object fallback.

This caused a strategy test to stall and provided a mechanism for excessive
memory consumption. Separately, Slurm explicitly reported a host-memory OOM
kill for the earlier run with a 24 GB limit. The serializer defect explains the
observed runaway test behavior, but no isolated memory profile was collected to
attribute every byte of that OOM to this one defect.

**Resolution:** only call `.tolist()` for actual NumPy arrays, NumPy scalar
values, and JAX arrays. Other objects use the existing unsupported-object
representation. A regression test checks that a mock's `.tolist()` is never
called and that numeric arrays still serialize correctly, including nonfinite
values becoming JSON `null`.

**Verification:** the serializer and composer strategy tests completed locally
in 2.7 seconds; the regression and installed-wheel smoke test also passed on the
H100. Subsequent full-suite work used a 64 GB allocation, so that run is not a
controlled comparison of peak memory before and after the fix.

## 5. Existing test fixtures described the previous output contract

**Classification:** test maintenance gaps accompanying the release changes.

| Test assumption | Why it failed | Correction |
| --- | --- | --- |
| The catalog's expected keys were complete | `order_crossover` and `native_sphere` were missing | Add both supported registrations and a crossover class-resolution check |
| Adapter results could contain only the older fields | New documented fields such as initialization, unavailable metrics, solution, and archive were rejected | Extend the shared allowed-key set while preserving required-key checks |
| A dummy evaluator could return an untouched population | Its fitness remained the initial `-inf`, which terminal-result validation correctly rejected | Use the base evaluator's real population evaluation method |
| A mocked runner result needed no output metadata | Composer reads the runner's output-options metadata when writing artifacts | Supply the current runner metadata contract in the mock |
| MAP-Elites mocks needed only a few scalar attributes | The adapter now reads metric settings, typed initialization configuration, and repertoire arrays | Supply representative settings, a real genome config, and array-backed repertoire data |
| Missing array-initialization configuration raised `NotImplementedError` | Current input validation raises `ValueError` | Assert the current validation exception and message |
| A timing-only engine could use default solution capture | Its synthetic state has no archive from which to extract a solution | Explicitly disable solution capture in that timing test |

These corrections did not remove the finite-fitness checks, suppress the new
output fields, or skip the failing tests. They made the fixtures represent the
current interfaces. The shared adapter correction repaired repeated failures in
EvoSAX, QDax, and Kozax; those were not separate defects in each backend.

## 6. The remote source snapshot was not a Git checkout

**Classification:** remote test staging problem.

I initially staged the release source using `git archive`. This copied the
tracked files but omitted `.git`. Five publication tests call
`git rev-parse HEAD` to verify revision provenance, so they failed with
`fatal: not a git repository` before reaching their intended assertions.

**Resolution:** supply genuine shallow-checkout Git metadata for the exact pushed
revision and verify that the staged tracked files match `HEAD`. The three
publication test modules then passed all eight tests without changes to their
assertions or the publication implementation.

For future complete source-tree test runs, stage a checkout with revision
metadata. A source archive is sufficient for many package checks but does not
satisfy tests that explicitly require Git provenance.

## Why earlier validation did not catch everything

Earlier release checks covered a narrower set of CPU tests and GPU smoke routes.
They did not establish that the entire optional-integration suite passed after
an additional dependency installation. The later installation changed the JAX
environment, while broad catalog, mocked-adapter, and publication tests exercised
paths outside the earlier selection.

The release implementation also updated production output contracts without
updating every older fixture and catalog allowlist. The full suite was valuable
because it exercised those shared assumptions across multiple extensions.

## Final validation and its limits

The fixes were committed and pushed as `36f57105`. All compute tests remained in
Slurm, as requested. Successful cases were retained by exact pytest node ID;
failed cases were rerun after their corrections, and previously unexecuted cases
were completed in a continuation job.

| Slurm job | Purpose | Result |
| --- | --- | --- |
| `4795` | Broad suite after environment and framework repairs | 729 passed, 5 skipped, 10 expected failures; 10 fixture failures identified |
| `4838` | Rerun those failures and execute remaining cases | 1,102 passed, 19 skipped; one timing fixture and five missing-Git failures identified |
| `4867` | Verify repaired timing module and rebuilt installed wheel | All 12 timing tests passed; GPU JIT, serializer, package consistency, and CLI skill checks passed |
| `4888` | Verify publication modules with exact revision metadata | All 8 tests passed |

After reconciling duplicate cases and replacing earlier failures with their
successful reruns, all **1,871 unique test cases** were accounted for:

- **1,837 passed**
- **24 skipped**
- **10 expected failures**
- **0 unresolved failures**

This is full-suite coverage across multiple Slurm runs, not a claim that one
uninterrupted invocation finished with those totals. Skipped and expected-failure
cases remain distinct from passes. Separate performance benchmarks were not part
of this default suite, and this investigation was not a before/after throughput
benchmark.

The repaired wheel was installed in DAH2's `venv-clean`. Server logs remain under
`/home/20240485/codex_space/release-45519e01/logs/`, and `REPAIR_COMMIT` records the
repaired revision. The original `SOURCE_COMMIT` retains the initial staging
revision.

## Additional local validation

On 18 November 2026, `make test` was also run with coverage in an isolated
macOS ARM64 CPU environment using Python 3.12.15 and JAX/JAXlib 0.11.2, with
development and all integration extras installed.

The initial run completed in 35 minutes 47 seconds with 1,822 passes, 24 skips,
10 expected failures, and 15 scaffold-test failures. Those 15 failures came from
the test invocation: `--basetemp` had been supplied through `PYTEST_ADDOPTS`, so
nested pytest processes inherited a temporary directory that was a parent of
their working directory. Pytest correctly rejected that unsafe configuration.

Supplying the temporary-directory option only to the parent pytest command
resolved the issue. All 17 tests in the scaffold module passed on retry in
47 seconds, without framework or test-source changes. Reconciled local totals
were again **1,837 passed, 24 skipped, 10 expected failures**, with **86% statement
coverage** and no unresolved failures. These totals include that targeted retry;
they do not describe a single all-green invocation.
