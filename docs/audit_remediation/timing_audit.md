# Timing procedure audit — 8 November 2026

Audited source: `e5971e968bb2c77428fa1d283a22c0ba666668fd`, branch `codex/audit-remediation`.

## Conclusion

The native MAP-Elites first-run measurement is demonstrably contaminated by tracing and compilation. Warmup and execution request semantically identical kernels using different Python cache keys. A local probe of the actual adapter and base-engine execution path captures XLA compilation inside both timing phases on seed 42, then no additional trace on seed 43.

This is a measurement defect, not evidence that native generation execution is intrinsically 100 times slower. It also does not establish a native speed advantage: the remaining warm observations are too few and are not yet controlled throughput measurements.

The correction must separate preparation from execution for every seed. Removing the unfavorable seed or adjusting its measured value is not a valid repair. Keep the original job outputs unchanged.

This audit changes no production execution/timing code. It adds this report, a runnable diagnostic, recorded local evidence, and a transcription of the cluster observations.

## Evidence and scope

- [Cluster observations](timing_cluster_observations.json): user-pasted output from job 1303; raw cluster files were not downloaded. JAX 0.12.0.1 on GPU, smoke workload, 32 offspring per generation, 15 generations, two seeds.
- [Diagnostic](timing_probe.py): actual native adapter/base-engine code with a tiny CPU toy engine, public `jax.log_compiles` instrumentation, and phase-boundary probes. It requires no optional QDax implementation or expensive RL benchmark.
- [Recorded local evidence](timing_probe_evidence.json): JAX 0.11.0 on CPU. This proves code-path and boundary behavior on that environment, not an H100 speed ratio or exact decomposition of the cluster's 632 ms.

Run the diagnostic from the repository:

```bash
python3 docs/audit_remediation/timing_probe.py
```

Existing tests were also run:

```bash
python3 -m pytest tests/test_timing_isolation.py tests/benchmarking/test_empirical.py tests/benchmarking/test_empirical_generator.py --no-cov -q
```

All 29 passed in 7.66 seconds despite the reproduced timing defects. Ruff passed for the diagnostic. The tests do not currently guarantee the absence of compilation inside measured execution.

## Observed cluster measurements

| Pipeline | Seed 42 execution | Seed 43 execution |
|---|---:|---:|
| Native MalthusJAX | 0.632122 s | 0.001463 s |
| Upstream QDax adapter | 0.004266 s | 0.002083 s |

The native first run is approximately 432 times its second run. The current ratio uses each backend's median across two runs. With two samples, the median is their arithmetic midpoint, so the cold measurement strongly affects the result: 21.1195 vs 0.211633 ms/generation, QDax/native ratio 0.0100208.

The second-seed ratio alone is approximately 1.42 in favor of native execution. That is descriptive of one sample and must not replace the original aggregate as a validated performance claim. Compilation, residual initialization, runtime first-use overhead, and device contention are not separately recoverable from the pasted durations.

## Findings

### T01 — High: native warmup and execution compile different cached functions

Locations: [MAP-Elites adapter](../../src/malthusjax/composer/adapters/map_elites_adapter.py:119), [genetic adapter](../../src/malthusjax/composer/engine_factory.py:124), [base run](../../src/malthusjax/engine/base.py:260), [kernel factory](../../src/malthusjax/engine/base.py:373).

Warmup calls:

```python
_get_evolution_kernel(params, compile_jit=True, unroll_num=1)
```

Base execution calls:

```python
_get_evolution_kernel(
    params, compile_jit=True, unroll_num=1,
    return_history=True, step_logging=None,
)
```

The `lru_cache` sees different supplied arguments rather than canonicalizing defaults. Each miss builds a new closure and JIT function. The returned functions are distinct even though the intended run configuration is identical. Explicit logging overrides create an additional configuration-matching obligation.

**Reproduced:** `same_warmup_and_execution_function=false`; seed 42 traces in warmup and execution. Public JAX logs show XLA compilation of `_evolve_loop` in both phases (about 46.7 ms and 37.3 ms in this local diagnostic). Seed 43 adds no trace. The local measured execution becomes approximately 47.8 ms vs 0.0759 ms; those toy durations are diagnostic only.

**Scope:** the same mismatched call forms appear in the genetic adapter. The event-level reproduction exercises the native MAP-Elites adapter. It does not quantify the genetic adapter's GPU contamination.

**Correction requirement:** one normalized run specification must select the exact same prepared executable for compilation and measured execution, including logging/history/dtypes/sharding/static configuration. Preserve engine-specific finalization when consolidating this path; bypassing `GeneticEngine.run()` blindly could omit its best-genome finalization.

### T02 — High: a failed warmup is silently accepted

Location: [MAP-Elites adapter](../../src/malthusjax/composer/adapters/map_elites_adapter.py:119).

The warmup `try/except Exception: pass` allows execution to continue even if lowering or compilation fails. A later successful run can compile on demand and report the result as execution time without any indication that preparation failed.

**Reproduced:** the probe deliberately raises during `.lower()`. The adapter suppresses the error and still emits execution timing through a stubbed successful run. This isolates error handling; it is not a claim that job 1303 encountered such an exception.

**Correction requirement:** fail the run or explicitly mark timing invalid and retain the failure reason. No silent fallback to a differently timed path.

### T03 — High: initialization and completion barriers are inconsistent

Locations: [native init/timer](../../src/malthusjax/composer/adapters/map_elites_adapter.py:112), [universal init/timer](../../src/malthusjax/composer/adapters/base.py:281), [base completion](../../src/malthusjax/engine/base.py:276), [genetic finalization](../../src/malthusjax/engine/genetic_fastengine.py:973).

Native MAP-Elites and universal adapters do not explicitly synchronize the complete initialized state before the execution timer begins. Lowering/compilation can overlap pending device initialization; it does not by itself prove all input work is complete. Genetic initialization blocks a best-fitness scalar, which is narrower than a full input-state readiness contract.

Native base execution blocks only `final_state.best_fitness`, whereas the universal adapter attempts to block final state and history. The diagnostic confirms the native barrier touches only that scalar, while the universal helper visits the three leaves in the toy object. A scalar can depend on the main computation, so this does **not** prove that some particular GPU output remains unfinished. It proves inconsistent completion contracts that should not underlie a cross-backend timing claim.

Genetic `.run()` also schedules final best-genome extraction after the base timer's barrier. The enclosing adapter stops its timer without a new full-output barrier. Its measured endpoint therefore differs from the upstream adapter's endpoint.

**Correction requirement:** input readiness before starting execution timing; readiness of the complete documented output tree before stopping; explicit treatment of finalization and host/device transfer. Test boundary order with pending leaves and validate actual accelerator behavior separately. JAX's [asynchronous dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html) explains why dispatch return alone is insufficient.

### T04 — High: MO and island adapters do not provide comparable execution measurements

Locations: [MO adapter](../../src/malthusjax/composer/mo_factory.py:44), [island adapter](../../src/malthusjax/engine/island_model/adapter.py:98), [other QD adapter run](../../src/malthusjax/engine/qd/adapter.py:114).

The MO adapter's eager one-step call does not compile the scanned run. It then times `run(..., time_it=False, compile=True)` without an output barrier. Its execution bucket can mix host compilation and asynchronous submission while device work is waited for later during history conversion.

The island adapter includes initialization in its execution bucket, provides warmup=0, and stops its timer before converting best fitness to a host scalar. A second QD adapter's `run()` method also captures elapsed time immediately after scan submission. These are source-confirmed procedure defects; this audit does not claim that the paper's Dataset 3 used all of these adapters.

**Important distinction:** the Dataset 3 scaling generator instead calls an explicitly compiled outer scan and blocks its history. That is stronger than the island adapter path, but it still lacks an explicit initialized-input barrier before starting timing. Record its separate provenance rather than applying the adapter finding indiscriminately.

### T05 — Medium/high: the universal synchronization helper is incomplete and suppresses errors

Location: [universal helper](../../src/malthusjax/composer/adapters/base.py:36).

The helper recursively inspects `vars(obj)` rather than using the registered PyTree structure. Its broad exception handler encloses recursive synchronization, so a child's device error can be swallowed.

**Reproduced:** an intentional synchronization failure in a nested object does not propagate. A registered slots-based PyTree containing a pending leaf receives no barrier. This is a robustness defect; whether a particular upstream state triggers it requires backend-specific verification.

**Correction requirement:** synchronize supported PyTrees through JAX tree semantics, handle any intentionally supported non-PyTree containers explicitly, and propagate device execution errors. Avoid exception swallowing around actual waits.

### T06 — High for inference: timing consumers disagree about what to aggregate

Locations: [paired timing selection](../../src/malthusjax/stats/comparator.py:522), [CSV timing selection](../../src/malthusjax/stats/comparator.py:792), [empirical median](../../src/malthusjax/benchmarking/empirical.py:121), [generic analyzer fallback](../../scripts/benchmark_analyzer.py:110).

Two statistical dataset paths exclude seed 0, or the first sorted seed if 0 is absent. The empirical generator retains every seed. Thus different reports can use different observations for the same nominal timing comparison. Exclusion is based on seed identity, not preparation metadata or recorded execution order.

The generic analyzer can silently fall back from execution to total or runner wall time under the same `execution_time` column. New empirical extraction is stricter and rejects absent execution measurements, but it cannot verify that an adapter assigned work to the correct bucket.

**Correction requirement:** versioned timing semantics and explicit validity/phase metadata. Keep every raw observation; do not silently select or relabel measurements. A requested steady-state analysis must reject incompatible/missing timing phases rather than mix them. Report any invalid timing and its reason separately from optimization success.

### T07 — Medium: cold execution, steady execution, and end-to-end cost are not distinguished

Locations: native/universal `run_once`, [pipeline order/cache clearing](../../src/malthusjax/composer/composer.py:722), runner and aggregation.

The universal adapter prepares the same JIT callable, which avoids T01, but it does not explicitly precondition execution hardware before measuring a first invocation. Compilation-free execution can still include runtime initialization, library autotuning, allocator activity, or different GPU clock state. This audit does not establish which affected QDax's first seed.

Pipelines run in fixed order and JAX caches are cleared after each pipeline. This is not inherently wrong, but the policy needs to be part of the benchmark specification. It does not reset every driver/runtime/device condition. Repeating whole scripts without recording these conditions is not a controlled cache experiment.

The adapter `total` generally ends before history/summary serialization, while runner duration includes additional host work; MO uses another endpoint. These can be legitimate separate metrics, but their labels need explicit scope.

**Correction requirement:** separate compile time, first execution, any prespecified preconditioning, measured repeated execution, postprocessing, and end-to-end wall time. Preserve first-execution observations even when the study's target is steady-state execution. Use a fixed repetition/order policy chosen before reviewing results.

### T08 — High assurance gap: timing tests validate fields, not isolation

Locations: [timing isolation tests](../../tests/test_timing_isolation.py:13), [adapter overhead tests](../../tests/composer/test_adapter_overhead.py:29).

The isolation test checks that fields exist and are positive. It does not observe tracing/compilation during the measured interval or verify barriers. Adapter-overhead tests run a warmup first and time the entire second `run_once`, which includes initialization and host postprocessing. They also substitute a mock native execution path and tolerate broad timing differences. Consequently these tests cannot catch the actual first-seed bug.

**Correction requirement:** deterministic event-order/tracing assertions against real adapter code, plus separate appropriately scoped hardware measurements. Avoid flaky CPU wall-clock thresholds as evidence of phase isolation.

## Required measurement contract

Adopt one shared protocol for all maintained adapters, with versioned metadata:

1. **Initialize:** create the designated seed's state, including population overrides and task state. Synchronize all timed inputs. Record initialization separately.
2. **Prepare:** resolve the complete run configuration, lower and compile the exact executable to be invoked. Record compilation separately. Preparation errors invalidate timing rather than silently falling back.
3. **Record first execution:** invoke the prepared path with the designated state; synchronize its complete output. Keep the first-execution measurement, seed, execution order, signature and cache policy.
4. **Measure the declared target:** if studying steady-state throughput, use a predetermined repetition/preconditioning policy applied to both backends. Reset/copy identical inputs as appropriate; prepare and synchronize buffer copies outside the measured interval. Respect donated buffers. Record repetitions and any preconditioning instead of deleting unfavorable observations.
5. **Finalize/export:** define whether device-side finalization belongs to execution and apply the same scope to all backends. Record host postprocessing and actual end-to-end duration separately. Keep raw samples, validity, and failure reasons.
6. **Aggregate:** require matching timing-schema and phase labels; pair the intended observations; report uncertainty. Do not infer causal architectural superiority without controlling algorithmic work and archive-update differences.

JAX's [benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html) supports separating compilation and asynchronous completion from the operation being compared. Python's [cache documentation](https://docs.python.org/3/library/functools.html#functools.lru_cache) explains why differing call argument patterns must not be assumed to share one cached entry. These references support the procedure; the project-specific root cause is established by the recorded probe.

## Regression gates before another expensive run

- No `_evolve_loop` trace or XLA compilation event inside the measured steady-execution interval on the first seed, subsequent seeds, or reordered seeds.
- Prepared and invoked configuration/executable agree for default arguments, explicit defaults, logging overrides, history enabled/disabled, dtype and sharding variants.
- Initialization/copy barriers precede the timer; every documented output barrier precedes its end. Include custom PyTrees and failure propagation.
- Warmup errors cannot emit apparently valid steady-execution timing.
- Warmup/preconditioning cannot advance the measured seed's generation, change its initial population, reuse donated buffers, or alter its expected optimization trajectory.
- MO, island, native genetic, native QD, and upstream adapters either meet the common contract or clearly declare an incompatible/unavailable phase.
- Aggregation retains all raw seeds/trials and rejects mixed timing definitions; old artifacts are explicitly marked legacy/unknown.
- First validate a small CPU event/tracing test suite. Then run a small H100 timing-focused experiment with several seeds and predetermined repetitions. Check compile logs and output equivalence before attempting another full RL-heavy suite.

## Recommended implementation order

Pull the timing portion of P6/P9 forward because it blocks trustworthy cluster comparisons. First fix executable preparation/reuse and error propagation; next standardize input/output barriers and timing fields; finally fix consumer selection/aggregation and add the regression gates. Coordinate this with P3/P4 without silently changing objective definitions or statistical hypotheses during the timing repair.

Keep `results/remediation-1303/` as historical evidence. The next measurement run should have a new output directory and timing-schema identifier. Both the apparent 100× slowdown and the second-seed 1.42× speedup remain descriptive observations until that protocol is validated.
