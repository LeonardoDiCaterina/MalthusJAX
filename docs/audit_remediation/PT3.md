# PT3 — adapter adoption and explicit timing availability

Implemented after PT1 `f32415c1` and PT2 `d27b2241` on `codex/audit-remediation`. These are local CPU procedure fixes, not H100 throughput evidence.

## Adapter coverage

| Invocation path | Procedure / limitation |
|---|---|
| Native genetic Composer backend | Retained prepared executable, full input/output/effect waits, original `GeneticEngine.run` best-genome finalization retained |
| Native MAP-Elites Composer backend | Retained prepared executable, full waits, preparation failures propagate |
| Universal wrappers: EvoSAX, QDax, TensorNEAT | Initialize and wait, lower/compile, invoke the returned compiled object, wait for complete state/history and effects |
| Universal Python-loop mode (including default Kozax path) | Optimization still runs; prepared execution timing is unavailable with an explicit reason |
| Universal `compile=False` | Optimization still runs; no potentially compilation-contaminated `execution` number is emitted |
| Direct legacy EvoSAX adapter | Compile the actual input/key signature, invoke that compiled object, full waits; no throwaway evolution with a different key |
| Native MO Composer adapter | Prepare the actual scanned run rather than an eager throwaway step; full waits and explicit warmup/execution buckets |
| IslandEngineAdapter | Cache and prepare the outer run including global-best finalization; full waits. Nondivisible/invalid horizons and unsupported logging declare execution timing unavailable pending P6/feature work |
| Alternate QD engine-catalog `.run` adapter | Prepare actual scan; expose `last_timings` / `last_timing_metadata` without changing the three-value algorithm API. Unsupported step logging invalidates timing |
| Legacy island-QD Python outer loop | Complete outputs are synchronized; only total diagnostic duration is recorded, with prepared execution explicitly unavailable |
| Bare engine `.run()` without preparation | Remains available; documented elapsed time can include compilation and must not be treated as prepared throughput |

Static option/object replacements invalidate cached universal/legacy loops, preventing a changed generation count or logging configuration from using an old closed-over computation. In-place mutation of third-party strategy internals is unsupported; reconstruct the adapter for those changes. Island/catalog caches likewise refresh when their engine/model or generation configuration changes.

## Phase meaning and compatibility

Valid adapter timings are labeled **prepared_first_execution**. They are not automatically steady-state measurements. Pending JAX effects (including debug callbacks) are explicitly drained at benchmark boundaries in addition to array/PyTree readiness. This prevents callback work from escaping the measured run merely because it has no returned array dependency.

The returned dictionary adds `timing_metadata`, declaring validity, phase, scope, and an unavailable reason when relevant. Clients that check an exact three-key result must accept this new field. Consumers must not assume every successful optimization has an `execution` timing: unsupported modes omit it while retaining optimization results and diagnostic total/warmup durations.

The runner preserves this declaration in `RunResult`, artifact serialization, and empirical per-seed extraction. Older result files remain readable with empty timing metadata. This minimal preservation is pulled forward from PT4 so adapter limitations are not lost on export; it is not the complete versioned timing schema or compatibility validator.

PT4 remains necessary: comparator seed exclusions and analyzer fallback from execution to total have not been repaired here. Until PT4, do not use those legacy analysis paths to claim a corrected speed ratio. Full manifests, first-use/repetition policy, and the focused H100 validation remain PT4/PT5 work. Job 1303 is unchanged historical evidence.

## Regression evidence

- Initial PT3 regressions: six failed and one passed against the prior adapter behavior. They exercised actual compiled-call invocation, initialized-input readiness, truthful mode reporting, MO scan preparation, and island timing availability.
- Core engine/benchmarking/statistics validation: 239 passed, 21 warnings, 84.02 s. Includes prepared-run, boundary, actual generator/export, result serialization, and statistical regressions.
- Broader QDax/TensorNEAT/Kozax/island selection: 77 passed, six skipped, 20 warnings, 437.06 s. Skips remain skips; no unexecuted optional path is claimed validated.
- Final prepared-run / readiness / adapter-contract selection before the callback refinement: 34 passed, two warnings, 55.07 s. Covers actual upstream EvoSAX, QDax and TensorNEAT factories, MO, alternate QD catalog path, first/reordered seeds, replaced static options, and unavailable timing round-trip.
- Extra genetic/sharding guards: 21 passed, 14 warnings, 70.68 s. Includes named sharding on one CPU, existing HOF/engine-logging/compiler invariants. This is not multi-GPU validation.
- Wider EvoSAX/legacy selection initially produced 139 passes, two schema-assertion failures, two skips and ten expected failures. Both failures expected the obsolete three-key result dictionary. The shared compliance assertions now require timing metadata and handle unavailable timing explicitly; the two corrected cases plus the first-genetic-execution compilation guard passed (three passed, one warning, 8.01 s).
- Legacy static-option replacement, result schema, and determinism follow-up: four passed, 8.05 s.
- After callback-boundary adoption, legacy result-schema/determinism checks were repeated: two passed, 7.73 s.

Final callback-aware verification: `python3 -m pytest tests/core/test_timing.py tests/engine/test_prepared_run.py tests/composer/test_timing_contract.py --no-cov -q` — **35 passed**, two warnings, 26.60 s. This includes explicit effect completion, named sharding on one CPU, and static configuration invalidation. All test counts describe their stated selections and overlap; they are not summed as distinct tests. No wall-clock speed threshold was used as an isolation pass condition. The full repository suite and GPU gate are not claimed complete.
