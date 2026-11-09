# Audit remediation implementation plan

Prepared 8 November 2026. Baseline: MalthusJAX `51009b8cf7464fe4a56bc40c472bfe8377ed9858`.

Revised after GPU smoke job 1303 and the [timing procedure audit](audit_remediation/timing_audit.md). The audit's T01–T08 findings are now included alongside A01–A22. This revision plans timing repairs; it does not mark them implemented.

## Objective and scope

Make MalthusJAX's execution, analysis, and publication pipeline trustworthy and reproducible, then determine which existing paper results need correction. This plan covers findings A01–A22 from `audit/2026-11-08/REPORT.md`. Implementation progress and verified dispositions are recorded in [the remediation ledger](audit_remediation/findings.md).

The first milestone, safe output handling, has been implemented through P0–P2. Job 1303 exercised that code on the cluster, but exposed a separate timing-boundary defect: native seed 42 included tracing/compilation in the execution bucket. Correct timing is now the immediate next milestone, followed by shared metric definitions, algorithms, and inference. Publication reruns follow an explicit impact assessment. General cleanup and broader refactoring remain deferred.

The audit contains reproduced defects, source-confirmed behavior, and unverified risks. Revalidate each against the current revision before changing code. A finding is closed by evidence, not by merely changing the cited line. If revalidation changes a conclusion, record the correction.

## Working rules

- Preserve existing raw results, paper tables, figures, and manuscript edits. The paper repository currently has substantial uncommitted work; do not stash, reset, overwrite, or commit it as part of a blanket operation.
- Use a dedicated `codex/audit-remediation` branch or isolated worktree when implementation starts. Record the baseline revision, dirty state, and hashes of evidence inputs. Create backups only where needed; avoid duplicating large result trees unnecessarily.
- Treat the two repositories separately: source changes in MalthusJAX, publication tooling changes in MalthusJAX_Paper. Record the compatible revision of each. Stage only files belonging to the current work package.
- Keep each work package in a separate reviewable commit where practical. A commit can include several tightly related tests and modules; avoid repository-wide formatting changes.
- For scientific defects, establish a failing behavioral regression first, implement the fix, then run the relevant integration tests. Use independent mathematical/reference expectations, not snapshots copied from the implementation.
- Promote reusable audit probes into normal test locations. The existing `audit/` directory is Git-ignored and cannot be the sole location of release-critical regression tests.
- New experiment outputs go to a new run directory. Generators do not automatically overwrite or synchronize to the manuscript.
- Do not select margins, seeds, transformations, or stopping rules after seeing whether they produce a favorable conclusion.
- Report unavailable evidence as unavailable. Never replace it with favorable measurements, p-values, success statuses, or placeholder numbers.
- Preserve `results/remediation-1303/` unchanged. Do not remove seed 42, subtract an estimated compilation cost from its duration, or promote seed 43's speed ratio into a paper claim.
- Timing repairs must preserve the designated seed's initial state, random stream, generation count, and optimization result. State copies and warmup runs must respect donated buffers and must not silently advance the measured trajectory.

## Revised execution order and checkpoints

| Order | Work | Status / gate |
|---|---|---|
| 1 | P0–P2: baseline, honest extraction, safe ingestion | Implemented; source `e5971e96`, paper `07700c6`. Full real ingestion still deferred to P10. |
| 2 | PT1–PT3: exact executable preparation, synchronization, adapter adoption | Implemented with CPU regression evidence; see [PT1](audit_remediation/PT1.md), [PT2](audit_remediation/PT2.md), [PT3](audit_remediation/PT3.md). |
| 3 | PT4–PT5: versioned samples, consistent consumers, focused GPU validation | Must pass before claiming corrected throughput or running another large timing comparison. |
| 4 | P3/P4: shared metric semantics and statistical validity | Still open; timing success does not close these findings. |
| 5 | P5/P6/P7: mathematical defects, exact island horizons, regression contracts | P6's shared timing machinery moves into PT1–PT4; scheduling stays in P6. |
| 6 | P8/P9: API contracts, complete run validation and manifests | Timing metadata moves forward into PT4; full provenance and timeout work remain here. |
| 7 | P10/P11: evidence tracing/regeneration and release gates | Publication-scale reruns need timing, scientific validity, and evidence-impact gates. |

**Timing gate A:** deterministic CPU tests prove the measured interval contains no tracing/XLA compilation, preparation errors propagate, barriers enclose the specified work, and seeded outcomes are preserved. This is a procedure gate, not a performance gate.

**Timing gate B:** a focused H100 run verifies the same boundaries under the cluster's JAX/CUDA environment, retains first-use observations and repeats, and exports a validated timing schema. The same 100× slowdown or any other slowdown remains acceptable as a measurement outcome if the procedure is correct. A favorable speed ratio is never a pass condition.

**Publication gate:** PT1–PT5 passing permits controlled performance investigation. It does not authorize equivalence claims, repair QD-score definitions, establish identical algorithmic work, or make old paper results publication-ready. Those require P3/P4/P10 and the relevant algorithm fixes.

## Immediate timing work packages

These packages pull the timing portions of P6/P9/P11 forward without waiting for objective or statistical redesign. Use separate reviewable commits, with failing behavioral regressions before each runtime fix. The [timing diagnostic](audit_remediation/timing_probe.py) and [recorded evidence](audit_remediation/timing_probe_evidence.json) are starting points; promote their contracts into maintained tests.

### PT1 — Prepare and invoke the exact same executable

**Findings:** T01, T02, T08.

**Files:** `src/malthusjax/engine/base.py`, native genetic/MAP-Elites adapters, new engine/adapter timing regressions.

**Changes:** introduce a single preparation boundary using a normalized run specification. Normalize default/explicit arguments before any cache lookup; include effective logging, history mode, static engine configuration, and input signature. Lower/compile the exact executable that measured execution will use, and retain a clear connection between the prepared object and its invocation. Avoid merely duplicating a longer keyword list at every call site. Preserve engine-specific finalization, including genetic best-genome extraction. Remove the silent MAP-Elites warmup exception fallback; failed preparation produces an error or explicitly invalid timing with its reason.

**Acceptance:** the actual first native adapter run has no evolution-loop trace or XLA compilation event inside its measured prepared-execution interval. Cover explicit versus omitted defaults, logging overrides, history modes, and changed signatures. Reordered seeds `(44,42,43)` must not change the contract. A deliberately failing lower/compile cannot produce valid timing. No wall-clock speed threshold is used to establish isolation. Signature changes may compile during preparation, but never silently during the declared steady interval.

**Dependency:** completed P0–P2. This is the next coding task.

### PT2 — Standardize readiness barriers and phase boundaries

**Findings:** T03, T05, T08; supports A07.

**Files:** shared timing/synchronization utility, `engine/base.py`, universal adapter helper, deterministic timing-event tests.

**Changes:** use JAX PyTree semantics for supported state/history containers; explicitly handle any additional documented containers. Synchronize initialized inputs and trial-state copies before starting execution timing. Synchronize the complete specified output tree before stopping it. Propagate device/synchronization errors. Define device finalization and host postprocessing separately and consistently. Use a monotonic high-resolution clock for durations. Do not assume `.lower().compile()` waits for input data or that one best-fitness scalar represents every output boundary.

**Acceptance:** controlled pending-leaf probes record input waits before timer start and all output waits before timer stop; include nested mappings, Flax/custom PyTrees, slots-based states, and synchronization failures. Failure validity is separate from a successful optimization result. No logger, serialization, or host metric conversion accidentally supplies the required barrier after the timer has stopped. State-copy work is outside a kernel-only measurement and recorded in preparation/end-to-end cost.

**Dependency:** PT1's prepared-execution interface. Keep synchronization validation independent of objective sign and statistical decisions.

### PT3 — Apply the contract across maintained adapters

**Findings:** T03, T04, T07, T08; A07.

**Files:** native genetic/QD adapters, universal upstream adapter, MO adapter, island adapter, alternate QD adapter(s), adapter compliance tests.

**Changes:** inventory which factory/catalog paths resolve to each adapter, then apply the same preparation/readiness/measurement contract. Replace eager one-step calls presented as scan compilation and timers ending immediately after asynchronous submission. Record the execution mode for Python-loop/non-JIT integrations; if they cannot provide the requested phase, report it unavailable instead of presenting a comparable steady-state number. Audit legacy adapters for supported callers before deciding whether to migrate or explicitly limit them. Do not alter island migration schedules in this timing commit; correct that independent algorithmic contract in P6 before using nondivisible horizons in comparisons.

**Acceptance:** native genetic, native MAP-Elites, upstream QDax/EvoSAX, MO, and island adapter tests cover their actual production invocation paths. Each offers the requested timing contract or a machine-readable unsupported reason. Preserve seeded final fitness/history/genomes versus the untimed reference path, including logging/history variants and donated-buffer safety. Record phase coverage for optional backends not available locally rather than treating skips as validation. A fixed-step toy island can validate timing before P6 repairs real scheduling.

**Dependency:** PT1/PT2. This establishes timing gate A, together with PT4's consumer tests.

### PT4 — Version timing records and stop mixing observations

**Findings:** T06, T07, T08; A19/A21 and timing-validity portion of A12.

**Files:** runner/result/IO models, empirical extraction, statistical comparator, benchmark analyzer, manifests and compatibility tests.

**Changes:** add a typed/versioned timing record, separate from objective success. Record initialization, compilation, first execution, any preconditioning, measured repetitions, device finalization scope, host postprocessing, and end-to-end duration. Keep seed, trial ID, run order, target phase, validity/reason, executable/input signature, cache policy, backend mode, and source/environment identity. Numeric legacy timing aliases need an explicit mapping to the new phase and cannot silently change meaning. Serialize metadata through runner/artifact round-trips without relying on code that currently converts every timing field into a float.

First execution remains recorded for every seed. Steady-state measurements use a predetermined repetition/reset/preconditioning policy applied consistently to both backends. Preserve those observations with phase labels; never choose favorable repetitions after examining them. The benchmark configuration must identify which phase it compares. A compile-free first execution is not automatically labeled steady-state throughput.

Remove seed-number exclusions from both comparator data paths. Reject mixed/unknown timing schemas in performance comparisons, and remove fallback from execution to total/runner wall time under the same column name. Legacy artifacts remain readable and visibly marked `legacy/unknown`; they cannot silently pass as corrected timings. Emit descriptive summaries from eligible measurements with the selected estimator and sample counts recorded. Inferential speedup claims still depend on P4's statistical policies.

**Acceptance:** changing seed order or using seed 0 does not remove observations. All raw seeds/trials survive exports; invalid phases remain visible. Missing phase, bad preparation, unsupported adapter mode, and legacy timing produce explicit diagnostics. JSON/table/CSV consumers choose the same declared observations and estimator. Unit tests cover phase mismatch, legacy loading, nonfinite/negative durations, and exact round-trips. Existing timing tests are strengthened to assert boundaries, not merely field names or positive durations.

**Dependency:** PT1–PT3 interfaces; implement before the next cluster timing test. Full P9 manifests may extend this record later without blocking these minimum provenance fields.

### PT5 — Validate on the cluster with a timing-only workload

**Findings:** T01–T08; hardware confirmation of gate A.

**Changes:** add a small timing-validation entrypoint/configuration that exercises the actual native and upstream adapters without repeating all MuJoCo/RL datasets. Update the Slurm wrapper to forward quoted arguments correctly, require/use fresh output destinations, preserve Slurm's allocated GPU, and identify the exact source/import paths. Make the entrypoint report phase events, raw durations, validity, compilation logs, and output agreement under the supported seeded contract.

**Initial proposed design, frozen before observing corrected durations:** Dataset 5-like Rastrigin (`D=10`, `P=32`, `G=15`, 20 centroids), seeds `(42,43,44,45)`, one separately recorded first execution and five reset-state measured repetitions per seed/backend. Any additional preconditioning must be specified and recorded identically for both backends. Use a fixed alternating backend schedule with cache policy recorded; do not clear caches inside measured intervals. Use divisible toy/real island horizons until P6 is complete. Broader generation-count/device-scaling experiments follow only after this diagnostic passes.

**Acceptance:** cluster imports the intended branch revision, uses the assigned H100, reports the exact JAX/CUDA environment and timing-schema version, and shows no tracing/compilation inside the declared steady interval. No preparation/synchronization error is hidden. All seeds and first-use/repeated observations are present; output-equivalence checks compare the same seed within a backend's timed/untimed paths. Cross-backend bitwise trajectory equality is not assumed. Performance ratios are reported as observations even if unfavorable. Device contention and unavailable optional integration checks remain explicit limitations.

**Execution gate:** prepare and locally test the entrypoint, configuration, and Slurm command first. Then present the concrete job with one GPU, an initial 30-minute wall-time cap, output destination, and expected scope before requesting authorization to launch. This is a proposed resource cap, not a proven runtime estimate. No publication-sized or full RL-heavy rerun is part of this step.

**Dependency:** timing gate A and PT4. A successful PT5 establishes timing gate B and is logged with its new job ID/revision; job 1303 remains unchanged historical evidence.

## Original work packages and remaining scope

### P0 — Record the baseline and remediation ledger

**Findings:** A19; groundwork for all findings.

**Changes:** add a tracked ledger listing each finding, evidence status, implementation commit, regression test, affected outputs, and closure status. Record the two repository revisions and current environment. Inventory paper-generated outputs without modifying them. Document the audit environment separately from the intended publication environment.

**Acceptance:** every A01–A22 has an owner/status and a link to its evidence. Existing paper modifications remain intact. There is a clear location for corrected outputs and a baseline against which to compare them.

**Dependency:** none. This is the first implementation task.

### P1 — Stop alteration of measured results

**Findings:** A01 and the unsafe extraction portions of A03.

**Files:** `examples/generate_empirical_paper_curves.py`; focused generator tests under `tests/`.

**Changes:** remove the timing override; retain raw per-seed timings and signed objectives. Require expected fields instead of silently substituting zero fitness or invented execution times. Preserve instantaneous and best-so-far curves as distinct fields. Prevent automatic copying of regenerated outputs into the paper repository; provide an explicit export destination.

**Acceptance tests:** a measured native slowdown remains a slowdown in JSON, curves, and tables; the exported ratio is computed from recorded timing samples. Signed rewards `[-10,-2]` stay signed and their best-so-far curve is `[-10,-2]`. Missing timing/objective data produces a clear error or unavailable result. Smoke-mode fixtures remain visibly identified as fixtures and cannot masquerade as publication measurements.

**Dependency:** P0. Defer final QD-score definitions to P3; do not publish comparative QD-score conclusions until then.

### P2 — Make publication ingestion safe

**Findings:** A04 and publication-tooling parts of A22.

**Files:** paper `scripts/ingest_cluster_results.py`; new isolated tests/fixtures for that script.

**Changes:** make dry-run genuinely read-only. Missing results fail clearly. Move mock data to fixture-only paths. Require explicit input/output paths for real generation, and show which source artifacts will generate which outputs. Disable any path that currently presents placeholders as ingested measurements.

**Acceptance tests:** byte hashes of a temporary manuscript directory are unchanged after dry-run, missing-input errors, and malformed input. Fixtures never write into real table paths. Existing valid table files are preserved on failure. Genuine end-to-end table generation is implemented later in P10, after analysis contracts are settled.

**Dependency:** P0; can follow P1 without waiting for statistical redesign.

**Checkpoint 1 (implemented):** future runs cannot force favorable timing, destroy signed rewards, or overwrite manuscript tables with placeholders. This output-safety checkpoint does not certify the adapters' measurement boundaries; those are addressed by PT1–PT5.

### P3 — Define shared metric semantics

**Findings:** A03, A14, A20; relevant documentation in A22.

**Files:** metric catalog, native/universal/QD adapters, QD engine metrics, empirical extraction.

**Changes:** document internal lower-is-better objective values and externally reported values, with explicit orientation and units per metric. Reuse/extend existing metric metadata rather than inventing a parallel registry. Negate objective location metrics where required, never standard deviation. Compute archive mean/std over occupied cells, with an explicit empty-archive convention. Make native and upstream QD outputs comparable under a declared definition.

**Proposed QD default:** report coverage, best objective, and occupied-cell objective statistics. Export an aggregate score only with its exact sign/offset definition. Do not call an absolute sum of costs a higher-is-better quality score. Any publication offset needs a task-based justification agreed before rerunning.

**Acceptance tests:** minimization/maximization transformations preserve objective meaning; standard deviations remain nonnegative and invariant under sign changes; unoccupied archive cells do not bias occupied-cell statistics; empty archives are explicitly unavailable; both backend adapters agree on metrics for the same small archive.

**Dependency:** P1. Execute after the immediate timing batch so cluster comparisons use declared timing phases; do not change metric semantics within PT timing tests.

### P4 — Unify statistical validity and decisions

**Findings:** A02, A08, A10.

**Files:** `src/malthusjax/stats/{core,tests,comparator,io}.py`; generator statistical helpers; statistical exports and relevant documentation.

**Changes:** route paper analysis through shared statistical functions. Validate finite, aligned, one-dimensional samples and required sample counts. Represent invalid, insufficient, and valid results explicitly. Handle exact equality/constant differences deliberately rather than manufacturing p-values. Separate difference detection, directional improvement, and equivalence; expose raw inference and family-corrected inference separately. Displayed suite conclusions must use the selected multiplicity policy. Replace ambiguous pass/fail output with hypothesis-specific conclusions, retaining documented compatibility aliases only where necessary.

**Acceptance tests:** one observation cannot establish TOST equivalence; NaNs/exceptions cannot become p=1 evidence; exact-equality and constant-difference cases have documented outcomes; a significant result in the expected beneficial direction is interpreted correctly. Two raw equivalence p=.03 results adjusted to .06 cannot appear as corrected equivalence successes. JSON, Markdown, and tables agree. Underpowered non-rejection is not labeled equivalence.

**Scientific choices:** record the comparison family, target estimand, practical margin, alpha, and publication sample-size policy before inferential runs. A mechanically valid minimum of two observations is not a claim of adequate statistical power. Smoke runs produce descriptive outputs when inference is unavailable.

**Dependency:** P3 for metric orientation; P1/P2 remain useful independently.

### P5 — Repair mathematical and operator defects

**Findings:** A05, A13.

**Files:** MO sorting/population code, crossover base fast path, existing MO/crossover tests.

**Changes:** normalize crowding distances within each front/objective; define degenerate objective and invalid-input behavior. Fix crossover batch-size inference using parent arrays and validate compatible shapes/key budgets. Avoid unrelated operator redesign.

**Acceptance tests:** the audit crowding example yields interior distances 1.1 and 1.5; adding a dominated front cannot change an existing front's crowding order. Validate multiple fronts, singleton/two-member fronts, constant objectives, and duplicates against a small independent reference. Crossover works with default/inferred and configured input lengths, typed/legacy keys, eager/JIT calls, and rejects mismatches clearly.

**Dependency:** P0; execute after P4 to keep the sequence easy to follow.

### P6 — Repair island scheduling; adopt validated timing

**Findings:** A06, A07, A21.

**Files:** island base/adapter and island scheduling tests. Shared timing machinery, metadata, and seed selection are owned by PT1–PT4.

**Changes:** execute exactly the requested number of generations, using complete migration blocks plus a remainder. Proposed semantics: migrate at completed interval boundaries; a final partial block evolves without an extra unscheduled migration. Zero generations is a true no-op. Validate interval/island/migrant bounds. Derive reported generation/evaluation counts from actual execution. Integrate these scheduling changes with the prepared executable and measurement contract established in PT1–PT4; do not introduce a second island-specific timing protocol.

**Acceptance tests:** requests 0, 2, and 10 with interval 4 execute/report exactly 0, 2, and 10; migration occurs at the declared boundaries; divisible cases retain intended behavior. Changing the remainder/migration configuration prepares the appropriate executable before timing. PT timing regressions continue to pass. Validate any island performance claims on the actual corrected scheduling path, separately from PT5's native/QDax smoke comparison.

**Dependency:** PT1–PT4 and P3. Coordinate scheduling impact with P10; timing tests may use divisible horizons before this algorithmic repair is complete.

### P7 — Repair regression input and reporting contracts

**Findings:** A09, A15, A16.

**Files:** statistical dataset builder, regression fit/analyzer/results, benchmark analyzer.

**Changes:** validate unique observation identities before joining. Reject ambiguous duplicates by default; support explicit replicate IDs when there are genuine repeated observations. Enforce one-to-one pairing and report exclusions. Make transformations explicit: raw objective, optimality gap, and execution time are different dependent variables. Store optimum, epsilon, units, transform, model formula, covariance choice, and sample identifiers with the fit. Carry actual adjusted R², standard errors, and t statistics from the fitted model.

**Acceptance tests:** the four-row duplicate example raises a diagnostic rather than expanding to sixteen rows. Genuine replicate IDs remain distinct. Offset-objective gap transforms match independently calculated values. Adjusted R² matches the reference fitted model and differs from R² when appropriate. Reports identify missing inputs and model-rank problems.

**Scientific choice:** decide whether pair/seed/configuration dependence requires a paired or clustered model for each paper analysis. Do not assume HC robust covariance resolves correlated observations. Revalidate the actual H1 analysis provenance before replacing its methodology.

**Dependency:** P4.

**Checkpoint 2:** core mathematical regressions pass, island execution/timing contracts are correct, and shared analysis cannot silently create favorable or duplicated evidence.

### P8 — Make optimization API limits explicit

**Findings:** A11.

**Files:** `packages/malthus_optimize/` APIs, result/bridge, weighted least-squares evaluator, docs/tests.

**Changes:** inventory every advertised argument and classify it as implemented, explicitly unsupported, or requiring implementation. Reject unsupported nondefault requests instead of ignoring them. Do not claim convergence or feasibility solely because a fixed iteration budget ended. Define callback behavior and propagate callback errors. Narrow the SciPy compatibility claim to what is tested. Repair weighted covariance and `absolute_sigma` semantics for supported curve-fit cases; either implement covariance-matrix sigma and full-output compatibility correctly or reject/document them explicitly.

**Acceptance tests:** constraints/integrality/tolerances/initialization/strategy/polish controls are honored or clearly rejected; callback behavior matches the documented contract; stopping status is truthful. Test weighted linear fits/covariances against an independent analytic or SciPy reference. Include return-shape and finite-input checks. Do not implement the whole SciPy API merely to preserve an overbroad marketing claim.

**Dependency:** P3/P4 interfaces stable. This need not delay reanalysis of paper results that do not use this package.

### P9 — Add run validation and immutable manifests

**Findings:** A12, A19; extends PT4's early timing metadata and A21 repair.

**Files:** runner/results/IO, CLI and empirical entrypoints, publication environment specification.

**Changes:** use consistent structured run statuses; validate required metrics and completed horizons under metric-specific rules. Preserve failure reasons and paired attrition. Record schema version, exact source revisions/dirty state, configuration hash, seeds/run order, package versions, hardware/backend, precision and relevant XLA settings, plus input/output hashes. Pin a tested publication environment separately from supported-library dependency ranges. Do not blindly freeze the current system environment as the publication environment.

**Timeout policy:** first reject a supplied timeout if it cannot be enforced. Implement hard timeouts only at an appropriate subprocess/job boundary; avoid pretending a timed-out background device operation has stopped. This can be a separate follow-up commit with explicit cancellation tests.

**Acceptance tests:** NaN objective summaries cannot be successful unless the metric contract explicitly permits unavailable values and status reflects that. Serialization is standards-compliant and unavailable values are explicit. Duplicate seeds are detected or have distinct run IDs. A manifest round-trip identifies the exact inputs/environment and timing phases. Legacy artifacts load through a documented migration path and are marked when provenance is unknown.

**Dependency:** P3/P4/P6/P7 and PT4 for complete schemas. Minimum timing metadata is implemented earlier in PT4; it must not wait for this package. Baseline capture in P0 is complete.

### P10 — Trace and regenerate publication evidence

**Findings:** A01–A04, A07–A10, A15/A16, A19/A21, publication claims in A22.

**Changes:** build a per-table/figure/claim evidence matrix: raw sources, exact analysis command/revision, transformation, test family/margins, current provenance, applicable defects, and remediation action. Complete real ingestion/rendering into a separate output directory. Derive inferential text from structured results rather than hardcoded success narratives.

Classify each item as: reproducible without change; reanalysis required; experiment rerun required; or unavailable pending raw data. Start with control rewards, Dataset 5 QD metrics/statistics/timings, island scaling, and MO results if used. Audit H1/H2 analysis code and claimed bootstrap/multiplicity details against actual inputs.

**Acceptance:** every retained numerical paper claim has a replayable source chain. Regeneration from the same saved inputs is deterministic apart from explicitly excluded metadata. No claim of practical equivalence comes from non-rejection alone. Report measured slowdowns and uncertainty. A comparison report lists changed values and reasons. Existing manuscript edits remain untouched until corrected replacements are reviewed.

**Dependency:** P1–P7, PT1–PT5, and P9. GPU reruns need a concrete configuration, compute estimate/limits, output location, and user authorization before launch. Reuse valid raw outputs where reanalysis suffices. Timing claims from mixed-phase legacy artifacts require a new measurement; do not estimate and subtract compilation from their recorded execution durations.

### P11 — Packaging, CI, and documentation gates

**Findings:** A17, A18, A22; complete A19.

**Changes:** move required TensorNEAT transform implementation inside the distributed package while preserving optional dependency boundaries. Test built wheels outside the checkout. Verify Docker builds before fixing suspected failures; include required metadata files and a supported Python runtime. Expand CI coverage to statistical/benchmarking paths, root-level scientific tests, and shipped satellite packages. Include the main genetic engine in honest coverage reporting; keep hardware-specific suites separate. Gate the PT timing regressions in CPU CI, including actual first-run tracing and failure propagation. Lint the maintained empirical generator and changed satellite code without launching an unrelated formatting cleanup. Correct exact reproduction commands, inference descriptions, outdated island documentation, and tracing behavior.

**Acceptance:** isolated wheel import/backend smoke tests pass with declared extras, optional imports remain optional, Docker examples build in the supported environment, and documented smoke commands exist and run. CI catches the audit's reproduced defects. Coverage reports state their included/excluded scope. Complete the full CPU test suite in a suitable environment; GPU claims require their separate hardware checks.

**Dependency:** packaging fixes can start earlier, but final release gates cover all completed packages. No new coverage-percentage target is invented before measuring an honest baseline.

## Review and user decisions

Routine code choices, regression tests, and reversible local fixes can proceed within an approved implementation stage. At each checkpoint provide: changed files, fixed findings, tests/results, compatibility impact, unresolved questions, and affected experiments.

Scientific decisions need an explicit recorded rationale: practical equivalence margins and sample sizes, QD score definition/offset, regression estimand/dependence model, and any change to the paper's claim. Prepare a concrete recommendation before asking. Do not ask the user to approve individual code lines.

Separate approval is appropriate for expensive GPU jobs and publication/deployment actions outside the agreed implementation scope. Preparing code, configs, tests, and a reviewable output comparison precedes those requests.

## Completion criteria

- All A01–A22 and T01–T08 have a verified disposition: fixed and tested, disproved with evidence, or explicitly limited/deferred with no unsupported public promise.
- Scientific generators preserve observed measurements and objective meaning; invalid evidence cannot become a favorable claim.
- Algorithms and execution honor their documented mathematical/budget contracts.
- Statistical results and exported decisions use consistent, declared policies.
- Each paper result being retained has an auditable provenance chain, with unresolved items visibly excluded from inferential claims.
- The publication environment and commands can be replayed; installed distributions and CI exercise the maintained paths.
- Full CPU validation and required hardware validation are recorded separately; no partial run is described as a full pass.
- Timing gates A and B pass without excluding unfavorable seeds or requiring a speedup. Performance consumers use compatible declared phases and retain raw first-use/repeated measurements.

## Next implementation batch

P0–P2 and PT1–PT4 are implemented within their stated scopes, with separate implementation commits and regression evidence. PT4 records and CPU consumer validation are documented in `docs/audit_remediation/PT4.md`. CPU gate A passes for the audited paths. PT5's timing-only entrypoint and Slurm command are prepared and CPU-validated in `docs/audit_remediation/PT5.md`. Next obtain authorization for the concrete one-H100, 30-minute diagnostic and inspect its cluster artifacts; gate B remains open until actual hardware validation passes. Resume P3/P4, then the remaining original work packages. Publication-sized reruns remain gated on scientific validity and the P10 evidence matrix.
