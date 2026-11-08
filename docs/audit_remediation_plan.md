# Audit remediation implementation plan

Prepared 8 November 2026. Baseline: MalthusJAX `51009b8cf7464fe4a56bc40c472bfe8377ed9858`.

## Objective and scope

Make MalthusJAX's execution, analysis, and publication pipeline trustworthy and reproducible, then determine which existing paper results need correction. This plan covers findings A01–A22 from `audit/2026-11-08/REPORT.md`. Implementation progress and verified dispositions are recorded in [the remediation ledger](audit_remediation/findings.md).

The first milestone is safe evidence generation. The next milestone is correct algorithms and inference. Publication reruns follow an explicit impact assessment. General cleanup and broader refactoring are deferred until those milestones pass.

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

## Ordered work packages

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

**Checkpoint 1:** future runs cannot force favorable timing, destroy signed rewards, or overwrite manuscript tables with placeholders.

### P3 — Define shared metric semantics

**Findings:** A03, A14, A20; relevant documentation in A22.

**Files:** metric catalog, native/universal/QD adapters, QD engine metrics, empirical extraction.

**Changes:** document internal lower-is-better objective values and externally reported values, with explicit orientation and units per metric. Reuse/extend existing metric metadata rather than inventing a parallel registry. Negate objective location metrics where required, never standard deviation. Compute archive mean/std over occupied cells, with an explicit empty-archive convention. Make native and upstream QD outputs comparable under a declared definition.

**Proposed QD default:** report coverage, best objective, and occupied-cell objective statistics. Export an aggregate score only with its exact sign/offset definition. Do not call an absolute sum of costs a higher-is-better quality score. Any publication offset needs a task-based justification agreed before rerunning.

**Acceptance tests:** minimization/maximization transformations preserve objective meaning; standard deviations remain nonnegative and invariant under sign changes; unoccupied archive cells do not bias occupied-cell statistics; empty archives are explicitly unavailable; both backend adapters agree on metrics for the same small archive.

**Dependency:** P1.

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

### P6 — Repair island scheduling and measurement

**Findings:** A06, A07, A21.

**Files:** island base/adapter, timing metadata and comparator timing selection, island tests.

**Changes:** execute exactly the requested number of generations, using complete migration blocks plus a remainder. Proposed semantics: migrate at completed interval boundaries; a final partial block evolves without an extra unscheduled migration. Zero generations is a true no-op. Validate interval/island/migrant bounds. Derive reported generation/evaluation counts from actual execution. Use a cached compiled execution function, explicit initialization/compilation/execution buckets, and synchronize returned arrays before ending the execution timer. Identify cold/warm measurements through metadata rather than seed number.

**Acceptance tests:** requests 0, 2, and 10 with interval 4 execute/report exactly 0, 2, and 10; migration occurs at the declared boundaries; divisible cases retain intended behavior. Metadata identifies cold/warm phases with seeds ordered `(44,42,43)`. CPU tests verify synchronization/order contracts; actual accelerator throughput is validated separately on the intended hardware. Do not use brittle wall-clock thresholds in unit tests.

**Dependency:** P3; coordinate schema changes with P4/P9.

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

**Findings:** A12, A19; integration with A21.

**Files:** runner/results/IO, CLI and empirical entrypoints, publication environment specification.

**Changes:** use consistent structured run statuses; validate required metrics and completed horizons under metric-specific rules. Preserve failure reasons and paired attrition. Record schema version, exact source revisions/dirty state, configuration hash, seeds/run order, package versions, hardware/backend, precision and relevant XLA settings, plus input/output hashes. Pin a tested publication environment separately from supported-library dependency ranges. Do not blindly freeze the current system environment as the publication environment.

**Timeout policy:** first reject a supplied timeout if it cannot be enforced. Implement hard timeouts only at an appropriate subprocess/job boundary; avoid pretending a timed-out background device operation has stopped. This can be a separate follow-up commit with explicit cancellation tests.

**Acceptance tests:** NaN objective summaries cannot be successful unless the metric contract explicitly permits unavailable values and status reflects that. Serialization is standards-compliant and unavailable values are explicit. Duplicate seeds are detected or have distinct run IDs. A manifest round-trip identifies the exact inputs/environment and timing phases. Legacy artifacts load through a documented migration path and are marked when provenance is unknown.

**Dependency:** P3/P4/P6/P7 for final schemas; baseline capture in P0 starts immediately.

### P10 — Trace and regenerate publication evidence

**Findings:** A01–A04, A07–A10, A15/A16, A19/A21, publication claims in A22.

**Changes:** build a per-table/figure/claim evidence matrix: raw sources, exact analysis command/revision, transformation, test family/margins, current provenance, applicable defects, and remediation action. Complete real ingestion/rendering into a separate output directory. Derive inferential text from structured results rather than hardcoded success narratives.

Classify each item as: reproducible without change; reanalysis required; experiment rerun required; or unavailable pending raw data. Start with control rewards, Dataset 5 QD metrics/statistics/timings, island scaling, and MO results if used. Audit H1/H2 analysis code and claimed bootstrap/multiplicity details against actual inputs.

**Acceptance:** every retained numerical paper claim has a replayable source chain. Regeneration from the same saved inputs is deterministic apart from explicitly excluded metadata. No claim of practical equivalence comes from non-rejection alone. Report measured slowdowns and uncertainty. A comparison report lists changed values and reasons. Existing manuscript edits remain untouched until corrected replacements are reviewed.

**Dependency:** P1–P7 and P9. GPU reruns need a concrete configuration, compute estimate/limits, output location, and user authorization before launch. Reuse valid raw outputs where reanalysis suffices.

### P11 — Packaging, CI, and documentation gates

**Findings:** A17, A18, A22; complete A19.

**Changes:** move required TensorNEAT transform implementation inside the distributed package while preserving optional dependency boundaries. Test built wheels outside the checkout. Verify Docker builds before fixing suspected failures; include required metadata files and a supported Python runtime. Expand CI coverage to statistical/benchmarking paths, root-level scientific tests, and shipped satellite packages. Include the main genetic engine in honest coverage reporting; keep hardware-specific suites separate. Lint the maintained empirical generator and changed satellite code without launching an unrelated formatting cleanup. Correct exact reproduction commands, inference descriptions, outdated island documentation, and tracing behavior.

**Acceptance:** isolated wheel import/backend smoke tests pass with declared extras, optional imports remain optional, Docker examples build in the supported environment, and documented smoke commands exist and run. CI catches the audit's reproduced defects. Coverage reports state their included/excluded scope. Complete the full CPU test suite in a suitable environment; GPU claims require their separate hardware checks.

**Dependency:** packaging fixes can start earlier, but final release gates cover all completed packages. No new coverage-percentage target is invented before measuring an honest baseline.

## Review and user decisions

Routine code choices, regression tests, and reversible local fixes can proceed within an approved implementation stage. At each checkpoint provide: changed files, fixed findings, tests/results, compatibility impact, unresolved questions, and affected experiments.

Scientific decisions need an explicit recorded rationale: practical equivalence margins and sample sizes, QD score definition/offset, regression estimand/dependence model, and any change to the paper's claim. Prepare a concrete recommendation before asking. Do not ask the user to approve individual code lines.

Separate approval is appropriate for expensive GPU jobs and publication/deployment actions outside the agreed implementation scope. Preparing code, configs, tests, and a reviewable output comparison precedes those requests.

## Completion criteria

- All A01–A22 have a verified disposition: fixed and tested, disproved with evidence, or explicitly limited/deferred with no unsupported public promise.
- Scientific generators preserve observed measurements and objective meaning; invalid evidence cannot become a favorable claim.
- Algorithms and execution honor their documented mathematical/budget contracts.
- Statistical results and exported decisions use consistent, declared policies.
- Each paper result being retained has an auditable provenance chain, with unresolved items visibly excluded from inferential claims.
- The publication environment and commands can be replayed; installed distributions and CI exercise the maintained paths.
- Full CPU validation and required hardware validation are recorded separately; no partial run is described as a full pass.

## Suggested first implementation batch

Implement P0, P1, and P2, then report Checkpoint 1. This establishes the tracked ledger and removes the most immediate evidence-generation hazards without waiting for broader API or model-design work. Next tackle P3/P4, followed by P5–P7. Plan expensive reruns only after the P10 evidence matrix identifies what is actually necessary.
