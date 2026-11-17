# Engine output options implementation plan

This plan makes optional result collection available to all MalthusJAX engine extensions through a common contract. It follows the researcher trials that surfaced stale results, missing winning genomes, and the need to inspect QD archives. It also addresses the costs introduced by collecting those outputs, especially device-to-host copies and keeping every pipeline's engine resident before a wide sweep.

Fix native solution provenance first. Then make expensive extraction optional while preserving the existing compiled generation loop. A later release can add a genuinely history-free loop for engines that can produce the required final metrics from bounded state. Result directory reservation, atomic CLI publication, seed membership validation, and the manifest remain active for every CLI run.

## 1. Current behavior and cost boundaries

| Area | Current path | Consequence |
| --- | --- | --- |
| Engine interface | [Engine protocol](../src/malthusjax/composer/engine_protocol.py) and [BenchmarkRunner](../src/malthusjax/benchmarking/runner.py) | The runner calls `run_once(key)`. Existing extensions can implement that signature on their own. |
| Population provenance | [Composer](../src/malthusjax/composer/composer.py) sets `record_initial_population` on capable native adapters. The [genetic adapter](../src/malthusjax/composer/engine_factory.py) hashes host copies of initial genes. | Comparative runs pay for a population transfer and hash while the initial state is still available. The compiled native run can donate the initial state, so extracting the fingerprint after the scan is unsafe. |
| Best genome | The native genetic adapter flattens and converts the best genome before returning. The runner accepts an explicit solution and falls back to the legacy `summary.best_solution`. | Filtering the value at serialization time cannot recover the cost of extracting it. Some algorithms still need to track the best genome for their own semantics. |
| QD archive | The [QDax adapter](../src/malthusjax/composer/qdax_adapter.py) attaches an extractor to the [universal adapter](../src/malthusjax/composer/adapters/base.py) when requested. The runner writes NPZ for the primary seed execution. | An enabled extractor can still run during timing repeats whose archive will be discarded. |
| History | `serialize_history=False` makes the runner drop rows after receiving them. The [artifact writer](../src/malthusjax/benchmarking/io.py) and [CLI](../src/malthusjax/benchmarking/cli.py) write history CSVs through separate paths. | Dropping retained rows saves result size. It leaves any scan history, formatting, or transfer already done by the engine in place. |
| Pipeline validation | Composer prepares every pipeline before executing any. | Configuration errors are found early, and a wide sweep can hold many prepared engine objects simultaneously. |

### Audit evidence

The macOS arm64 CPU audit reproduced a score mismatch in all three native Sphere runs with `TrackBest.LIGHT`, zero elitism, and strong mutation. For seed 1, the exported genome scored 3.0228884 while the best-ever scalar was 0.6449453. The corresponding `FULL` run exported a genome that matched its score. The audit also verified empty scan history and unchanged final state in a minimal `AbstractEngine` subclass, prepared-mode mismatch rejection, cached native wrapper reuse when toggling modes, and a single call to a legacy `run_once(key)` plugin. An ordered callback still ran with history disabled. GPU costs and optimized-code savings remain unmeasured.

### First fix · Pair each solution with its own score

Give a structured solution separate `representation`, `value`, `selection_scope`, and `fitness` fields. Source fitness from the same state and candidate index that selected the genome. Use the raw objective direction exposed by the engine. For stochastic objectives, preserve the score from the original evaluation. A fresh evaluation can observe a different outcome.

| Native tracking mode | Exported scope | Source of solution fitness |
| --- | --- | --- |
| `TrackBest.LIGHT` | `final_population_best` | Fitness at the selected index in the final population. The run's `best_fitness` remains the best-ever scalar. |
| `TrackBest.NONE` | `final_population_best` | Fitness at the selected index in the final population. |
| `TrackBest.FULL` | `best_seen` | Fitness retained with the tracked winning genome. |
| Legacy solution payload | `unknown` | Absent unless the producer provides a score for that candidate. |

Preserve the solution payload and raw score through comparison normalization and JSON round trips. Correct the native export before exposing more collection modes. Add a regression using Sphere with population 8, four generations, zero elitism, Gaussian mutation rate 1.0 and strength 3.0, and seeds 1, 2, and 3. Confirm the `LIGHT` genome matches its own score even when it differs from the run metric. Cover `FULL`, `NONE`, maximization, and legacy records. Keep the current `LIGHT` default.

## 2. Public options and defaults

Use two immutable option values, resolved once per effective pipeline. Put the engine value alongside the engine protocol and keep artifact options in the benchmarking layer. The names below are proposed public API names.

| Scope | Option | Default | Semantics |
| --- | --- | --- | --- |
| Engine | `capture_population_fingerprint` | `False` | Hash the starting genomes used by this execution, if the adapter can observe them. Always retain the inexpensive initialization policy and available seed metadata. |
| Engine | `capture_solution` | `True` | Return the engine's supported solution payload for the primary run. The adapter skips extraction when disabled. Core selection and best tracking follow the algorithm and its existing configuration. |
| Engine | `export_archive` | `False` | Request archive extraction for the primary run and persistence through the runner. Retain this name because the QDax pipeline already exposes it. |
| Engine | `collect_history` | `True` | Request per-generation collection. Expose `False` publicly when an end-to-end no-history implementation is available for the initial supported engines. A backend must declare whether it can honor the request. |
| Artifacts | `retain_history` | `True` | Retain rows in the returned result and seed JSON. Continue to accept existing `serialize_history=False` as an alias. |
| Artifacts | `write_history_csv` | `True` | Write per-pipeline and combined history CSVs. The CSV setting can be disabled while retaining rows in Python and JSON. For disabled retention, skip CSVs and record their effective setting. |

Keep Python and TOML spelling consistent for the new options. Prefer explicit typed option objects at the runner and engine boundaries and flat keys at Composer's existing Python and TOML configuration boundary. The proposed TOML looks like this:

```toml
[experiment.shared]
capture_population_fingerprint = false
capture_solution = false
retain_history = true
write_history_csv = false

[pipelines.layouts]
backend = "qdax"
export_archive = true
```

1. Parse the options in `ExperimentConfig.from_quick_run_kwargs` and expose the same fields through typed configuration. Normalize the legacy aliases in this parser. Preserve the effective options when the configuration is serialized.
2. Resolve defaults, then shared settings, then pipeline overrides. Apply alias normalization per scope before merging so a pipeline can override a shared setting using either spelling. If both spellings are supplied at one scope with different values, raise an error naming both fields.
3. Remove recognized options from backend passthrough kwargs. Forward one typed engine request to the runner, which is responsible for dispatch. Preserve direct adapter construction with the currently documented QDax flag through a compatibility translation.
4. Reject non-boolean values at the boundary. When archive export is explicitly requested through the runner, validate that a destination is available before starting. Under the CLI it is the staged result directory. Direct adapter calls can return an in-memory archive without involving the runner.
5. Record effective collection and artifact settings per pipeline. For a requested engine output that is unavailable, include an output-specific reason in the result. Record the requested setting, effective setting, and availability separately. For example, a legacy backend may return history despite a request to disable collection, while the runner still suppresses retention and export. Report engine-side collection as uncontrolled rather than assuming it was skipped. An intentionally disabled output does not need an error. Readers of older results should continue to accept absent metadata.

## 3. Extension contract and dispatch

Retain the legacy `Engine.run_once(key)` entry point. Introduce an explicit optional protocol for engines that accept output requests, for example:

```python
class OutputConfigurableEngine(Protocol):
    def run_with_options(
        self, key: chex.Array, options: EngineOutputOptions
    ) -> RunOutput: ...
```

Use an explicit method with a new name so legacy adapters with different `run_once` keyword arguments keep working. Detect and bind the method once for each engine in the runner. Keep the engine-level options immutable and pass them per execution. Two runners that reuse an engine should be able to request different outputs safely. The common path must avoid mutating an adapter's flags, probing arbitrary attributes for individual features, or catching `TypeError` to retry a backend after it has already begun executing.

### 3.1 Shared engine-side output handling

- Add named optional extractors at the adapter boundary for an initial population, a final solution, and a final archive. Use the existing `@adapter` and `UniversalAdapterEngine` for framework adapters. Let native adapters and independently implemented plugins call the same collection helpers or implement the optional protocol directly. Inheritance is unnecessary.
- Call the population extractor after the final injected or generated initial state has been installed and before any use that donates or invalidates it. A helper hashes the extracted genes only after the request has been checked and the extractor is available. Include a hash scheme identifier and deterministic leaf order, dtype, shape, and bytes. Document which representation was hashed. Keep the older field readable when loading previously generated results.
- Call solution and archive extractors after synchronized execution, with access to final state. Check the request before invoking the hook, flattening a genome, calling `device_get` or `tolist()`, or building archive arrays. Hooks can return `unavailable` when a configured tracking mode, algorithm, or state cannot produce the requested object.
- Keep the typed return additive. Define optional `solution`, `initialization`, `archive`, and per-output availability fields on `RunOutput`. Keep scalar metrics numeric and preserve the solution provenance fields defined above, for example the `encoded_genome` representation and `final_population_best` scope. Confine archive payloads to memory between the engine and runner, then store the artifact reference in `RunResult`.
- Preserve the existing `run_once(key)` behavior for direct calls. For an in-tree adapter that supports both entry points, share one internal execution method. This keeps direct callers and the runner on the same optimization path.

### 3.2 Legacy engines and adapter coverage

| Engine family | First migration |
| --- | --- |
| Native genetic adapter | Replace the mutable population flag with the initial-state hook. Gate flattening and conversion of the winning genome early. Preserve required best tracking and its current selection semantics. The first host guard skips flattening and transfer. Native post-scan winner gathering is a separate optimization that would need to preserve direct engine behavior and account for its current timing boundary. |
| Universal adapter and decorated extensions | Thread the request through the wrapper once. Accept optional named extractors from adapter implementations. Automatic host-side gating applies to every registered extractor. |
| QDax MAP-Elites | Register the existing archive extraction logic as a final-state hook. Preserve occupied cell indices, descriptors, genotypes, centroids, and the archive fitness convention. Keep unsupported archive representations visible through an availability reason. |
| EvoSAX, native MAP-Elites, MO, island, and other in-tree variants | Inventory their actual initial and final state access, return types, history dependencies, and every direct `run_once` caller. Adopt the common dispatch where state access is available. Advertise only the outputs each adapter can produce and identify the archive or Pareto format explicitly when needed. Include legacy EvoSAX and custom or stub engines in the inventory. |
| Existing third-party and registered engines | Call `run_once(key)` as before. The runner can retain or discard their returned fields and control writes. Record that engine-side collection was uncontrolled when a legacy engine emitted a disabled field. Preserve legacy `summary.best_solution` conversion when solution retention is enabled. |

For output-aware engines, disabled hooks must never be called. For legacy engines, the guarantee is limited to framework handling of returned data and output files. Capture the distinction in developer documentation and in the result metadata where a request could not be enforced.

## 4. Runner, exports, and timing

1. In `BenchmarkRunner.run`, derive a separate request for the primary seed execution and for timing repeats. Repeats use the same evolutionary configuration, horizon, PRNG sequence, and history mode needed for the existing agreement checks. Turn off optional fingerprint, solution, and archive extraction for repeats. Keep the configured objective and history comparisons intact. Exclude optional extraction metadata from those comparisons and assert that the primary and repeats use the same prepared execution signature.
2. Apply retention independently from writing. A no-CSV run should avoid constructing CSV rows or opening CSV files. Pass a dedicated history CSV setting to `write_experiment_artifacts` and apply the same setting in CLI combined export. Keep timing CSV and run integrity behavior independent. When a CLI comparison has mixed pipeline settings, include only the pipelines with history CSV enabled in the combined history and list them in the manifest.
3. Preserve strict JSON, legacy result loading, normalized comparison copies, and relative paths for CLI artifacts. For archive output, write to a sibling temporary file and replace the destination after successful serialization. Include the archive reference only after writing succeeds. Keep CLI artifacts under the staged root so failed publication cannot contaminate a previous run.
4. Record optional extraction and persistence separately from prepared execution. Keep capture and synchronization boundaries explicit. A fingerprint must be taken before donation, so measure its duration there and report it as an extra diagnostic phase. Document how it relates to the observed wall-clock warmup. Use performance-default requests with fingerprint and archive capture off for comparable new measurements. Keep existing versioned timing records and historical exports readable.
5. Ensure each result carries the effective request, reported availability, and timing metadata sufficient to distinguish a lightweight run from one that collected and exported diagnostics. A solution or archive request changes output handling and should leave fitness, evaluation counts, generations, and PRNG behavior unchanged.

## 5. Base scan, adapter summaries, and tracking capabilities

### 5.1 Native history suppression uses the existing base scan

1. Pass the resolved `collect_history` value as `return_history` to both native `prepare_run()` and `run()`. Use the base engine's existing cache key and prepared-mode check. Extensions calling the base scan inherit the empty stacked output. An extension overriding the loop must declare and implement support.
2. Skip history formatting entirely when no scan history was returned. Use a real terminal summary. Native GA already has a terminal best score and completed horizon. Mark AUC unavailable if no equivalent accumulator exists. For native MAP-Elites, derive terminal archive metrics from its final repertoire using existing occupancy and sign rules. For MO, preserve the existing scalar summary convention and Pareto count. Report optional metrics as unavailable if their only source was history. Validate settings such as `use_history_for_final` before execution when they require retained rows.
3. Confirm identical final state in a minimal custom subclass with history on and off. Test true → false → true, cached wrapper identity, and mismatched prepared-run rejection. Verify empty history contains zero rows, including zero generation-only rows. Test native adapters independently so a correct base scan cannot conceal a bad adapter summary.

### 5.2 Universal and other custom loops need terminal metrics first

1. Remove the universal wrapper's finite-zero fallback for missing required `best_fitness`. A missing or nonfinite endpoint must fail the run. Confirm the failure with a test that deliberately omits it.
2. Before enabling history suppression for universal EvoSAX and QDax, define a trustworthy terminal source per adapter. Carry a bounded record of the last normalized metrics through both JIT and Python loops, or extract an equivalent terminal summary from the final state. Preserve direction, evaluation counts, and completed horizon. Verify the terminal result with distinguishable starting, first-generation, and final objectives.
3. Select the JIT variant statically and include history mode in the universal cache key. The current wrapper cache keeps one entry. Test correct specialization independently of reuse when switching back, and add bounded per-variant caching if returning to an earlier mode needs to avoid rebuilding its wrapper. Keep the host-only extraction flags outside this key.
4. Inventory islands and other separately implemented loops. Enable no-history mode only when each loop can honor the request and return a valid terminal summary. Reject a no-history request for an output-aware engine without the necessary terminal capability. Keep legacy plugins callable and report collection as uncontrolled.

### 5.3 Metric reductions and best tracking are independent capabilities

The base scan guarantees suppression of stacked outputs. Each `step()` still runs, and logging, callbacks, and algorithmic state can retain work. Keep optimizer semantics intact when dropping reports.

- Preserve `TrackBest` as a scalar Hall-of-Fame capability. QD and multi-objective engines retain their archive or Pareto semantics. Avoid adding a common ranking solely to make their modes look like native GA's.
- Reuse native GA's `TrackMetrics.NONE/BASIC/ALL` where that enum's mean and standard-deviation semantics apply. Wire the typed setting through the GA factory explicitly. The existing base boolean has different behavior. Avoid implicit integer coercion, which would interpret `True` as `BASIC`. Map legacy settings deliberately per engine.
- Preserve a valid terminal best when disabling optional MAP-Elites reductions. Its current false branch can produce a NaN best and still gather the winner. Keep required archive, migration, and Pareto operations active. Report intentionally suppressed optional metrics as unavailable, including dummy zeros from low-level GA modes.
- For each claimed saving, inspect lowered output shape or optimized HLO for the selected mode, with logging on and off as relevant. Lower the chosen native kernel directly or extend the helper to select its mode. Measure synchronized runtime separately.

## 6. Bound preflight memory without delaying errors

Preflight and artifact options solve different problems. Keep validation before the first optimizer run.

1. Inventory what the eager prepare step validates today. Include catalog parameter names, evaluator resolution, genome bounds and shapes, strategy parameters, dependency availability, and provider-specific build checks. Identify which checks instantiate heavy backends or allocate device arrays.
2. Split resolution and validation from engine construction where the provider supports it. Reuse the same validators in the eventual construction call. Have preflight produce normalized, lightweight pipeline configurations without compiling or allocating large populations. Generate shared initial populations consistently from the effective request and seed when the corresponding engine is constructed.
3. For providers whose validation currently requires construction, keep eager preparation until they expose an equivalent validation path. Report their preparation behavior. Avoid replacing a known early failure with a late failure silently. Keep the memory improvement separate from landing the output switches.
4. Verify that a malformed final pipeline still raises before any optimizer run and that fully migrated providers hold at most one constructed engine at a time. Use a fake provider with construction and lifetime counters for the latter, then measure peak host and device memory on a representative wide sweep.

## 7. Landing order and verification

### Change 0 — Correct solution provenance

Land the native score/scope pairing and legacy semantics first. Use the audit's no-elitism regression and exercise the JSON and normalized comparison paths. Review the current CLI solution examples against the new fields.

### Change 1 — Define and route option values

Add typed requests, config parsing, alias handling, optional protocol dispatch, and result metadata. Keep existing engine and file behavior during the migration. Contract tests cover a minimal legacy `run_once(key)` plugin, a new-style plugin, forwarding through Composer and typed configuration, mixed per-pipeline TOML settings, and old JSON files. Verify the engine is called once if it raises an internal `TypeError`.

### Change 2 — Gate extraction and export at the source

Migrate native GA, the universal wrapper, and QDax. Set population fingerprints to opt-in and keep archive export opt-in. Enable the winning-genome and CSV switches. Cover enabled hooks, disabled hooks that raise if invoked, unsupported requests with a reason, and reuse of one engine with different requests. Assert disabled cases skip host transfer or serialization at the targeted call site. Verify timing repeats invoke zero optional extractors and that disabling history CSV leaves seed JSON, summaries, and manifest validation working.

Rebuild the wheel and run the native sensor example in normal and reduced-output modes. Check identical final metrics, evaluation counts, and histories when collection remains enabled. In normal mode, decode and rescore the selected genome from the CLI record. Exercise EvoSAX and QDax with their product extras and verify QDax produces a readable archive only on request. Verify the five-to-one-seed rerun, concurrent same-destination rejection, and offline stats workflow against the same wheel.

### Change 3 — Wire existing native history suppression and profile

Enable the native path using the existing base-engine variant and adapter-specific summaries. Include the minimal subclass and prepared-mode regressions. Measure before changing the universal scan or claiming that optional metric operations have been compiled away.

Compare the current wheel and the proposed wheel on matched seed blocks, with the same supported dependency versions and actual population and generation budgets. Include small and large native populations, a long-history workload, a QDax archive workload, and a wide preflight workload. Run CPU first and a representative GPU separately. Toggle each optional output in isolation.

Report cold preparation, synchronized prepared execution, extraction, artifact writing, total wall time, output bytes, and peak host and device memory where measurable. Record hardware, software, precision, seed policy, and synchronization points. Inspect lowered output shapes for any compiled no-history variant. Use measurements to set regression budgets and distinguish observed results from hypotheses. Historical campaign artifacts and their timings remain as collected.

### Change 4 — Add universal history suppression and metric capabilities

Land universal history suppression only after replacing the missing-score fallback and verifying real terminal metrics in both loop modes. Migrate static metric capabilities engine by engine. Document their limits and test an installed third-party-style plugin for each new contract.

### Change 5 — Reduce preflight memory by provider

Land lightweight validation by provider, with early-validation coverage, a fixed-population reproducibility check, and a peak-memory check. Keep the eager path where validation still needs backend construction.

Update the [quickstart](source/quickstart.md), [Composer orchestration guide](source/5-COMPOSER_EXPERIMENT_ORCHESTRATION.md), and [benchmarking API README](../src/malthusjax/benchmarking/README.md) as each capability ships. Show the default researcher workflow, a metrics-only example, the difference between retaining and collecting history, and one minimal extension with an optional extractor. A release is ready when those examples run against the built wheel outside the source checkout and the verification record identifies the adapters and platforms exercised.

## Implementation verification to date

The implementation keeps `TrackBest.LIGHT` and full history as defaults.
Population fingerprints and QDax archive export are opt-in. Native GA, native
MAP-Elites, and MO route history requests through the base scan. Universal
EvoSAX and QDax runs use a bounded terminal metric record when history is off.
Legacy engines continue through `run_once(key)`. Eager provider preflight
remains in place where validation needs construction. Completed pipelines now
release their prepared callables before the next execution.

The local CPU pilot compared the prior and revised wheels on macOS arm64 with
Python 3.12 and JAX 0.12.0.1. Each case used six matched native Sphere seeds.
The table gives medians over seeds 2–6 in one process per mode, in milliseconds.
Runner duration includes initialization and result extraction. The pilot
disabled file writes, so these timings exclude CSV and archive persistence.

| Population × dimensions × generations | Prepared execution, prior / default / reduced | Runner duration, prior / default / reduced |
| --- | --- | --- |
| 32 × 8 × 40 | 0.94 / 0.99 / 0.96 | 7.32 / 7.37 / 2.76 |
| 512 × 128 × 60 | 91.25 / 91.12 / 90.65 | 100.58 / 99.81 / 92.82 |
| 64 × 16 × 250 | 7.42 / 7.52 / 7.42 | 36.02 / 35.40 / 9.33 |

Fitness matched exactly across all three modes. Reduced mode disabled history
collection and solution extraction. The pilot does not establish a regression
budget, a GPU speedup, or a reduction in peak preflight memory. Provider-specific
lightweight validation needs equivalent early-error checks and memory profiling
before replacing eager preparation.

### Ongoing supervision

The scientific CPU workflow runs the cross-backend output contract, including
both objective directions and the universal JIT and Python loops. Package CI
runs default and reduced native configurations from the installed wheel
outside the checkout. It checks terminal metrics and the artifact set.
Hardware timing uses a separate, threshold-free benchmark.

Run the same script with each wheel environment's interpreter from a scratch
working directory. Use an absolute script path and a fresh process for each
mode. The following commands run a matched comparison on a supported GPU.

```bash
JAX_PLATFORMS=cuda "$wheel_bin/python" -I "$repo/scripts/benchmark_engine_outputs.py" \
    --mode default --population 512 --dimensions 128 --generations 60 > default.json
JAX_PLATFORMS=cuda "$wheel_bin/python" -I "$repo/scripts/benchmark_engine_outputs.py" \
    --mode no-history --population 512 --dimensions 128 --generations 60 > no-history.json
```

Use `JAX_PLATFORMS=cpu` for host-only runs. Compare a prior wheel and the revised
wheel in default mode first. Alternate process order across repetitions and
validate the seed objectives and evaluation counts before comparing times.
The script reports cold and subsequent per-seed timings, runner wall time,
standard export time, output size, and process high-water RSS. Device memory
is reported when the backend supplies it. Archive writing happens during the
runner phase and is included in runner wall time. Compare archive runs against
QDax defaults with the same workload.
