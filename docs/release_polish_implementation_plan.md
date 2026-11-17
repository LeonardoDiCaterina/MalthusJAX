# MalthusJAX release polish implementation plan

The first milestone prepares the repository for a targeted research preview. The second prepares the native multi-objective and quality-diversity workflows for a self-serve launch. The release review and two independent routing trials identified the same practical gap. Researchers could reuse an unusual genome across engines, but they had to trace several interfaces to recover actual designs.

Work from the dedicated `codex/clean-install-smoke` worktree, based on `681ac82d`. The working tree contains earlier wheel, research workflow, and engine-output changes. Build and test the candidate from that state. Keep the publication runners, archived results, hashes, and provenance paths intact.

## Milestone 1. Make the research preview self-explanatory

### 1. Put the verified CPU path first

- Reorder the README so a direct description, installation, and the smallest runnable example appear before the four-level architecture. Use the existing `native_sphere` run with one seed and two generations. Give readers one next step based on the question they want to answer.
- Use the verified wheel build and install route. Link to a published installation channel only after verifying the candidate is available there. Put `make install-dev` in contributor setup and describe its integration dependencies there.
- Run the README, [quickstart](source/quickstart.md), and advertised introductory notebook against the same base wheel in an environment without EvoSAX. Change introductory Sphere snippets to `native_sphere`. Retain the explicit BBOB objective and dependency instructions for examples that intentionally use seeded BBOB.
- Move the optional-backend, statistical analysis, and reduced-output walkthroughs into linked workflow pages. Keep the quickstart short enough that a reader can follow it from top to bottom.

**Done when** all three entry points run outside the repository source path, import the installed wheel, and report successful runs and finite fitness. Execute the notebook top to bottom in a clean kernel and verify its earlier cells supply everything later cells need.

### 2. Repair links and verify the hosted navigation

- Replace the dangling `docs/source/extension_cli_guide.md` and `docs/source/toml_guide.md` symlinks with maintained pages or redirects. Preserve their public destinations where practical. Update the README, contributor guide, and Sphinx toctree together.
- Replace the missing `configs/benchmark.toml` reference in the Composer guide with a verified config. Make relative paths and the intended working directory explicit.
- Remove or regenerate stale API pages for `bbob_evaluator` and `real_evaluators`; update references to the existing modules where appropriate. Check the generated API navigation for duplicate and missing entries.
- Reorder the Sphinx entry points around installation, first run, research workflows, extension guides, and API reference. Run a full HTML build, triage existing warnings and suppressions, and fix release-facing toctree and cross-reference failures. Add a PR check for changed local links and docs build failures. Keep the Pages publishing trigger separate.

**Done when** the old guide links resolve, the rendered index reaches each supported entry point, and the PR gate catches a deliberately broken link. Record any remaining warnings with their exact source and scope. Verify hosted URLs after deployment when a preview is available.

### 3. Curate examples and scope the claims

- Index three maintained entry points. Start with base Sphere, follow with the corridor comparison, then show a compact permutation example with NSGA-II or native MAP-Elites. Put the required extras, exact command, expected output, and result path next to each entry.
- For the preview permutation example, use the working typed Level 3 route and show how to recover actual designs. Document `evaluate` and `evaluate_qd`, donation and copying for reused states or populations, and which options belong to the engine versus the runner. Verify this example from an installed wheel.
- Add a compact capability table for tested engine and genome combinations. Distinguish terminal scalar metrics, winning genomes, Pareto designs, and archives. Give the public access path and availability for each. Keep diagnostics, publication campaigns, and historical examples discoverable through labeled index entries. Inventory references before considering any relocation of root handoff or research scripts.
- Retain the measured benchmark table, denominators, intervals, and inconclusive outcomes. Scope statements about fusion, speed, strategy support, GPUs, and TPUs to the documented evidence. Move benchmark migration details to the research or release documentation.

**Done when** a reader can find a plain TOML experiment and a structured-genome experiment from the README without learning the entire architecture. Verify the index commands and the stated output paths. For structured designs, validate feasibility and rescore against an independent check.

## Milestone 2. Make the public implementation easier to follow

### 4. Clarify imports, construction, and plugin discovery

- Document `from malthusjax.composer import EngineFactory` as the canonical public import. Keep compatible re-exports for existing callers. Explain which registry owns operators, native engines, and backend providers. Use the canonical import in maintained examples and public guides.
- Split Composer's internal preparation from execution using a private typed result or callable. Keep the public `quick_run` signature and `ExperimentResult` return. Have comparison use the same preparation logic so invalid later pipelines still fail before optimization starts. Preserve explicit engine overrides, output options, shared initialization, execution order, and existing plugin contracts. Avoid a repository-wide rename or formatting pass.
- Explain that local discovery evaluates Python from the working directory's `plugins/` and describe the current module-stem naming behavior in the extension walkthrough. Document the trusted project directory assumption. Treat explicit loading and changes to module identity as separate compatibility work.

**Done when** existing and canonical imports resolve to the same factory, public `quick_run` returns the documented type, and existing quick-run, TOML, backend, and output contract tests pass. Include a regression that a bad final pipeline fails before any runner begins.

### 5. Resolve emitter configuration and preserve typed populations

- Add a host-side `genome_config` property to `MixingEmitter` in `operators/emitters/mixing.py`. Resolve the first non-None config from either child. When both children provide configs, validate compatibility at setup using the existing representation rules or give an actionable error. Update the native MAP-Elites adapter to check for a usable value instead of relying on `hasattr`; a mixture with no config must still reach the existing alternative initialization path.
- Give `build_mo_engine` an explicit typed genome-config path and support `genome_type="categorical"` when the full categorical configuration can be resolved. Preserve real and binary defaults. Validate the configuration, emitter, evaluator, and supplied initial population together.
- Remove the unconditional `RealPopulation.from_array(..., RealGenome)` conversion from supplied-population handling in both `GeneticEngineAdapter` and `MOEngineAdapter`. Accept a compatible typed population directly. For raw genes, construct the population through the configured representation and validate shape, dtype, and permutation requirements before execution. Prefer an existing typed constructor to generating a dummy random population. Preserve copying where caller-owned buffers need to survive a donated run.
- Keep ordered objective vectors through MO construction and execution. Document the existing representative scalar separately from the Pareto set.

**Done when** native MAP-Elites initializes a compatible `MixingEmitter` without a user subclass, including a mixture whose first child provides no config. Test incompatible and missing configurations. Verify generated and supplied categorical populations through the public MO factory and supplied permutations through native GA. Compare public and direct paths with matched input and cover existing real and binary routes.

### 6. Export native designs through ordinary results

- Gate native MAP-Elites solution export on `capture_solution`. Use the genome and stored score from the same best occupied cell, restore the run's objective direction, and label the solution as archive-best. Add an opt-in native MAP-Elites archive artifact using the existing output-aware runner and archive persistence path. Include occupied-cell IDs, descriptors, genotypes, centroids, and explicitly documented fitness orientation. Preserve the external QDax schema semantics and distinguish native representations where needed.
- Add `run_with_options(key, options)` to `IslandEngineAdapter` and retain the legacy `run_once(key)` entry point. On `capture_solution=True`, export the terminal island winner and its stored score in the run's objective direction. Label the scope as final-population best across islands. Keep the initial change focused on result extraction.
- Add an opt-in Pareto artifact for MO with each selected genome and its recorded objective vector, objective directions, and clear membership semantics. Define whether it exports the final rank-zero set or a maintained historical front. Start with the final rank-zero set. Document the duplicate policy explicitly and test repeated vectors. Make Pareto capture an explicit bulk-output request. A single `capture_solution` result requires an explicit selection rule and its full objective vector; the representative scalar alone cannot identify a Pareto winner.
- Represent supported, unavailable, and disabled outputs through the existing result contract. Keep files under the staged run root in CLI runs and serialize artifact references in seed JSON. Gate extraction before device transfer and skip optional extraction on timing repeats.
- Add acceptance tests against a small permutation oracle. Check feasibility, objective direction, seed-to-artifact references, occupied-cell selection, and Pareto nondominance. Verify that optional exports leave terminal metrics and evaluation counts intact. Exercise the default and reduced modes and confirm disabled hooks never run.

**Done when** a researcher can recover native Pareto routes, MAP-Elites designs, and the terminal island winner from a normal run without calling the engine adapter directly. Require this milestone before advertising end-to-end native MO and QD design export.

### 7. Give the island scan an explicit history mode

- Keep the current migration cadence, PRNG sequence, generations, remainder handling, and terminal state. Thread a static history mode through the island adapter's preparation and scan. When disabled, return an empty scan output such as `None`. When enabled, expose useful history through the output-aware entry point. Define how generation-by-island metrics become public rows and preserve the axes for direct low-level callers.
- Keep the history mode in the prepared-run cache key. Keep host-only solution and archive switches outside it. Preserve documented direct-call and legacy `run_once` behavior during migration.
- Compare full and reduced runs using identical seeds and initial state, including a horizon that is not divisible by the migration interval. Assert matching terminal state, metrics, and evaluation counts. Verify that reduced lowering omits the stacked history outputs. Inspect optimized code before claiming that particular KPI operations disappeared. Metrics used by state, migration, or telemetry can remain live. Benchmark synchronized prepared execution and memory before claiming a speedup.

### 8. Ship permutation Order Crossover

- Add `OrderCrossover` (OX1) for permutation-mode categorical genomes in `operators/crossover/` and register it in the crossover catalog. Use the research prototype as a starting point and follow existing operator contracts for static shapes, offspring count, and PRNG key budgeting. Reject incompatible genome configs with an actionable error.
- Compare deterministic parent/cut-point fixtures with an independent OX1 reference. Cover permutation validity, short genomes and cut boundaries, batched and JIT execution, catalog construction, and one maintained example that combines crossover with swap mutation. Check that variation preserves the intended representation in both native GA and MAP-Elites.
- Measure the cost of selecting OX1 separately from the default path. Its sorting and indexing execute inside the compiled operator. The scheduling pilot reported 3.99 ms with OX1 plus swaps and 2.30 ms with swaps alone on one CPU case. These observations describe the tested configurations, with no fusion or accelerator guarantee.

### Performance boundaries

Emitter config lookup and typed initial population handling occur during setup. Optional design and archive extraction occurs after synchronization and can incur device transfers and I/O when enabled. Disabled extraction should skip those steps, including on timing repeats. Keep these changes out of the generation loop. Island history suppression intentionally changes scan outputs; verify its own lowering and timings separately. Preserve the current optimizer kernels and donation policy for the other changes.

## Milestone 3. Ship an agent skill with the release

### 9. Package one skill and expose it through CLI and Python

Write the skill after the earlier release milestones are implemented and the
candidate wheel has passed its verification. Use the actual released APIs,
commands, and capability matrix as the sources for the guidance. The outline
below is a starting point for that final review.

- Keep one canonical UTF-8 `SKILL.md` within the package, with a short name
  and description in frontmatter. Include it in the wheel and source
  distribution. Load it through `importlib.resources` so retrieval works
  outside the checkout.
- Propose three read-only entry points for the same text. Use the existing
  console script spelling, `mjax`.
  - `mjax skill` writes the raw Markdown to stdout for piping or redirection.
  - `from malthusjax import skill; text = skill.read()` returns it as a string.
  - `python -m malthusjax.skill` prints the same Markdown when the console
    script is unavailable.
- Keep stdout free of status lines, color, or Markdown wrappers. Retrieval
  should require only the base package and leave experiment state untouched.
  Check the current eager Composer import and CLI plugin discovery before
  routing retrieval through them. Ensure the retrieval path never executes a
  working directory's local plugin. Keep ordinary Composer plugin discovery
  behavior covered by compatibility tests.
- Add quick discovery cues to `mjax --help`, the README, and the quickstart.
  Document `mjax skill > SKILL.md` for agents that ingest a file. Keep the core
  skill short enough to read in one call. Any companion material needed to
  complete an instruction must be packaged or reachable from public docs.
- Make the package smoke read the skill from a base wheel in a scratch
  directory. Assert CLI, module, and Python output match the bundled resource;
  verify help discovery and a local-plugin canary that would write a marker
  if imported. Also use the skill as the only starting guide for an agent to
  run the base example and recover one structured design with its documented
  extra. Compare the recorded output to the release capability matrix.

### Skill outline

1. **When to use MalthusJAX.** Describe the research questions it supports and
   how to choose between Composer, direct engines, and authoring an extension.
2. **Get a first result.** Check the installed version and environment, run the
   base CPU Sphere example, and verify success, finite fitness, and artifact
   location. Identify when a backend extra is required.
3. **Run a reproducible comparison.** Show the minimal Python and TOML routes,
   explain seed and initial-population policy, choose a fresh result directory,
   and inspect the manifest and seed records before analysis.
4. **Recover the result you need.** Show how to read a genome and its recorded
   score, a Pareto set, or a MAP-Elites archive through the supported interfaces.
   Explain solution scope, raw objective direction, and the distinction between
   history collection and export. Include a metrics-only recipe.
5. **Extend one component.** Use the maintained structured-genome example to
   show typed populations, permutation-safe operators, and a reusable scalar
   or QD evaluator. Cover plugin registration and the ownership of buffers when
   an initial population is reused.
6. **Diagnose common failures.** Cover missing extras, unsupported genome and
   adapter combinations, unavailable optional outputs, invalid pipeline
   parameters, and existing output destinations. Point agents to the relevant
   maintained examples and version-specific support notes.

**Done when** the release wheel exposes the same verified skill via CLI and
Python, the examples in it run against that wheel, and a fresh agent can use it
without access to the checkout or earlier reports. The final release report
should include the published skill's outline, retrieval commands, and the
observed checks.

## Landing order and release checks

1. Land navigation, README, and base-wheel first-run fixes together. Then curate the examples and verify rendered docs.
2. Land the Composer preparation and public-import cleanup as a separate internal change, with no public API break.
3. Prioritize two small boundary fixes. Land `MixingEmitter` configuration forwarding and native MAP-Elites output hooks first, then add island solution export. Follow with typed GA/MO initialization and opt-in Pareto export. Land the island history variant and OX1 separately. Update capabilities and examples as each ships.
4. Rebuild the wheel. Run the base install, notebook, maintained examples with their declared extras, and an outside-checkout rerun of the structured-genome example. Run the package and scientific checks, local link validation, and Sphinx build. Record wheel hash, revision, platform, and versions.
5. Have a fresh researcher follow only the public first-run and structured-design paths. For the preview, verify that the documented Level 3 path yields actual designs. For a wider launch, verify the factory and ordinary result APIs yield them. Report hosted docs and accelerator results only for environments actually exercised.

6. Write and package the skill against the verified release candidate. Rebuild and repeat the outside-checkout wheel checks, then report the final skill outline and retrieval commands alongside release verification.
