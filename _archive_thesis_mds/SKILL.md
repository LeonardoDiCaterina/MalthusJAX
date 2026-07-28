# MalthusJAX Parity Skill

This skill captures the project intent and the exact parity rules for generating TOML experiments that compare MalthusJAX against an `evosax` baseline.

## Skill intent

- Help authors generate parity-safe TOML configs.
- Document the backend semantics that differ between MalthusJAX and `evosax`.
- Prevent the class of bugs we saw earlier by making alignment rules explicit.
- Support both experiment authors and developers maintaining parity test workflows.

## Who this helps

- Experiment authors writing TOML for MalthusJAX or parity comparisons.
- Developers debugging why a `Composer.compare(...)` run disagrees with a toy reference.
- QA reviewers checking whether a TOML config is parity-safe.

## Core parity contract

These parameters must be aligned exactly across both backends when comparing results:

1. Shared initial population
   - Use the same population construction seed for both pipelines.
   - In Composer, enable `shared_initial_population=True` and pass `pop_seed` explicitly.
   - Do not allow one backend to generate its own seed independently.

2. RNG seeding and determinism
   - Use the same seed for initial population and any backend-specific RNG.
   - Document which seed is used for population initialization and which is used for operator randomness.

3. Operator semantics
   - `MalthusJAX` `elite_pool` is not the same as `evosax` `SimpleGA` by default.
   - Explicitly map:
     - elite preservation
     - pool size and selection pressure
     - injection/replacement behavior

4. `num_offspring`
   - Align offspring count between backends.
   - This changes effective selection pressure and replacement volume.
   - Document whether each parent pair produces 1 offspring, 2 offspring, or more.

5. Mutation schedule and strength
   - Use the same mutation schedule semantics on both sides.
   - For fair comparisons, prefer a constant mutation schedule like `optax.constant_schedule(...)`.
   - Ensure `mutation_strength` means the same thing in both configs.

6. Crossover and selection mapping
   - Keep crossover rate, type, and parent pairing semantics aligned.
   - Keep selection pressure aligned via `num_selections`, `elite_k`, tournament size, or equivalent.
   - Confirm whether the backend uses full replacement or partial injection.

7. Output metric alignment
   - Use exactly the same metric to compare pipelines.
   - For Sphere experiments, gap-to-optimum should be computed consistently, e.g. `abs(best_fitness)` or a canonical `gap_to_optimum` field.
   - Don’t compare raw values from different objective conventions without explicit conversion.

## Backend alignment notes

### MalthusJAX

- Uses `Composer`, pluggable operators, and a JAX-native engine.
- `num_offspring` is a first-class operator parameter.
- `elite_pool` selection can preserve elites and choose parents from a candidate pool.
- Mutation and crossover wrappers must be configured explicitly for parity.

### evosax

- Uses `SimpleGA` or another high-level strategy.
- The default replacement policy may differ from MalthusJAX.
- `num_offspring` is implied by how many offspring are produced and whether the population is fully replaced.
- The schedule for mutation strength may be passed through `optax`.

### Parity hazards to avoid

- Different offspring counts per generation.
- Different elite pool sizing rules.
- One backend using a constant schedule while the other uses adaptive or default scheduling.
- One backend generating its own initial population or using a different seed.
- Different output targets (`best_fitness` vs `gap` vs signed objective).
- TOML fields being overridden unexpectedly by pipeline-level defaults.

## TOML grammar and capabilities

The official TOML grammar guide is `docs/source/toml_grammar_guide.md`. Use that file as the authoritative reference for all supported config structure and syntax in this repo.

It documents:

- top-level `[experiment]` metadata and optional fields like `output_dir`, `description`, and `name`
- a shared configuration section `[experiment.shared]` for default settings inherited by all pipelines
- pipeline sections under `[pipelines.NAME]` for algorithm-specific overrides
- optional data source sections under `[data.ID]` for complex problem definitions like knapsack or TSP
- operator string grammar for `fitness`, `selection`, `crossover`, and `mutation`
- boolean, integer, float, array, and string value syntax
- best practices for naming, descriptions, and version-controlled experiment configs

Use this guide when answering questions about the correct way to address TOML in MalthusJAX and when generating new config templates.

## Canonical TOML patterns

### Shared experiment metadata

```toml
[experiment]
name       = "sphere_parity"
output_dir = "results/parity"

[experiment.shared]
fitness       = "sphere:dim=5"
genome_type   = "real"
genome_length = 5
bounds        = [-5.0, 5.0]
pop_size      = 100
generations   = 100
seeds         = [0, 1, 2]
```

### MalthusJAX pipeline

```toml
[pipelines.mj]
backend      = "malthusjax"
selection    = "elite_pool:num_selections=100,elite_k=50"
crossover    = "uniform:crossover_rate=0.5,num_offspring=2"
mutation     = "gaussian:mutation_rate=0.1,mutation_strength=0.05,num_offspring=1"
injection_mode = true
```

### evosax pipeline

```toml
[pipelines.evosax]
backend            = "evosax"
evosax_strategy    = "SimpleGA"
selection          = "elite_pool:num_selections=100,elite_k=50"
crossover          = "uniform:crossover_rate=0.5,num_offspring=2"
mutation           = "gaussian:mutation_rate=0.1,mutation_strength=0.05,num_offspring=1"
mutation_schedule  = "constant"
injection_mode      = true
```

### Compare-run wrapper

```toml
[pipelines.compare]
backend = "compare"
shared_initial_population = true
pop_seed = 0
```

Use `scripts/run_composer_shared.py` or `Composer.compare(...)` to enforce shared population seeding.

## Skill checklist for authors

- [ ] Shared experiment metadata is identical across pipelines.
- [ ] `pop_seed` / shared initial population is explicit.
- [ ] `num_offspring` is aligned.
- [ ] Elite preservation parameters are aligned.
- [ ] Mutation schedule semantics are identical.
- [ ] Crossover rate and type are aligned.
- [ ] The output metric is normalized before comparison.
- [ ] No pipeline overrides silently change the shared fields.

## Safe patterns vs anti-patterns

### Safe pattern

- Explicitly define every parity-relevant parameter.
- Use `Composer.compare(...)` for shared population comparisons.
- Use stable schedule definitions, not defaults.
- Convert both backends to the same gap metric before comparing.

### Anti-patterns

- Using `evosax` and MalthusJAX pipelines with different offspring/replacement semantics.
- Relying on one backend’s defaults for mutation schedule.
- Comparing raw `best_fitness` values from different objective conventions.
- Allowing one pipeline to generate its own initial population.

## Key files for this skill

- `scripts/run_composer_shared.py` — TOML runner that forces `shared_initial_population=True` and passes `pop_seed`.
- `scripts/run_programmatic_parity_sweep.py` — programmatic workflow for multi-seed parity comparison.
- `examples/toy_gap_convergence.py` — reference parity toy experiment.
- `tests/composer/test_integration_initial_population.py` — integration tests for shared population behavior.
- `src/malthusjax/operators/selection/elite_pool.py` — elite selection semantics and pool sizing.

## How to use this skill

- When asked for TOML generation, produce a template that includes shared metadata, explicit pipeline mappings, and a compare wrapper.
- When asked about parity failures, trace the issue to one of the checklist items.
- When asked about backend differences, explain how MalthusJAX operator arguments map to `evosax` strategy settings.
- When asked about a correct comparison, recommend `Composer.compare(...)` and gap normalization as the source of truth.
