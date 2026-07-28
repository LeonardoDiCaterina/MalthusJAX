# Experiment Orchestration via Composer

This section covers the `malthusjax.composer` module, which provides a high-level, config-driven interface for running evolutionary experiments. Rather than manually instantiating operators, engines, and fitness functions, users can describe experiments via string specifications or TOML files and let Composer resolve the details. This section bridges the gap between low-level operator/engine APIs and user-facing experiment workflows.

## 5.1. The Composer Pattern: Config-Driven GA Runs

Composer abstracts the complexity of engine construction, operator composition, and fitness evaluation behind a simple `quick_run()` method. It supports multiple backends (MalthusJAX native, QDAX, evosax, TensorNEAT) and provides sensible defaults for population size, generations, and mutation schedules.

### 5.1.1. The Composer Class and quick_run() Entry Point

**Concept:** Providing a minimal, product-first interface for launching experiments.

**Knowledge Point:** Core method signature, parameter resolution, backend dispatch, and integration with BenchmarkRunner.

Prose to be written:
- Role of Composer as the top-level orchestrator
- `quick_run()` as the main user-facing API
- Support for external adapters alongside native engines
- How Composer resolves operator specs → operator instances
- Integration with BenchmarkRunner for multi-seed experiments
- Return value (ExperimentResult) and its contents
- Example: basic `quick_run()` call with minimal arguments
- Example: specifying custom operators and fitness functions via strings
- Fallback to StubEngine when operators are not provided

### 5.1.2. Default Configurations and Parameter Inheritance

**Concept:** Establishing sensible defaults and allowing hierarchical parameter override.

**Knowledge Point:** Default population size, generations, mutation rates; how --shared-- configs in TOML are inherited by individual pipelines; override precedence (explicit > pipeline > shared > global defaults).

Prose to be written:
- Rationale for defaults (small pop sizes for fast iteration, standard mutation rates)
- Multi-level configuration hierarchy (global, shared, pipeline, inline)
- Merging strategy and precedence rules
- How to override defaults at different levels
- Example: TOML with shared section and pipeline overrides

### 5.1.3. Multi-Seed Experiment Orchestration

**Concept:** Running the same algorithm configuration across multiple random seeds to gather statistics.

**Knowledge Point:** Seed sequences, parallel/sequential execution modes, aggregation of results into ExperimentResult.

Prose to be written:
- Role of BenchmarkRunner in multi-seed loops
- Reproducibility guarantees (same seed → same trajectory)
- Aggregation of per-seed histories and fitness traces
- Progress tracking (tqdm integration)
- Example: running 10 seeds and analyzing fitness convergence

## 5.2. Operator Catalog: String Specification Resolution

The OperatorCatalog maps string specifications (e.g., `"tournament:tournament_size=4"`) to operator instances. This decouples user code from direct imports and enables runtime operator selection.

### 5.2.1. Catalog Registration and Lookup

**Concept:** Dynamic operator discovery and factory registration.

**Knowledge Point:** How operator packages self-register at import time; the global registry and its forward-compatibility guarantees.

Prose to be written:
- Self-registration pattern in each operator subpackage `__init__.py`
- Lazy import enforcement in `_ensure_registered()`
- Registry structure and lookup semantics
- How custom operators can be registered via `catalog.register()`
- Example: registering a custom selection operator
- Compatibility guarantees (new operators don't break old code)

### 5.2.2. String Specification Parsing and Parameter Binding

**Concept:** Parsing user-facing operator specs into instantiated, configured objects.

**Knowledge Point:** Regular expression parsing of `operator_type:param1=val1,param2=val2` format; type coercion; validation.

Prose to be written:
- Format: `operator_name:key1=val1,key2=val2`
- Type induction (string → int/float/bool)
- Parameter validation (out-of-range checks, conflicting settings)
- Getting help/documentation for an operator via `catalog.get_help()`
- Example: parsing `"simulated_binary:eta=20,cprob=0.9"`
- Example: listing all available operators and their signatures

### 5.2.3. Fitness Evaluator Resolution

**Concept:** Using the same catalog pattern for fitness functions.

**Knowledge Point:** Registration of continuous tests (Sphere, Rastrigin, BBOB), discrete tests, and custom evaluators.

Prose to be written:
- Fitness evaluators follow the same catalog pattern
- Built-in BBOB integration via `BBOBEvaluator`
- Custom evaluator registration
- Example: `"rastrigin:dim=30"`
- Example: `"bbob:fn_name=sphere,dim=10"`

## 5.3. Configuration Loading: TOML-Driven Experiments

Composer supports TOML files for declaring multi-pipeline experiments. A TOML file specifies shared defaults, separate pipeline configurations, and seeds/output directories.

### 5.3.1. TOML Schema and Experiment Structure

**Concept:** Declarative experiment definition without code.

**Knowledge Point:** Expected TOML structure (`[experiment]`, `[experiment.shared]`, `[pipelines.*]`); field meanings and types.

Prose to be written:
- TOML sections: `[experiment]`, `[experiment.shared]`, `[pipelines.pipeline_name]`
- Metadata keys: `name`, `output_dir`, `seeds`
- Shared defaults section: `fitness`, `pop_size`, `generations`, `genome_length`, `bounds`, `prng_impl`, etc.
- Per-pipeline overrides: `selection`, `crossover`, `mutation`, `backend`, `engine_type`
- Type coercion in TOML parsing
- Example TOML: declaring a multi-pipeline comparison (e.g., Tournament vs. Roulette vs. Elite Pool)

### 5.3.2. Parsing and Merging Strategies

**Concept:** Loading TOML and resolving parameter precedence.

**Knowledge Point:** Functions `load_experiment_config()` and `load_config()`; merging logic.

Prose to be written:
- Parsing via `tomllib` (Python 3.11+) or `tomli` backport
- Loading a specific pipeline from the TOML
- Merging strategy: shared → pipeline → inline kwargs
- Handling missing fields and defaults
- Example: loading and running all pipelines from a TOML file

### 5.3.3. Reproducibility and Artifact Organization

**Concept:** Ensuring experiments are reproducible and outputs are organized.

**Knowledge Point:** Output directory structure, result file naming, trace directories, PRNG backend specification.

Prose to be written:
- Output directory layout: results/, artifacts/, traces/
- Seed-level organization (one subdirectory per seed)
- Result file formats (JSON summary, CSV histories, HDF5 traces)
- PRNG backend control via `prng_impl` (Threefry, Philox, etc.)
- Reproducibility via fixed seeds and backend specification
- Example: directory structure for a 10-seed experiment

## 5.4. Engine Factory: Building Engines from Specs

The engine factory module creates fully configured engines from string specifications or explicit parameters. It resolves both native MalthusJAX engines and external adapters.

### 5.4.1. Resolving Native Engines (GA, MO, QD, Island)

**Concept:** Dynamic dispatch to the correct native Engine based on user requests.

**Knowledge Point:** EngineRegistry class; registering new engine builders; resolving engine type strings (e.g., `"mo:num_objectives=2"` or `"qd:archive_size=1000"`).

Prose to be written:
- The registry of native engines (GeneticEngine, MOEngine, QDEngine, IslandModelEngine)
- Parameter parsing in engine type strings (e.g., `"qd:archive_size=1000"`)
- Engine-specific parameter defaults
- Example: declaring a Quality-Diversity run in Composer

### 5.4.2. External Adapters (EvoSAX, QDAX, TensorNEAT)

**Concept:** Wrapping external libraries via Adapter patterns for fair benchmark comparison.

**Knowledge Point:** Adapter responsibilities; bridging the Composer contract to external library semantics; population injection.

Prose to be written:
- Rationale for interop (fair framework comparisons)
- **EvoSAX Adapter**: Wrapping `ask`/`tell` loops and strategy parameters.
- **QDAX Adapter**: Wrapping emitters, metrics, and Map-Elites containers.
- **TensorNEAT Adapter**: Wrapping node/edge structural mutation semantics.
- How MalthusJAX fitness evaluators integrate with these external adapters
- Population injection for fair comparison
- Example: comparing QDEngine (MalthusJAX) vs. QDAX using Composer

## 5.5. Integration with Benchmarking Infrastructure

Composer orchestrates not just single runs but multi-seed comparative benchmarks via the BenchmarkRunner.

### 5.5.1. The Engine Protocol and BenchmarkRunner Contract

**Concept:** Unifying different evolutionary backends under a single interface.

**Knowledge Point:** `Engine` protocol definition; required methods; return value structure.

Prose to be written:
- Engine protocol: `run_once(key) -> Dict`
- Why the protocol matters (extensibility, framework-agnostic)
- Required dict keys: `history`, `summary`, `timings`
- History format: list of per-generation dicts
- Summary format: final fitness, evaluations, etc.

### 5.5.2. BenchmarkRunner Usage with Composer

**Concept:** Orchestrating multi-seed runs and collecting results.

**Knowledge Point:** BenchmarkRunner.run() method; seed iteration; result aggregation into ExperimentResult.

Prose to be written:
- How Composer instantiates BenchmarkRunner with an engine
- Seed loop and PRNG key derivation
- RunResult and ExperimentResult structures
- Aggregated metrics (mean, std, best, worst across seeds)
- Result serialization (JSON, CSV exports)
- Example: running a full multi-seed benchmark loop

### 5.5.3. Result Analysis and Visualization Integration

**Concept:** Connecting Composer results to visualization and comparison tools.

**Knowledge Point:** ExperimentResult and ComparisonResult formats; feeding results to EngineComparator and FunctionalDataAnalyzer.

Prose to be written:
- ExperimentResult contains: RunResult list (one per seed), summary metrics
- ComparisonResult for comparing multiple engines/experiments
- Using visualization.multi_run to plot convergence curves
- Filtering/selecting pipelines for comparison
- Example: running 3 pipelines and generating comparison plots

## 5.6. Advanced Patterns and Extensibility

Extensions and customization patterns for users wanting to build on top of Composer.

### 5.6.1. Custom Operator Registration and Inline Definitions

**Concept:** Adding domain-specific operators at runtime.

**Knowledge Point:** `catalog.register()` for new operators; using Composer with custom operator classes.

Prose to be written:
- Registering a custom selection operator
- Registering a custom fitness evaluator
- Using Composer with explicitly passed operator instances (bypassing catalog lookup)
- Integration with existing Composer workflows
- Example: custom novelty-based selection operator
- Example: custom domain-specific fitness function

### 5.6.2. Programmatic Experiment Composition (Pipeline API)

**Concept:** Composing multi-stage evolutionary pipelines programmatically.

**Knowledge Point:** Pipeline and Node classes; sequential composition of evolutionary stages.

Prose to be written:
- Motivation for multi-stage pipelines
- Pipeline class and its composition semantics
- Node abstraction for evolutionary stages
- Chaining pipelines in Composer
- Example: coevolution pipeline (multiple populations)
- Example: memetic algorithm (local search after GA)

### 5.6.3. Integration with Custom Engines and Strategies

**Concept:** Substituting Composer's default engine with a custom implementation.

**Knowledge Point:** Passing a pre-built engine to `quick_run(engine=...)`; bypassing factory logic.

Prose to be written:
- Rationale for engine substitution (custom algorithm, research extensions)
- Implementing the Engine protocol for custom engines
- Passing engine to quick_run; effect on all other parameters
- Ensuring compatibility with multi-seed orchestration
- Example: custom multi-objective engine via protocol
- Example: hybrid GA+PSO engine

## 5.7. Practical Workflows and Recipes

Common usage patterns and recommended practices.

### 5.7.1. Quick One-Off Experiments

**Concept:** Fast iteration for initial algorithm prototyping.

**Knowledge Point:** Minimal parameter specification; using defaults effectively.

Prose to be written:
- Using quickrun with just a few lines of code
- Relying on sensible defaults
- Example: comparing selection operators in 10 lines
- When to stick with defaults vs. when to customize

### 5.7.2. Reproducible Benchmark Suites

**Concept:** Declaring and running comprehensive benchmark comparisons.

**Knowledge Point:** TOML-based declarative approach; seed management; version control.

Prose to be written:
- TOML as experiment specification (version-controllable)
- Seed specification for statistical rigor
- Organizing results for multiple machines/GPUs
- Capturing timing and hardware metadata
- Example: GECCO-style benchmark suite in TOML

### 5.7.3. Tuning and Hyperparameter Search

**Concept:** Systematic exploration of operator parameters.

**Knowledge Point:** Programmatic loop over parameter ranges; Composer in a hyperparameter search loop.

Prose to be written:
- Template TOML with parameter placeholders
- Looping over selection pressure ranges, mutation rates, etc.
- Aggregating results and identifying best configurations
- Example: grid search over tournament_size ∈ [2,3,4,5]
- Example: random search over 100 random configurations

### 5.7.4. Cross-Framework Comparison (External Libraries vs. MalthusJAX)

**Concept:** Fair algorithm benchmarking across libraries.

**Knowledge Point:** Using External Adapters alongside native engines; controlling for population initialization and seed.

Prose to be written:
- Setting up identical experimental conditions
- Using population injection to ensure fair comparison
- Seed consistency and PRNG backend specification
- Interpreting convergence differences
- Example: MAP-Elites (QDAX) vs. QDEngine (MalthusJAX)
- Fair comparison caveats

