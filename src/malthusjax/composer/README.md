# `malthusjax.composer` — Reference

Scope: `malthusjax.composer.composer`, `malthusjax.composer.catalog`, `malthusjax.composer.config`, `malthusjax.composer.engine_catalog`, `malthusjax.composer.engine_factory`, `malthusjax.composer.genome_catalog`, `malthusjax.composer.decorators`, `malthusjax.composer.evosax_adapter`, `malthusjax.composer.qdax_adapter`, `malthusjax.composer.tensorneat_adapter`, `malthusjax.composer.kozax_adapter`, `malthusjax.composer.pipeline`. Every claim below is traceable directly to source code and docstrings.

---

## Overview & Architecture

The `malthusjax.composer` package provides the top-level experiment orchestration layer. It bridges low-level core, operator, and engine APIs with high-level experiment workflows, allowing users to define, run, and compare evolutionary algorithms declaratively or programmatically.

Key capabilities:
- **String DSL & Operator Specs**: Parse string specifications (`"name:key1=val1,key2=val2"`) into instantiated PyTree operators and evaluators.
- **Declarative TOML Experiments**: Load TOML experiment suites with shared baseline defaults and per-pipeline overrides.
- **Fair Multi-Seed Benchmarking**: Share initial population states and seeds across pipeline comparisons to isolate algorithmic differences.
- **Universal External Framework Adapters**: Wrap third-party EAs (EvoSAX, QDAX, TensorNEAT, Kozax) under a single unified `Engine` protocol interface (`run_once(key)`).

---

## `malthusjax.composer.composer`

The `Composer` class is the primary entry point, exposing three main orchestration methods:

### `quick_run(...) -> ExperimentResult`
Interactive method for running single pipeline sweeps across multiple random seeds.
- **Operator Specs**: Accepts string specifications for `fitness`, `selection`, `crossover`, `mutation`, `genome_type`.
- **Backend Selection**: `backend="malthusjax"` (default), `"evosax"`, `"qdax"`, `"tensorneat"`.
- **Seed Handling**: Normalizes `seeds` (iterable or integer count) via `_normalize_seeds()`.
- **Returns**: `ExperimentResult` containing per-seed `RunResult` records and aggregated statistics (`.aggregated_summary()`).

### `from_toml(path, ...) -> ComparisonResult`
Declarative entry point for loading experiment TOML files via `load_experiment_config`.
- Parses shared defaults (`[experiment.shared]`) and pipeline overrides (`[pipelines.*]`).
- Executes all pipelines in sequence across specified random seeds.
- **Returns**: `ComparisonResult` containing pipeline results, statistical summary tables (`.summary_table()`), and convergence plotting tools (`.plot_convergence()`).

### `compare(pipelines, ...) -> ComparisonResult`
Programmatic multi-pipeline benchmarking entry point.
- Accepts a dictionary mapping pipeline names to `quick_run` parameter keyword dictionaries.
- Enforces shared initial seeds and populations across pipelines for fair statistical comparison.

---

## `malthusjax.composer.catalog` & Registries

### `OperatorCatalog` (`catalog.py`)
Parses operator string specifications:
- **Regex Parsing**: `"operator_name:param1=val1,param2=val2"`.
- **Type Coercion**: Automatically coerces numerical parameter strings to `int` or `float` and booleans (`"true"`, `"false"`) to `bool`.
- **Methods**: `get_selection(spec)`, `get_crossover(spec)`, `get_mutation(spec)`, `list_operators()`.

### Component Decorators (`decorators.py`)
Provides decorators for registering custom components into global registries:
- `@register_selection(name="...")`
- `@register_crossover(name="...")`
- `@register_mutation(name="...")`
- `@register_fitness(name="...")`
- `@register_engine(name="...")`
- `@register_genome(name="...")`

### `EngineRegistry` (`engine_catalog.py`)
Factory registry mapping engine keywords (`"ga"`, `"mo"`, `"qd"`, `"island"`) to native engine builder functions in `engine_factory.py`.

### `GenomeCatalog` (`genome_catalog.py`)
Resolves genome type specs (`"real"`, `"binary"`, `"categorical"`, `"linear"`) to default `GenomeConfig` instances.

---

## `malthusjax.composer.config`

The `load_experiment_config(path)` function parses TOML files using `tomllib` (Python 3.11+) or `tomli`.

### Verified TOML Structure
- `[experiment]`: Top-level metadata (`name: str`, `output_dir: str`).
- `[experiment.shared]`: Shared baseline configuration (`fitness`, `pop_size`, `generations`, `genome_length`, `bounds`, `seeds`, `prng_impl`, `elitism`, `maximize`). Lists for `bounds` and `seeds` are automatically converted to tuples.
- `[pipelines.<pipeline_name>]`: Per-pipeline overrides (`backend`, `engine_type`, `selection`, `crossover`, `mutation`).
- `[data.<data_id>]`: Optional data registry sections extracted by `_parse_data_section`.

Returns an `ExperimentLoadResult` containing metadata, resolved pipeline dictionaries, and data registry definitions.

---

## External Framework Adapters

Composer provides universal adapter functions to bridge third-party frameworks into MalthusJAX's `Engine` benchmark interface (`run_once(key) -> Dict` returning `"history"`, `"summary"`, `"timings"`):

- **EvoSAX Adapter (`evosax_adapter.py`)**: `build_evosax_engine()` wraps EvoSAX strategies (e.g., `SimpleGA`, `CMA_ES`) into `UniversalAdapterEngine`.
- **QDAX Adapter (`qdax_adapter.py`)**: `build_qdax_engine()` wraps QDAX emitters, repertoires, and metrics into `UniversalAdapterEngine`.
- **TensorNEAT Adapter (`tensorneat_adapter.py`)**: `build_tensorneat_engine()` wraps TensorNEAT algorithms and problems into `UniversalAdapterEngine`.
- **Kozax Adapter (`kozax_adapter.py`)**: `build_kozax_engine()` wraps Kozax GP strategies into `UniversalAdapterEngine`.

---

## Result Objects & Serialization

Composer connects execution to `BenchmarkRunner` (`benchmarking` package):
- `ExperimentResult`: Holds multi-seed `RunResult` outputs for a single pipeline. Supports `.aggregated_summary()`, `.combined_history()`, and `.canonical_summary`.
- `ComparisonResult`: Holds multi-pipeline `ExperimentResult` objects. Supports `.summary_table()` (LaTeX/Markdown table export), `.plot_convergence()`, and `.convergence_data()`.
- Result Serialization: Outputs structured JSON files (`metadata/config_snapshot.toml`, `data/<pipeline>/seed_<X>.json`, `analysis/summary.json`).
