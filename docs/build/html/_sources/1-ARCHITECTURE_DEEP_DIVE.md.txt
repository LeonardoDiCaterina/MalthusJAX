# Architecture Deep Dive: MalthusJAX Framework

MalthusJAX is architected as a vertical integration spanning six tiers—from low-level genome representations and fitness evaluation through evolutionary operators, orchestration engines, configuration-driven experiment composition, multi-seed benchmarking infrastructure, and finally visualization and analysis. This hierarchical organization establishes standardized interfaces at tier boundaries, permitting independent development of components while maintaining tight interoperability. The framework treats each tier as a distinct responsibility layer, with information flowing both upward (to higher-level abstractions) and downward (to lower-level implementations). By enforcing standardized contracts—PyTree genomes, pure functions for operators and evaluators, symmetric state/output tuples from engines—the design enables researchers to substitute components (e.g., swapping a crossover algorithm or evaluation function) without propagating changes through adjacent layers.

This chapter explains the foundational philosophy guiding the system's organization, the concrete responsibilities and boundaries of each tier, how the framework's six chapters map to these tiers, and the terminology used consistently throughout the thesis.

## 1. Design Principles

### 1.1 Complete End-to-End Pipeline

Rather than isolating component development, the framework builds a functional end-to-end pipeline in which all tiers are simultaneously operational. This allows each component to be tested against standardized benchmarks and to interoperate with other tiers from inception.

### 1.2 Standardized Contracts

Each tier communicates with adjacent tiers through narrow, published interfaces:

- **Tier 0 ↔ Tier 1**: Genomes must implement PyTree protocol; fitness evaluators must be pure functions.
- **Tier 1 ↔ Tier 2**: Operators must implement `(key, inputs, generation) → outputs` signature.
- **Tier 2 ↔ Tier 3**: Engines must implement `init_state()` and `step()` returning standardized state/output tuples.
- **Tier 3 ↔ Tier 4**: Experiments are serialized to `ExperimentResult` dataclass with JSON export.

These contracts are enforced through type annotations and protocol definitions.

### 1.3 Code Quality Enforcement

All code is subject to:

- **Type checking**: `mypy --strict` with zero tolerance for untyped code or type: ignore exceptions.
- **Linting**: `ruff check` enforces style consistency and catches common errors.
- **Test coverage**: ≥80% statement coverage on all modules; critical paths (operators, engine kernels) approach 100%.

These checks are non-negotiable preconditions for code acceptance.

## 2. Organizational Structure

### 2.1 Six-Tier Architecture

The framework is organized into six horizontal tiers, each with a distinct responsibility:

```
Tier 5: Visualization
  └─ Single-run and multi-run analysis plots

Tier 4: Benchmarking
  └─ Multi-seed orchestration, result aggregation, I/O

Tier 3: Composer
  └─ Config-driven experiment specification and resolution

Tier 2: Engine
  └─ Evolutionary loop implementation, JIT compilation

Tier 1: Operators
  └─ Selection, Crossover, Mutation implementations

Tier 0: Core
  └─ Genomes, Fitness Evaluators, PRNG utilities
```

Information flows bidirectionally: higher tiers use lower-tier abstractions; lower tiers produce data consumed by higher tiers (e.g., history arrays from Tier 2 feed into Tier 4 result objects).

### 2.2 Tier Responsibilities

| Tier | Module | Primary Responsibility |
|------|--------|------------------------|
| 0 | `malthusjax.core` | Immutable data structures (genomes, populations), fitness evaluation, PRNG state management |
| 1 | `malthusjax.operators` | Pure genetic operators implementing standardized call signatures |
| 2 | `malthusjax.engine` | Evolution loop orchestration, XLA compilation, state threading |
| 3 | `malthusjax.composer` | String-to-operator resolution, TOML parsing, multi-backend support |
| 4 | `malthusjax.benchmarking` | Multi-seed experiment execution, result serialization, format-agnostic I/O |
| 5 | `malthusjax.visualization` | Convergence plotting, multi-run analysis, statistical comparison |

## 3. Terminology

The following terms are used with consistent meaning throughout this thesis:

- **Genome**: An immutable PyTree containing a JAX array and structural metadata. Represents a candidate solution.
- **Population**: A collection of genomes with associated fitness values and optional auxiliary metrics (distance matrices, KPIs).
- **Evaluator**: A pure function computing scalar fitness from a genome.
- **Operator**: A pure function transforming population state (e.g., Selection, Crossover, Mutation).
- **Engine**: The orchestrator implementing the evolution loop via Tier 2.
- **PRNG Key**: JAX's explicit random state; used exactly once and never reused.
- **Generation**: One iteration of the evolution loop.
- **ExperimentResult**: Serialized container holding all runs, histories, and aggregated metrics from an experiment.

## 4. Related Work

The architecture of MalthusJAX synthesizes design patterns from several established frameworks in the evolutionary computation and machine learning ecosystems. The ask/tell population-based interface from evosax informed the design of the Engine tier, enabling decoupled execution from state management. The operator modularity and registry-based plugin system of DEAP provided the foundation for the Composer's operator catalog and string-to-instance resolution mechanism, allowing experiments to be specified declaratively. The benchmarking and result aggregation infrastructure of pygmo shaped the multi-seed execution strategy and comparative analysis utilities of the Benchmarking tier, establishing patterns for cross-algorithm comparison and result serialization. Configuration-driven orchestration patterns from the data pipeline framework kedro influenced the TOML-based specification system and separation of concerns between experiment definition and execution. Rather than adopting any single framework wholesale, MalthusJAX integrates these insights into a coherent architecture unified by type safety, standardized contracts, and PyTree-based data structures native to JAX.

