# Benchmarking Test Suite Architecture & Harness (`tests/benchmarks/`)

This directory contains the continuous performance regression, JIT compilation profiling, and evolutionary parity benchmark suites for **MalthusJAX**.

> [!NOTE]
> For unit tests covering the benchmarking orchestration library itself (`malthusjax.benchmarking.*`), see [`tests/benchmarking/`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarking). This directory (`tests/benchmarks/`) houses the active micro- and macro-benchmark execution suites (BM-01 through BM-12).

---

## 1. Architectural Overview

The benchmark test suite is designed around three primary pillars:
1. **Zero-Overhead JAX Verification**: Proving that MalthusJAX abstractions (composable evaluators, genome containers, modular genetic operators, engine adapters) compile down to pure, fused XLA HLO without runtime penalty compared to direct monolithic implementations (such as `evosax`).
2. **Evolutionary & Statistical Parity**: Validating that algorithmic improvements (such as injection-mode operators or parallel key derivation) maintain identical mathematical optimization trajectories against golden-standard baselines.
3. **Phase-Level Profiling & Microbenchmarks**: Pinpointing exact latency attribution across PRNG entropy allocation, selection, crossover/mutation reproduction, buffer merging, and fitness evaluations.

```
tests/benchmarks/
├── README.md                               # This documentation
├── conftest_benchmarks.py                  # Shared fixtures, engine adapters, and parity runners
├── test_benchmark_01_single_step.py        # BM-01: Single-step warm dispatch latency
├── test_benchmark_02_multi_gen_throughput.py # BM-02: Multi-generation scan throughput
├── test_benchmark_03_compilation.py        # BM-03: Cold JIT compilation overhead
├── test_benchmark_04_operators.py          # BM-04: Isolated operator-level microbenchmarks
├── test_benchmark_05_convergence.py        # BM-05: BBOB fitness convergence parity
├── test_benchmark_06_unroll.py             # BM-06: lax.scan loop unrolling sweep
├── test_benchmark_07_phases.py             # BM-07: GeneticEngine.step() phase breakdown
├── test_benchmark_08_scaling.py            # BM-08: Population size scaling sweep
├── test_benchmark_09_injection.py          # BM-09: Injection-mode vs standard operator latency
├── test_benchmark_10_key_derivation.py     # BM-10: SPLIT vs FOLD PRNG derivation cost
├── test_benchmark_11_injection_parity.py   # BM-11: Injection operator multi-seed statistical parity
└── test_benchmark_12_adapter_overhead.py   # BM-12: Adapter facade zero-overhead verification
```

---

## 2. Benchmark Harness & Fixtures (`conftest_benchmarks.py`)

All benchmark suites build on shared utilities, engine wrappers, and timing controls defined in [`conftest_benchmarks.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/conftest_benchmarks.py).

### 2.1 Engine Protocol Adapters

Both frameworks are wrapped into unified classes satisfying the [`Engine`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/benchmarking/engine_protocol.py) protocol (`run_once(key) -> Dict[str, Any]`):

- **[`MalthusJAXBenchEngine`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/conftest_benchmarks.py#L376-L542)**:
  - Wraps a configured [`GeneticEngine`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/engine/genetic_fastengine.py).
  - Uses the 4-axis Composable Evaluator: [`OptimizationEvaluator`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/evaluators.py) paired with [`BBOBEnv`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/environments.py), [`IdentityTransform`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/base.py), [`IdentityInterpreter`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/interpreters.py), and [`ScalarOutput(maximize=False)`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/base.py).
  - Configurable operator injection, key derivation strategies (`SPLIT` vs `FOLD`), selection mechanisms (`ElitePoolSelection`, `TournamentSelection`, `RouletteSelection`), and evosax operator wrappers for isolation.
- **[`EvosaxBenchEngine`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/conftest_benchmarks.py#L544-L681)**:
  - Wraps `evosax.algorithms.SimpleGA` and `evosax.problems.BBOBProblem` using evosax 0.2.0 `ask`/`tell` semantics.

### 2.2 Fair Benchmarking Methodology

To ensure scientific rigor when comparing JAX frameworks:

1. **Deterministic Canonical Populations**:
   [`_canonical_population(key, pop_size, dims, bounds)`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/conftest_benchmarks.py#L266-L284) generates an identical float32 gene array `(pop_size, dims)` from the seed key. When `canonical_init=True`, both MalthusJAX and Evosax start from the exact same initial genomes, eliminating initialization noise from parity metrics.
2. **Workload Equivalence (`elitism=0`)**:
   Evosax `SimpleGA` selects parents from an elite pool but applies crossover and mutation to *all* offspring. In MalthusJAX, non-zero elitism skips operator evaluation for preserved elite copies. Benchmark engines enforce `elitism=0` when matching `SimpleGA` so both pipelines execute identical tensor arithmetic per generation.
3. **Pre-Compilation & Warmup**:
   Both engines pre-compile their `jax.lax.scan` loop during `_setup()` on a dummy key before the timing window opens.
4. **Asynchronous Barrier Synchronization**:
   All benchmark timing blocks invoke `jax.tree_util.tree_map(lambda x: x.block_until_ready(), ...)` on both final state and history before the clock stops, preventing premature timing on asynchronous device queues.
5. **Dead-Code Elimination (Stripped History)**:
   In `MalthusJAXBenchEngine`, the inner scan body extracts only scalars needed for telemetry (`generation`, `best_fitness`, `mean_fitness`). Unused PyTree branches are pruned by XLA dead-code elimination, mirroring Evosax memory footprints.
6. **Bulk PCIe Host Transfer**:
   A single `jax.device_get` pulls all collected metrics to host memory *after* execution timing completes.

---

## 3. The 12 Benchmark Suites

| ID | File | Focus Area | Frameworks / Components | Key Parameters |
|---|---|---|---|---|
| **BM-01** | [`test_benchmark_01_single_step.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_01_single_step.py) | Single-Step Latency | MalthusJAX vs Evosax | $P \in \{100, 500, 1024, 1025\}$, $D \in \{10, 50\}$, native/evosax ops, roulette, tournament |
| **BM-02** | [`test_benchmark_02_multi_gen_throughput.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_02_multi_gen_throughput.py) | Multi-Gen Scan Throughput | MalthusJAX vs Evosax | Configurable via `--pop-sizes` and `--num-gens`, $D \in \{10, 50\}$ |
| **BM-03** | [`test_benchmark_03_compilation.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_03_compilation.py) | Cold JIT Compilation Time | MalthusJAX vs Evosax | First-call compilation overhead over $P \in \{100, 500, 1024, 1025\}$, $D \in \{10, 50\}$ |
| **BM-04** | [`test_benchmark_04_operators.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_04_operators.py) | Operator Microbenchmarks | MalthusJAX Isolated Ops | `ElitePoolSelection`, `TournamentSelection`, `UniformCrossover`, `GaussianMutation`, `BBOBEnv` |
| **BM-05** | [`test_benchmark_05_convergence.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_05_convergence.py) | Convergence Parity | MalthusJAX vs Evosax | 10 seeds, 500 gens, 5 BBOB functions (`sphere`, `rastrigin`, `schwefel`, etc.) |
| **BM-06** | [`test_benchmark_06_unroll.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_06_unroll.py) | Loop Unroll Factor Sweep | MalthusJAX `lax.scan` | `unroll_num` $\in \{1, 5, 10, 25\}$, $P=100$, $D=10$, 50 gens |
| **BM-07** | [`test_benchmark_07_phases.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_07_phases.py) | Step Phase Breakdown | MalthusJAX Profiling | Phase 0 (entropy), Phase 1 (selection), Phase 2 (reproduction), Phase 3a (merge), Phase 3b (eval) |
| **BM-08** | [`test_benchmark_08_scaling.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_08_scaling.py) | Population Scaling Sweep | MalthusJAX vs Evosax | $P \in \{50, 100, 200, 500, 1000\}$, $D=10$, per-step scaling |
| **BM-09** | [`test_benchmark_09_injection.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_09_injection.py) | Injection Operator Latency | Standard vs Injection | Crossover (`uniform`, `blend`, `sbx`, `binomial`), Mutation (`gaussian`, `ball`, `polynomial`) |
| **BM-10** | [`test_benchmark_10_key_derivation.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_10_key_derivation.py) | Key Derivation Strategy | `SPLIT` vs `FOLD` | Sequential `jr.split` vs parallel `jr.fold_in`, single-step and 50-gen scan |
| **BM-11** | [`test_benchmark_11_injection_parity.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_11_injection_parity.py) | Statistical Quality Parity | Injection vs Evosax Golden | 30 seeds, 100 gens, canonical initialization, full CI validation |
| **BM-12** | [`test_benchmark_12_adapter_overhead.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/tests/benchmarks/test_benchmark_12_adapter_overhead.py) | Adapter Facade Overhead | `EvosaxEngineAdapter` vs Raw | Proves composer adapter overhead is statistically zero inside XLA compiled graph |

---

## 4. Running Benchmarks

### 4.1 Environment Requirements

Always run benchmark suites within the dedicated project environment (`GP_env_2`) where JAX, Pytest, and `pytest-benchmark` are installed:

```bash
conda activate GP_env_2
```

> [!IMPORTANT]
> Always pass `--no-cov` when executing benchmark tests. Pytest code-coverage instrumentation hooks into Python byte-code execution and introduces severe measurement distortion that invalidates JAX dispatch timings.

### 4.2 Using Pytest Directly

```bash
# Run a specific benchmark suite with timing measurements
pytest tests/benchmarks/test_benchmark_01_single_step.py --no-cov -v --benchmark-only

# Run functional smoke tests across all benchmarks (timing disabled)
pytest tests/benchmarks/ --no-cov -v --benchmark-disable

# Export benchmark measurements to JSON for automated comparison
pytest tests/benchmarks/test_benchmark_02_multi_gen_throughput.py \
  --no-cov --benchmark-only --benchmark-json=results/throughput_bm.json

# Run multi-generation throughput with custom population and generation grids
pytest tests/benchmarks/test_benchmark_02_multi_gen_throughput.py \
  --no-cov --pop-sizes="64,256,1024" --num-gens="50,200"

# Filter by parameter combination using pytest -k
pytest tests/benchmarks/test_benchmark_12_adapter_overhead.py \
  -k "test_adapter_overhead_execution[10-100]" --no-cov --benchmark-disable -v
```

### 4.3 Using Makefile Targets

The root [`Makefile`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/Makefile) provides convenient targets for all benchmark groups:

```bash
# Functional tests (validates execution without collecting benchmark timers)
make test-bench

# Full pytest-benchmark snapshot suite
make test-bench-snapshot

# Run individual benchmark groups
make test-bench-group-01   # Single-step latency
make test-bench-group-02   # Multi-generation throughput
make test-bench-group-03   # Cold JIT compilation overhead
make test-bench-group-04   # Operator microbenchmarks
make test-bench-group-05   # Convergence parity
make test-bench-group-06   # Unroll sweep
make test-bench-group-07   # Step phase breakdown
make test-bench-group-08   # Scaling sweep
make test-bench-group-09   # Injection operator latency
make test-bench-group-10   # Key derivation strategy
make test-bench-group-11   # Injection statistical parity
make test-bench-group-12   # Adapter overhead verification

# Background execution (nohup)
make test-bench-group-01-nohup
make test-bench-group-11-nohup
```

---

## 5. Statistical Parity & Artifact Outputs

Parity benchmark groups (**BM-05** and **BM-11**) evaluate optimization dynamics across multiple independent seeds using [`BenchmarkRunner`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/benchmarking/runner.py).

### 5.1 Optimization Sign Convention

MalthusJAX follows the repository-wide **minimization** standard (lower fitness is better):
- The evaluators use [`ScalarOutput(maximize=False)`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/core/fitness/composable/base.py).
- BBOB benchmarks report raw minimization values.
- Improvement across generations is calculated as:
  $$\Delta_{\text{best}} = \text{start\_best\_fitness} - \text{end\_best\_fitness} \ge 0$$
- A valid improving run satisfies $\Delta_{\text{best}} \ge 0$.

### 5.2 Artifact Storage

When parity tests write artifacts, outputs are organized under the directory defined by the `MALTHUSJAX_PARITY_RESULTS` environment variable (defaults to `results/fitness_parity/`):

```
results/fitness_parity/
└── mjx_sphere_p200_d10_inj_uniformx_gaussianm_split/
    ├── summary.json              # Aggregated metrics (mean, std, 95% CI)
    ├── histories_combined.csv    # Per-generation metrics across all seeds
    └── runs/                     # Per-seed detailed run telemetry
```

The returned [`ComparisonResult`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/benchmarking/results.py) produces formatted summary tables and convergence curves accessible programmatically:

```python
table = comparison.summary_table()
mjx_mean = table["malthusjax"]["best_fitness"]["mean"]
esx_mean = table["evosax"]["best_fitness"]["mean"]
ratio = mjx_mean / esx_mean  # Should remain close to 1.0
```

---

## 6. Developer Guidelines for Adding Benchmarks

When authoring a new benchmark test file:
1. **Inherit from conftest helpers**: Import engine wrappers (`MalthusJAXBenchEngine`, `EvosaxBenchEngine`), canonical fixtures, and constants from `conftest_benchmarks.py`.
2. **Warm up explicitly**: Call `block_until_ready()` on compilation and initialization warmup arrays before invoking `benchmark.pedantic(...)`.
3. **Use pedantic measurement**: Use `benchmark.pedantic(_run, iterations=1, rounds=..., warmup_rounds=...)` to manage JAX compilation separation.
4. **Group names**: Set `benchmark.group` and `benchmark.name` clearly so reports can group comparative runs (e.g. `single_step/pop{pop_size}_d{dims}`).
5. **Memory hygiene**: JAX compilation caches are automatically cleared via the `pytest_runtest_teardown` hook in `tests/conftest.py`, ensuring large sweeps do not cause GPU/CPU memory leaks.
