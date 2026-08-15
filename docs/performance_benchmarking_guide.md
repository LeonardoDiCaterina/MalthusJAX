# MalthusJAX Performance Benchmarking Guide

## Goal

Build a native MalthusJAX engine implementation that:
1. **Matches EvoSAX's execution speed** on small-to-medium continuous optimisation problems (the primary open problem).
2. **Preserves statistical parity with EvoSAX** — identical optimisation quality under seed-aligned comparison (already demonstrated).
3. **Keeps the modular operator architecture** — users must still be able to plug in any `BaseMutation`, `BaseCrossover`, or `BaseSelection` operator via the TOML/Composer API.

This guide documents the complete measurement workflow that was built for this iteration process, explains what has already been tried (and why it failed), and provides a structured iteration protocol for future experiments.

---

## 1. The Performance Gap

### Current Baselines (CPU — Apple Silicon M-series, pop=195, dims=9, gens=387)

| Engine | Wall-clock (avg) | Notes |
|---|---:|---|
| **EvoSAX SimpleGA** | ~29 ms | Pure flat-array pipeline, the target |
| `SimpleGAEngine` (MJX) | ~200 ms | Hardcoded EvoSAX logic, MJX PyTree carry |
| `LightenedGeneticEngine` | ~250 ms | Vectorised operators, no elite merge |
| `GeneticEngine` (standard) | ~340–500 ms | Full modular pipeline |
| ~~`VectorizedEngine`~~ | ~~440–750 ms~~ | **Deleted** — confirmed dead-end |

> **On GPU (H100):** All times compress significantly, but the *relative* ratios remain similar. The throughput advantage of EvoSAX is structural, not hardware-specific.

### Root Cause: XLA HLO Graph Complexity

Extensive profiling (via `extract_hlo.py` and Perfetto traces) ruled out the following as the bottleneck:
- ❌ JAX JIT recompilation (warmup phase correctly isolates compilation)
- ❌ PyTree metadata / carry structure (both engines carry identical 6-leaf arrays, 653 parameters)
- ❌ History tracking (`TrackBest.LIGHT` adds zero overhead)
- ❌ Elite merging with `dynamic_update_slice` (measurable but not the primary bottleneck)

The **actual bottleneck** is the number and structure of XLA operations in the HLO graph:

| Engine | HLO IR Lines | `while` loops | `fusion` kernels |
|---|---:|---:|---:|
| EvoSAX SimpleGA | ~1,800 | ~0 | high |
| `GeneticEngine` | ~2,247 | ~1,775 | medium |
| ~~`VectorizedEngine`~~ | ~~2,675~~ | ~~2,317~~ | ~~low~~ |

The core issue: MalthusJAX's `lax.scan` loop continuously **unpacks and repacks PyTree dataclasses** (`BasePopulation`, `RealGenome`) between each phase (selection → crossover → mutation → merge → evaluate). XLA sees these as distinct tuple creation/destruction operations and cannot fuse them into a single kernel. EvoSAX operates exclusively on flat `jax.Array` matrices with no intermediate dataclass instantiation.

---

## 2. What Has Been Tried

### ✅ VectorizedEngine (abandoned)

**Idea:** Bypass `BaseCrossover.__call__` and `BaseMutation.__call__` (Tier 1 PyTree interface) and call the vectorised math kernels (Tiers 2/3) directly on flat arrays. Keep a lightweight `VectorizedEvolutionState(genes, fitness, rng_key)` in the scan carry.

**Result:** 30–50% **slower** than `GeneticEngine`. Reason: the Tier 2/3 operators (`_generate_noise`, `_mutate_values`) still each contain their own `jax.vmap` calls for noise generation. Adding a vectorised engine wrapper on top created nested `vmap` stacks in the HLO graph, increasing the `while`-loop count from 1,775 to 2,317.

**Lesson:** You cannot achieve EvoSAX-level fusion by wrapping existing operators. The fusion must happen **inside** the operator call, at the point where random noise is generated and applied in a single mathematical expression.

### ✅ SimpleGAEngine (proof of concept)

**Idea:** Port EvoSAX's exact `ask()`/`tell()` logic into a MJX engine. No modular operators — the crossover and mutation are a single fused function:
```python
def recombine_and_mutate(key_c, key_m, p1, p2):
    mask = jax.random.uniform(key_c, p1.shape) < crossover_rate
    offspring = jnp.where(mask, p2, p1)
    return offspring + std * jax.random.normal(key_m, p1.shape)
```

**Result:** ~200ms (vs ~29ms EvoSAX). Still 7× slower. Reason: even with fused arithmetic, MalthusJAX carries `BasePopulation` PyTrees through the scan, and the `lax.scan` carry touches more memory than EvoSAX's flat `(population, fitness, std)` state.

**Lesson:** The carry structure is a second bottleneck. To achieve true parity, the scan carry must be as flat as EvoSAX's.

---

## 3. The Complete Benchmark Workflow

### Setup

All commands use the `perf-*` Makefile targets added in August 2026. The default TOML is `configs/perf/h1_speed_vs_evosax.toml`. Override any variable on the command line.

```bash
# Verify the harness works in ~60 seconds
make perf-bench-smoke
```

### Step 1 — HLO Analysis (Fast, No GPU Required)

Extract the optimised XLA HLO for all pipelines and compare graph complexity:

```bash
# Default params: dims=9, pop=195, gens=387
make perf-hlo

# Custom params
make perf-hlo PERF_DIMS=50 PERF_POP=512 PERF_GENS=100
```

**Output:** `results/perf/h1_speed_vs_evosax/hlo/`
- `<pipeline>.hlo.txt` — full optimised HLO for each engine
- `hlo_summary.md` — side-by-side table of IR line counts, fusion kernels, while-loops

**What to look for:**
- **IR lines**: fewer is better. EvoSAX target: ~1,800.
- **`while` loops**: should approach 0. Each loop is a fusion barrier.
- **`fusion` kernels**: more is better. Each fusion means XLA merged multiple ops into one GPU kernel.

HLO analysis is fast (seconds) and requires no GPU. **Always run this first** before a full benchmark — if the HLO graph is worse, the timing will definitely be worse.

### Step 2 — Timing Benchmark (Quality + Speed)

Run the full statistical benchmark across all pipelines and seeds:

```bash
make perf-bench
```

**Output:** `results/perf/h1_speed_vs_evosax/`
- Per-seed JSON files with convergence history and `timings` dict (compile time, run time)
- Reference pipeline: `evosax_baseline`

**What to look for:**
- `execution_time_ms` in the JSON — wall-clock time for the compiled scan (excludes JIT warmup)
- `best_fitness` trajectory — must be statistically equivalent to `evosax_baseline`

### Step 3 — Perfetto Profiling (Kernel-level Breakdown)

Generate Perfetto traces for each pipeline (one subprocess per pipeline for isolation):

```bash
make perf-perfetto

# Custom port hint in printed instructions
make perf-perfetto PORT=6007
```

**Output:** `results/perf/h1_speed_vs_evosax/perfetto/<pipeline>/`

### Step 4 — TensorBoard

Launch TensorBoard to view Perfetto traces interactively:

```bash
# Foreground (blocking, Ctrl+C to stop)
make perf-tb PORT=6006

# Background (prints PID, logs to logs/tensorboard_6006.log)
make perf-tb-bg PORT=6006
```

Open `http://localhost:6006` → **Profile** tab. Look for:
- **Step time breakdown**: how much time is selection vs crossover vs mutation vs evaluate
- **XLA kernel names**: fused kernels appear as single entries; unfused ops as separate entries
- **Host-device transfer**: should be zero during the scan (all on-device)

### Step 5 — Full Chain

```bash
# Run all steps sequentially (no TensorBoard)
make perf-all

# Then inspect
make perf-tb PORT=6006

# Headless (cluster / nohup)
make perf-all-nohup PORT=6006
```

### Step 6 — Statistical Parity Analysis

After `perf-bench`, validate that quality is preserved:

```bash
make benchmark-analyze TOML=configs/perf/h1_speed_vs_evosax.toml
```

This runs Wilcoxon signed-rank and TOST equivalence tests comparing each pipeline to `evosax_baseline`.

---

## 4. Adding a New Engine Candidate

This is the core iteration loop. Each new idea follows the same protocol.

### A. Implement the engine

Create `src/malthusjax/engine/<my_engine>.py`. Requirements:
- Inherit from `AbstractEngine` (for `get_hlo_text()` support in `extract_hlo.py`)
- Use `@struct.dataclass` for state (for `lax.scan` compatibility)
- Implement `init_state(rng_key)` → `EvolutionState` and `step(state)` → `(state, metrics)`

### B. Register it (optional)

For Composer/TOML access, add to `engine/__init__.py` `_register_engines()`:
```python
register_table([("my_engine", _my_engine_factory, {...})])
```

### C. Add a pipeline to the harness TOML

Edit [`configs/perf/h1_speed_vs_evosax.toml`](../configs/perf/h1_speed_vs_evosax.toml):
```toml
[pipelines.mjx_candidate_v1]
backend     = "malthusjax"
engine_type = "my_engine"
selection   = "evosax_mimic_selection:num_selections={pop_size},elite_k={elite_k}"
crossover   = "evosax_uniform_crossover:crossover_rate=0.3"
mutation    = "evosax_gaussian:mutation_strength=1.0"
elitism     = 0
```

### D. Run the harness

```bash
# 1. Check XLA graph first — fast, no GPU needed
make perf-hlo

# 2. If HLO improved → run timing + quality benchmark
make perf-bench

# 3. If timing improved → profile kernels
make perf-perfetto && make perf-tb PORT=6006

# 4. Validate statistical parity
make benchmark-analyze TOML=configs/perf/h1_speed_vs_evosax.toml
```

**Decision rule:** An engine variant is promising if and only if:
1. HLO IR lines < `GeneticEngine` baseline AND
2. `execution_time_ms` < `GeneticEngine` baseline AND
3. Statistical parity with `evosax_baseline` is maintained (Wilcoxon p-value > 0.05, or TOST equivalence confirmed)

---

## 5. Design Directions to Explore

### Direction A: Flat-Carry Scan with `FastPath` Operator Protocol (Recommended)

**Idea:** Keep modular operators but define a second, lighter protocol alongside the existing `BaseMutation.__call__`:

```python
class FastPathMutation(Protocol):
    def apply_flat(self, key: jax.Array, values: jax.Array) -> jax.Array:
        """Apply mutation directly on flat gene matrix (pop_size, num_dims).
        No PyTree wrapping. Must be JIT-compatible."""
```

The engine maintains the scan carry as `(values: jax.Array, fitness: jax.Array, rng_key: Array)` — matching EvoSAX's structure exactly. It calls `mutation.apply_flat()` and `crossover.apply_flat()` inside the loop. The `BasePopulation` PyTree is only constructed **once** at the very end to produce the output.

**Key benefit:** Operator modularity is preserved. Users implement `apply_flat()` in their custom mutation class and register it normally. The catalog API is unchanged.

**HLO hypothesis:** Should produce ~2,000 IR lines (vs EvoSAX's 1,800), since the carry overhead is eliminated but per-operator vmaps remain.

### Direction B: Fused Crossover+Mutation Kernel

**Idea:** The biggest XLA fusion opportunity: combine crossover and mutation into one `jax.vmap` call (like EvoSAX does). Requires a new `FusedCrossoverMutation` operator class:

```python
class FusedCrossoverMutation(Protocol):
    def apply_flat(self, key: jax.Array, p1: jax.Array, p2: jax.Array) -> jax.Array:
        """Fuse crossover and mutation into a single elementwise op."""
```

This eliminates the intermediate `offspring` array that is created between crossover and mutation in the current pipeline.

### Direction C: JIT the Scan Body Independently

**Idea:** Manually JIT the scan body function and pass it as a compiled XLA computation to `lax.scan`. May help XLA's fusion planner by providing a tighter type signature for the body.

**Status:** Unexplored. Medium complexity.

### Direction D: Accept the Gap, Focus on Scale

**Observation:** The 7–17× gap exists only on tiny problems (dims=9, pop=195). At H1/H2/H3 benchmark scale (dims=10–500, pop=64–4096) on H100 GPU, actual compute dominates and the relative gap narrows. If thesis benchmarks run at target scale in acceptable wall-clock time, the performance gap may not be research-blocking.

---

## 6. Reference: Key Files

| File | Purpose |
|---|---|
| `configs/perf/h1_speed_vs_evosax.toml` | Harness TOML — add new candidate engines here |
| `configs/perf/smoke_speed_vs_evosax.toml` | Smoke TOML — 3 seeds, fast validation |
| `scripts/extract_hlo.py` | HLO extraction for all pipelines + summary table |
| `scripts/trace_pipelines.py` | Perfetto trace generation with configurable port |
| `scripts/benchmark_runner.py` | Full quality+timing benchmark (TOML-driven) |
| `scripts/benchmark_analyzer.py` | Statistical analysis (Wilcoxon, TOST, tables) |
| `src/malthusjax/engine/genetic_fastengine.py` | `GeneticEngine` — current production engine |
| `src/malthusjax/engine/simple_ga.py` | `SimpleGAEngine` — hardcoded EvoSAX logic, speed reference |
| `src/malthusjax/engine/genetic_lightened.py` | `LightenedGeneticEngine` — vectorised operators variant |
| `docs/architectural_challenge_report.md` | Original diagnosis of the XLA PyTree overhead problem |

---

## 7. Reference: EvoSAX SimpleGA Pattern

EvoSAX's speed comes from this pattern — study it carefully when designing new engines:

```python
# evosax/algorithms/population_based/simple_ga.py
# ask() — the entire generation step on flat arrays

idx = jnp.argsort(state.fitness)
population = state.population[idx]              # flat (pop_size, num_dims) sort

p = jnp.arange(pop_size) < num_elites          # boolean elite mask

# Parent sampling: direct indexing on flat arrays
parents_1 = jax.random.choice(k1, population, (pop_size,), p=p)
parents_2 = jax.random.choice(k2, population, (pop_size,), p=p)

# Crossover + Mutation: single vmap, XLA fuses into one kernel
population = jax.vmap(crossover)(k_cross, parents_1, parents_2, crossover_rate)
population = jax.vmap(mutation)(k_mut, population, state.std)

# tell() — store back, no dataclass reconstruction
return state.replace(population=population, fitness=fitness, ...)
```

The `crossover` and `mutation` are pure elementwise functions — no PyTrees anywhere:

```python
def crossover(key, p1, p2, rate):
    mask = jax.random.uniform(key, p1.shape) < rate
    return p1 * (1 - mask) + p2 * mask      # XLA fuses: mask + select → 1 kernel

def mutation(key, solution, std):
    return solution + std * jax.random.normal(key, solution.shape)  # XLA fuses: scale + add → 1 kernel
```

The scan carry is 6 flat arrays — the same count as MalthusJAX after PyTree flattening. The critical difference: **EvoSAX never calls `.replace()` on a dataclass inside the scan body**. MalthusJAX calls `population.replace(genes=...)` on every generation, creating tuple reconstruction overhead that XLA cannot fuse away.
