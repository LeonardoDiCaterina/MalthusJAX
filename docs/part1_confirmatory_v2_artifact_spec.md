# Part 1 GECCO Confirmatory Benchmark Artifact Specification (Schema v2)

This document describes the upgraded **Part 1 Confirmatory Parity & Steady-State Speed Pipeline (`schema_version: 2`)** implemented in:
- `configs/publication/part1_confirmatory.json`
- `scripts/run_publication_part1_confirmatory.py`
- `scripts/submit_publication_confirmatory.sh`

---

## 1. Three-Leg Comparison Design

For both **EvoSAX (`SimpleGA`)** and **QDax (`MAP-Elites`)**, Part 1 evaluates a three-leg comparison within every `(cell, seed)` block using identical initial populations and archive centroids:

| Library | Leg A (Direct Upstream) | Leg B (MalthusJAX Adapter) | Leg C (MalthusJAX Native Mimic) |
| :--- | :--- | :--- | :--- |
| **EvoSAX** | `evosax_direct`: Upstream `evosax.algorithms.SimpleGA` inside a `jax.lax.scan` loop (`crossover_rate=0.3`, `elite_ratio=0.16666666`) | `evosax_adapter`: `Composer` `backend="evosax"` (`EvosaxEngineAdapter`) | `evosax_native`: `Composer` `backend="malthusjax"` (`GeneticEngine` with `evosax_mimic_selection`, `evosax_uniform_crossover`, `evosax_gaussian`) |
| **QDax** | `qdax_direct`: Upstream `qdax.core.map_elites.MAPElites` + `MixingEmitter` inside a `jax.lax.scan` loop (`C=100`, `sigma=0.1`, `var_pct=0.5`) | `qdax_adapter`: `Composer` `backend="qdax"` (`QDaxEngineAdapter`) | `qdax_native`: `Composer` `backend="malthusjax"` (`MapElitesEngine` with `QDAXReplicaMixingEmitter`) |

- **Contrast B − A (Adapter Fidelity)**: Because Leg A and Leg B share the exact PRNG split (`k_loop, k_init, _ = jax.random.split(PRNGKey(seed), 3)`) and initial buffers, they achieve **bit-for-bit trajectory identity (`max_abs_diff == 0.0`)**, isolating any adapter compilation or execution overhead.
- **Contrast C − B & C − A (Native Equivalence & Speedup)**: Leg C uses MalthusJAX's native `GeneticEngine` / `MapElitesEngine` with pre-allocated `ResourceMapper` key splitting (`fold_in`). Paired TOST equivalence tests whether Leg C achieves statistically equivalent solution quality and archive diversity while delivering faster steady-state GPU execution.

---

## 2. Experimental Grids (`smoke`, `pilot`, `full`)

| Stage | BBOB Functions (All 5 Canonical Groups) | Instances | Dimensions ($D$) | Population ($P$) | Generations ($G$) | Disjoint Seeds | Cells | Total Condition Runs | Steady Repeats |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`smoke`** | `sphere` ($f_1$) | `[1]` | `[10]` | `[16]` | `[2]` | `1000..1001` ($n=2$) | 1 | 12 | 2 |
| **`pilot`** | `sphere` ($f_1$), `rastrigin` ($f_3$), `rosenbrock` ($f_8$), `ellipsoidal_rotated` ($f_{10}$), `schaffers_f7` ($f_{17}$), `gallagher_101_me` ($f_{21}$) | `[1]` | `[10, 50]` | `[128]` | `[50, 150, 500]` | `1000..1019` ($n=20$) | **36** | **4,320** | 5 |
| **`full`** | All 6 functions above | `[1, 2]` | `[10, 20, 50]` | `[128]` (+ `[512]` sensitivity) | `[50, 150, 500]` | `2000..2099` ($n=100$) | **108** | **64,800** | 5 |

---

## 3. Phase-Separated AOT Compilation & Retained-Executable Timing Protocol

1. **Cell Boundary Cold Compilation (Once per `(cell, condition)`)**:
   - `jax.clear_caches()` is called at the start of every cell.
   - `setup_seconds`: Wall time to construct the strategy, operators, evaluator, and adapter/engine objects.
   - `compile_seconds`: Wall time for explicit AOT lowering and XLA compilation (`.lower(*args0).compile()` or `engine.prepare_run(inputs0).executable`).
2. **Per-Seed Execution & Steady-State Timing (Counterbalanced Across Conditions)**:
   - Condition execution order rotates deterministically per seed block (`shift = seed_idx % 6`).
   - `init_state_seconds`: Time to construct and synchronize the seed's initial device state (`block_until_ready`).
   - `first_call_seconds` (`phase = "prepared_first_execution"`): First invocation of the retained compiled executable on a fresh device copy of the seed's initial state; extracts all solution quality and QD archive metrics.
   - `warmup_seconds` (`phase = "warmup"`): 1 discarded warmup execution on a fresh device copy of the seed's initial state.
   - `steady_repeat_seconds` (`phase = "prepared_execution_repeat"`): $R=5$ synchronized steady-state executions, each preceded by an out-of-timer device buffer copy and bounded by `block_until_ready(..., effects=True)`.
   - `repeat_output_agreement`: Verifies bit-for-bit equality of flattened PyTree outputs across all repeats.

---

## 4. Output Directory Schema

- `manifest.json`: Hardware, Git SHA, package versions, shard index, active grid, and timing protocol.
- `runs.csv`: Flat per-run table (1 row per `cell × seed × condition`) with all quality, QD (`qd_score`, `coverage`, `max_fitness`, `archive_entropy`, `occupied_cells`), compile, and steady-state timing metrics.
- `timings.csv`: Flat per-repeat table (1 row per `cell × seed × condition × repeat_idx`) with `prepared_first_execution`, `warmup`, and `prepared_execution_repeat` timings and determinism flags.
- `pilot_analysis.json`: Pooled and cell-by-cell paired TOST statistics, Holm–Bonferroni and Benjamini–Hochberg adjusted $p$-values, margin sensitivity ($\Delta \in \{0.025, 0.05, 0.10\}$), cell-level power sizing ($n_{\text{req}}$ at 80% and 90% power), steady-state speedup ratios, and 108-cell GPU-hour budget projections.
- `summary.json`: Campaign completion ledger + embedded `pilot_analysis` and block index.
- `<cell_id>/block.json`, `<cell_id>/<condition>.json`, `<cell_id>/initial_population.npy`, `<cell_id>/centroids.npy`.
