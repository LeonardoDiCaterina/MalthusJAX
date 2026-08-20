# `malthusjax.core` — Reference

Scope: `malthusjax.core.base`, `malthusjax.core.random`, `malthusjax.core.genome.*`, `malthusjax.core.fitness.*`. Every claim below is traceable to source (code + tests where a docstring makes a behavioral claim). Sections marked `UNCONFIRMED` require verification against the `engine`/`operators` modules, not `core` alone. This file supersedes prior narrative descriptions of these modules; see "Known Issues With Current Documentation" at the end for what it corrects.

---

## `malthusjax.core.base`

### `DistanceMetric`
Namespace of string constants: `HAMMING`, `EUCLIDEAN`, `MANHATTAN`. Not all genome types implement all three (see per-genome tables below).

### `BaseGenome`
**Purpose:** Abstract, immutable, JAX PyTree–compatible genome representation. Single-genome methods compose with `jax.vmap` to implement population-level operations via the Struct-of-Arrays (SoA) pattern.

**Public API:**
- `random_init(key, config) -> Genome` — abstract, must be implemented per subclass
- `distance(other, metric) -> scalar` — abstract
- `autocorrect(config) -> Genome` — abstract; re-enforces domain constraints, typically called post-mutation/crossover
- `size` / `shape` — abstract properties
- `from_tensor(arr, config=None) -> Genome` — abstract; wraps a batched array with no validation, kept JIT-traceable
- `copy() -> Genome` — deep-copies all array leaves via `tree_map`, to avoid JAX buffer-donation errors across multi-seed/multi-pipeline runs
- `create_population(key, config, pop_size) -> Genome` (batched) — splits `key` into `pop_size` subkeys, `vmap`s `random_init`
- `__getitem__` / `__iter__` — behavior depends on whether the instance is batched (population) or single, and on the `subscriptable` flag (see per-genome table; this flag is **not uniformly defaulted** across subclasses)

**Invariants:**
- Subclasses without domain-specific structure conventionally store their primary payload in a `.values` attribute — this is convention, not enforced by `BaseGenome` itself, since Tier-3 vectorization treats subclasses as opaque PyTrees.

### `BasePopulation[G]`
**Purpose:** Struct-of-Arrays container: `genes` holds one batched genome instance (leading dim = population size N), `fitness` is a `(N,)` array, `config` is static, `info` is a free-form dict.

**Public API:**
- `copy()` — deep-copy, same rationale as `BaseGenome.copy()`
- `from_array(arr, config, genome_cls, axis=0)` — builds a population from a raw array along a given axis; initializes `fitness` to `-inf`
- `spawn_offspring(new_genes, fitness=None, info=None)` — if `fitness` is omitted, allocates a NaN vector of the correct length rather than a real value
- `__getitem__` — int key returns an unwrapped single genome; slice/mask returns a sub-population (including sliced `info` arrays, non-array `info` entries pass through unchanged)
- `__iter__` — Python-side, documented as slow; the docstring explicitly recommends `jax.vmap`/`jax.lax.scan` instead for real computation
- `autocorrect(config)` — vmaps genome-level `autocorrect` over the whole population
- `distance_matrix(metric=EUCLIDEAN)` — nested double `vmap`, returns full `(N,N)` matrix, not a Python loop

**Notes:** Docstring states that as of v2.0, the generic `P` (population) type parameter and an explicit `GENOME_CLS` class property were removed from this base class. **This claim does not hold uniformly across concrete genome population subclasses** — see Cross-Module Notes.

---

## `malthusjax.core.random`

**Purpose:** Centralizes PRNG key construction across backends and provides legacy-key detection.

**Public API:**
- `PRNGImpl` (enum): `THREEFRY` (`threefry2x32`, default), `PHILOX` (`philox4x32_10`), `RBG`, `UNSAFE_RBG`
- `resolve_prng_impl(name)` — accepts short aliases, full backend strings, an enum member, or `None` (→ default); raises `ValueError` listing valid choices otherwise
- `create_key(seed, impl=None)` — tries `jax.random.key(seed, impl=...)` first; falls back to legacy `jax.random.PRNGKey(seed)` if the new API is absent or rejects `impl=`. Fallback due to a missing API emits `DeprecationWarning`; fallback due to a rejected `impl=` kwarg emits `RuntimeWarning` — these are two distinct warning types for two distinct failure modes.
- `_is_legacy_prngkey(key)` — **heuristic**: `dtype == uint32 and size == 2`. Documented explicitly as a heuristic, not a guaranteed classifier.
- `is_new_style_key(key)` — negation of the above
- `validate_key(key, context="")` — warns (`DeprecationWarning`, `stacklevel=3`) on legacy keys; docstring states it "should be called at engine boundaries (e.g. `init_state`)"

**UNCONFIRMED:** Whether any `engine.init_state()` actually calls `validate_key()`. This file only documents that it's *intended* to be called there — that is a design note in this module, not an observed behavior. Do not describe key validation as automatic until the `engine` module is audited.

---

## `malthusjax.core.genome`

### Binary (`binary_genome.py`)
| | |
|---|---|
| `BinaryGenomeConfig` | `shape: (1,)` default (chosen explicitly to avoid accidental scalar genomes); `length: int\|None` legacy alias, **deprecated**, overrides `shape` when set; `p: float = 0.5` Bernoulli bit-1 probability; `dtype: int32` default |
| `BinaryGenome.values` | `{0,1}` integers, not float |
| `BinaryGenome.subscriptable` | defaults `True` |
| `.distance()` | `"hamming"` or `"euclidean"` only; raises on anything else |
| `.to_int(msb_first=True)` | positional-weight conversion; result **may exceed native int precision** for long genomes |
| `.autocorrect()` | clips to `[0,1]` + dtype cast — a safety net for float-drift after mutation/crossover, not a mutation mechanism itself |
| `BinaryPopulation` | docstring states it is (as of v2.0) a strongly-typed alias of `BasePopulation[BinaryGenome]` with no mechanic overrides |

### Real (`real_genome.py`)
| | |
|---|---|
| `RealGenomeConfig` | `shape: ()` default (**scalar shape** — differs from Binary's deliberate `(1,)` default; no equivalent "avoid scalar" note here); `bounds: (-inf, inf)` default; `dtype: float32` default |
| **Bounds enforcement** | Enforced only at initialization (`random_init`). **Not automatically enforced after mutation/crossover.** Docstring is explicit: caller must either enable clipping in the mutation operator, call `.autocorrect()` manually, or accept and handle out-of-bounds values downstream. |
| `RealGenome.subscriptable` | defaults `True` |
| `.distance()` | `"euclidean"`, `"manhattan"`, or an **approximate** `"hamming"` (values differ if they exceed 1% of the observed value range — not a true Hamming distance on discretized reals) |
| `.magnitude()` / `.normalize()` | L2 norm; `.normalize()` uses `jnp.where`/`jnp.maximum` guards to avoid div-by-zero under JIT |
| `.add_noise(key, noise_std=0.1)` | Gaussian jitter, documented as usable as a mutation primitive |
| `RealPopulation` | Same "v2.0 thin alias" pattern as `BinaryPopulation`. Contains a `TODO` comment: multidimensional genome population shapes (non-1D genomes) are **not yet supported** — `init_random` takes a scalar `size`, not arbitrary shape tuples. |

### Categorical (`categorical_genome.py`)
| | |
|---|---|
| `CategoricalGenomeConfig` | `num_categories: int` — **required, no default**; `shape: ()` default (scalar); `dtype: int32` |
| `CategoricalGenome` | **Does not declare a `subscriptable` field at all** (unlike Binary/Real, which both default it to `True`). Under `BaseGenome`'s `getattr(self, "subscriptable", False)` fallback, this means single-instance indexing/iteration is **disabled by default** for `CategoricalGenome`, asymmetric with the other two genome types. |
| `.distance()` | Hamming, Euclidean, or Manhattan — all three, unlike Binary (two) or Real (two + an approximation) |
| `.is_permutation()` | JIT-safe check via `jnp.unique(..., fill_value=-1)` + sentinel absence check |
| `.to_permutation(config)` | `argsort`-based — deterministic, not a random shuffle |
| `.swap_positions`, `.count_category` | straightforward index-swap / count helpers |

### Linear / Linear GP (`linear_genome.py`, `linear_gp_evaluator.py`)
**Explicitly flagged in the module's own docstring as WIP and orphaned:**
> "This genome is currently an orphaned experimental feature. There are no matching crossover or mutation operators implemented for it in the `operators/` module, meaning it cannot currently be used in a standard evolutionary loop."

- Encodes programs as `(ops, args)` pairs with an enforced DAG constraint: instruction `i` may only reference external inputs or **prior** instructions, preventing cycles.
- `.autocorrect()` re-clips both opcodes and per-row argument index limits to restore DAG validity after any out-of-range mutation.
- `.render()` produces a human-readable assembly-style dump of the program.
- `LinearPopulation` declares a `GENOME_CLS: ClassVar` — **this contradicts** the `core_base.py` claim that `GENOME_CLS` was removed as of v2.0 (see Cross-Module Notes).
- `linear_gp_evaluator.py` implements the interpreter (arithmetic/logical/ternary ops dispatched via `jax.lax.switch`) but is built entirely on `LinearGenome`, so it inherits the same "cannot be used in a standard evolutionary loop" limitation.

---

## `malthusjax.core.fitness`

### Base (`fitness/base.py`)
- `BaseEvaluatorConfig.maximize: bool = False` — **the framework-wide contract**: `False` means the value `evaluate()` returns should be *lower-is-better* ("follows evosax convention"). Every evaluator below is checked against this contract.
- `BaseEvaluator.evaluate(genome) -> scalar` — abstract, single genome
- `BaseEvaluator.evaluate_population(population)` — `vmap`s `evaluate` over `population.genes`
- `BaseEvaluator.f_opt` / `.x_opt` — return `None` in the base class; concrete evaluators with known optima override them (confirmed: `BBOBEvaluator` does)
- `StochasticEvaluator` — adds an `rng` argument; `evaluate_population` raises `ValueError` if `rng is None`; splits one subkey per individual via `jax.random.split`
- `dispatch_evaluate_population(evaluator, population, key=None)` — routes to the stochastic path via `isinstance(evaluator, StochasticEvaluator)`

### `BBOBEvaluator` (`bbob_evaluator.py`)
- Wraps `evosax.problems.BBOBProblem`. Name aliasing table maps ~30 lowercase/hyphenated BBOB function name variants to canonical evosax names.
- **`evaluate()` and `evaluate_population()` both construct a fixed `jax.random.PRNGKey(0)` internally** rather than accepting a caller-supplied key — evaluation is deterministic regardless of any RNG state the caller has, for a given problem instance/seed.
- `f_opt`/`x_opt` are overridden and delegate directly to the underlying evosax problem's known optimum.
- Module contains explicit `TODO` comments: intent to migrate off evosax's problem registry to `bbobax`, and to add a separate `gymnax`-based adapter for RL problems. These are roadmap notes, not implemented capabilities.

### Binary evaluators (`binary_evaluators.py`)
- `BinarySumEvaluator` (OneMax): `evaluate()` returns `zeros_count` when `maximize=False` — correctly lower-is-better per the framework contract.
- `KnapsackEvaluator`: computes `total_value - penalty` (a profit-like quantity, higher=better), then returns `value if maximize else -value` — correctly converts to lower-is-better under the default. Penalty is a linear excess-weight term (`jax.lax.select`-free, pure `jnp.maximum` arithmetic, XLA-safe). Three constructors: `create_random_problem` (returns just a config), `create_synthetic` (full evaluator, random weights/values), `create_from_data` (full evaluator from caller-supplied arrays).

### Real evaluators (`real_evaluators.py`)
- `SphereEvaluator`, `GriewankEvaluator`, `BoxEvaluator`: each computes a cost-like quantity (sum of squares / Griewank combination / distance+penalty — lower=better by their own docstrings), then returns `raw if maximize else -raw`.
- **This is verified, intentional behavior** — `test_real_evaluators.py::test_sphere_optimization_direction_precision` explicitly asserts `min_result == -sphere_value` when `maximize=False`. The docstrings on all three evaluators, which describe the return value as "returned directly," **are wrong and should be corrected** (see Known Issues below) — the code and test agree with each other, not with their own docstrings.
- `BoxEvaluator.create_random_problem` samples a random target point and builds symmetric bounds around it (`margin = box_size / 4`).

### `TSPEvaluator` (`tsp_evaluator.py`)
- Exists in source. Uses `RealGenome` with random-key encoding (`argsort(genome.values)` decodes to a city permutation) — not `CategoricalGenome`, despite permutations being the categorical genome's stated use case.
- **Not present in the currently published Sphinx API index** for `malthusjax.core.fitness` (confirmed by comparing source directory contents against the published module list). This is an omission to fix, not a fabrication.

---

## Cross-Module Notes

- **`subscriptable` default is inconsistent across genome types**: `True` for `BinaryGenome` and `RealGenome`, unset (effectively `False`) for `CategoricalGenome`. `LinearGenome` does not use `.values` as a plain array at all (it's a computed property returning `(ops, args)`), so `subscriptable` doesn't apply the same way.
- **Default `shape` is inconsistent**: `BinaryGenomeConfig` defaults to `(1,)` specifically to avoid scalar genomes (stated rationale in its docstring); `RealGenomeConfig` and `CategoricalGenomeConfig` both default to `()`, a scalar shape, with no equivalent rationale given.
- **`GENOME_CLS` removal is not universal**: `core_base.py` states it was removed as of v2.0; `BinaryPopulation`/`RealPopulation` docstrings confirm this for themselves; `LinearPopulation` still declares `GENOME_CLS: ClassVar`. Likely explained by `LinearGenome` being flagged WIP/orphaned separately from the v2.0 cleanup, but this hasn't been confirmed with the maintainer.
- **Sign-convention split**: evaluators fall into two groups by what their raw internal value means — "profit-like" (`Knapsack`, `BinarySum`, higher=better) vs "cost-like" (`Sphere`, `Griewank`, `Box`, lower=better). Both groups use the *same* code shape (`raw if maximize else -raw`), which is correct for profit-like quantities and — per the sign-convention check above — also correct (if confusingly documented) for cost-like ones. Any new evaluator should be checked against a test, not just against an existing evaluator's code shape, before assuming the pattern generalizes.

---

## Known Issues With Current Documentation

Three concrete docstring errors, confirmed against source and tests, should be fixed before republishing: `SphereEvaluator`, `GriewankEvaluator`, and `BoxEvaluator` all claim their `evaluate()` method "returns the value directly" under minimization, when the code (and `test_real_evaluators.py`) confirm it actually returns the negated value in that case — the sign convention is correct, only the prose describing it is wrong. Separately, `tsp_evaluator.py` is a complete, working evaluator that is simply absent from the published Sphinx API index for `malthusjax.core.fitness`, an omission rather than an error. The `core_base.py` claim that `GENOME_CLS` was removed as of v2.0 is contradicted by `LinearPopulation`, which still declares it — likely because `LinearGenome` is explicitly marked WIP/orphaned and was exempted from that cleanup, but this hasn't been confirmed with the maintainer and shouldn't be asserted either way in prose yet. More generally, several structural asymmetries across the four genome types (default `shape`, default `subscriptable`) are currently undocumented anywhere; a reader moving from `BinaryGenome` to `CategoricalGenome` would reasonably assume indexing behavior carries over, and it doesn't. None of this file's content should be taken as a description of `malthusjax.engine`, `malthusjax.operators`, or `malthusjax.composer` behavior — claims like automatic PRNG key validation "at engine boundaries" are stated as intent in `core.random`'s own docstring but have not been verified against the engine source, and should not be presented as confirmed capability until that module is audited on the same terms as this one.
