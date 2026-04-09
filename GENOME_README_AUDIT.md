# Genome Module README Audit & Discrepancy Report

## Summary
Thorough review of `/src/malthusjax/core/genome/README.md` against actual implementation files. Found **2 major discrepancies**, several **minor inconsistencies**, and areas that could benefit from clarification.

---

## DISCREPANCY #1: Distance Method Default Parameters Inconsistent ⚠️

### README Claims (Section: "Standard vs. Extended Metrics"):
> "Domain-specific metric classes (e.g. `RealDistanceMetric`) can extend `DistanceMetric` to add more specialized metrics (cosine, normalized Lp, etc.)."

And show generic usage:
> "`BaseGenome.distance(...)` polymorphically"

### Actual Implementation:

**RealGenome** (line ~170):
```python
def distance(self, other: BaseGenome, metric: str = "euclidean") -> chex.Numeric:
```
- Default: `"euclidean"`

**BinaryGenome** (line ~165):
```python
def distance(self, other: BaseGenome, metric: str = "hamming") -> chex.Numeric:
```
- Default: `"hamming"`

**CategoricalGenome** (line ~65):
```python
def distance(self, other: BaseGenome, metric: str = DistanceMetric.HAMMING) -> chex.Numeric:
```
- Default: `DistanceMetric.HAMMING` (which is `"hamming"`)

### The Problem:
1. **Inconsistent default values across genome types** — different genomes have different default metrics
   - RealGenome: `"euclidean"` (a string)
   - BinaryGenome and CategoricalGenome: `DistanceMetric.HAMMING` (a class attribute)
2. **README doesn't mention this variation** — it implies all genomes follow `DistanceMetric` constants, but RealGenome uses raw strings
3. **Type inconsistency** — RealGenome defaults are strings (`"euclidean"`), others use `DistanceMetric` constants

### Status: 🟡 **DOCUMENTATION MISSING / IMPLEMENTATION INCONSISTENCY**

### Recommendation:
Either:
1. Standardize all distance methods to use `DistanceMetric` constants (preferred for consistency)
2. Update README to explicitly document the per-genome default metric choices
3. Add typed enum for metrics instead of string-based names

---

## DISCREPANCY #2: RealGenomeConfig "length" Backward Compatibility Not Clearly Explained ⚠️

### README Claims (Section: "Usage Example"):
```
Note: `BinaryGenomeConfig` also supports a legacy `length` keyword that is treated as
`shape=(length,)`. `BinaryGenomeConfig.shape` defaults to `(1,)` to avoid accidental
scalar genomes when a shape is omitted.
```

**Issue**: The README mentions this ONLY for BinaryGenomeConfig, but doesn't mention RealGenomeConfig's handling.

### Actual Implementation:

**BinaryGenomeConfig** (lines 45-55):
```python
shape: Tuple[int, ...] = _field(pytree_node=False, default_factory=lambda: (1,))
length: int | None = _field(pytree_node=False, default=None)

@property
def resolved_shape(self) -> Tuple[int, ...]:
    """Return the effective shape, honoring legacy `length` if present."""
    if self.length is not None:
        return (self.length,)
    return self.shape
```

**RealGenomeConfig** (lines 125-130):
```python
shape: Tuple[int, ...] = _field(pytree_node=False, default_factory=lambda: ())
bounds: Tuple[float, float] = _field(pytree_node=False, default=(-jnp.inf, jnp.inf))
dtype: type[jnp.floating[Any]] | jnp.dtype[jnp.floating[Any]] = _field(
    pytree_node=False, default=jnp.float32
)
```

**Critical difference**:
- **BinaryGenomeConfig**: Has `resolved_shape` property that honors legacy `length` parameter
- **RealGenomeConfig**: Has NO such property or legacy `length` parameter support
- Yet RealGenome docstring says shape can be shaped: "(shape=(5, 3) → 5×3 matrix)"

### Status: 🟡 **INCOMPLETE DOCUMENTATION**

### Recommendation:
1. Clarify that ONLY BinaryGenomeConfig supports the legacy `length` parameter
2. Document that RealGenomeConfig requires explicit `shape=` for multi-dimensional tensors
3. Note that default RealGenomeConfig `shape=()` creates scalar genomes (contradicts the "avoid scalar" philosophy mentioned for Binary)

---

## MINOR ISSUES & INCONSISTENCIES

### Issue A: Abstract Methods Not Fully Listed in README

**README** describes:
- `random_init()` ✓
- `distance()` ✓
- `autocorrect()` ✓
- `size` (property) ✓
- `shape` (property) ✓
- `from_tensor()` ✓

**Actually in BaseGenome** - ALL the above are correct.

**But README DOESN'T mention**:
- `__len__()` - implemented in BaseGenome
- `__getitem__()` - for indexing/slicing
- `__iter__()` - for iteration
- `create_population()` - class method for vmap initialization

These are all present in base.py but not documented in README. Users following docs won't know these exist.

### Status: 🟡 **INCOMPLETE DOCUMENTATION**

### Recommendation:
Add section on "Common Operations" or "Convenience Methods" covering:
- Length queries (`len(genome)`)
- Indexing and slicing (`genome[0]`, `genome[1:5]`)
- Iteration (`for value in genome`)
- Batched initialization (`BaseGenome.create_population()`)

---

### Issue B: RealGenome Default Shape Documentation Contradicts Implementation

**README** (RealGenomeConfig description):
```
Default: () (empty shape for backward compatibility)
```

**But usage section shows**:
```python
config = RealGenomeConfig(length=10, bounds=(-1.0, 1.0), dtype=jnp.float32)
```

**Problem**: 
- Default shape `()` creates a SCALAR genome
- Example uses `length=10` which doesn't exist in RealGenomeConfig
- This is copy-paste from an outdated API

### Status: 🔴 **BROKEN EXAMPLE CODE**

### Recommendation:
Update the usage example to use `shape=` correctly:
```python
config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)
```

---

### Issue C: RealGenome Multi-dimensional Tensor Discussion Is Misleading

**RealGenomeConfig docstring** includes:
```
- shape=(5, 3) → 5×3 matrix (15 total parameters)
- shape=(4, 4, 4) → 4×4×4 tensor (64 total parameters)
```

**But RealPopulation.init_random** has a TODO comment (line 280):
```python
# TODO: implement a more broad verison where size is actually shape for multidimensional genomes
```

**Implication**: Multi-dimensional genomes are documented as supported but admit they're not fully implemented.

### Status: 🟡 **MISLEADING DOCUMENTATION**

### Recommendation:
Either:
1. Add clear note that multi-dim support is experimental/incomplete
2. Remove the multi-dim examples and focus on 1D vectors
3. Implement the TODO for proper multi-dim support

---

### Issue D: CategoricalGenome Methods Not Documented in README

The README has NO section on CategoricalGenome. But implementation includes useful methods:
- `is_permutation()` - check if values form valid permutation
- `to_permutation()` - convert via argsort
- `swap_positions(pos1, pos2)` - exchange values at indices
- `count_category(category)` - count occurrences

**Status**: 🟡 **MISSING DOCUMENTATION**

### Recommendation:
Add a "Categorical Genomes" section covering:
- Use cases (permutation problems, categorical choices)
- Permutation helpers and how/when to use them
- Example: TSP with permutation checking

---

### Issue E: Population Methods Under-documented

**README** documents:
- `spawn_offspring()` mention ✓

**Missing documentation**:
- `__getitem__()` for slicing (fancy indexing, integer indexing both supported)
- `__iter__()` for iteration
- `from_array()` - build population from tensor
- `distance_matrix()` - compute pairwise distances
- `autocorrect()` - apply genome corrections to entire population

### Status: 🟡 **INCOMPLETE DOCUMENTATION**

### Recommendation:
Add subsection "Population Operations" with examples of:
```python
# Slicing
sub_pop = pop[10:20]  # get individuals 10-19
best = pop[0]  # get individual 0 (unwrapped to genome)

# Distance matrix
dists = pop.distance_matrix(metric="euclidean")

# Batched correction
corrected_pop = pop.autocorrect(config)
```

---

### Issue F: spawn_offspring() Behavior & NaN Sentinel Unclear

**README Comment**:
> "Passing `fitness=None` triggers allocation of a NaN vector of the appropriate length; supplying an array avoids the allocation cost when the values are immediately overwritten."

**Issues**:
1. Why use NaN sentinel? Not explained (prevents accidental fitness escapes?)
2. When should users pass NaN vs. actual values? Guidance is vague
3. The "hot-path" optimization comment mentions "FB-2" — what is this? (seems like internal ticket reference)

### Status: 🟡 **UNCLEAR SEMANTICS**

### Recommendation:
Clarify:
```python
# NaN signals "fitness not yet computed" for safety checks
offspring_pop = parent_pop.spawn_offspring(new_genes)
# NaN acts as sentinel; any downstream fitness usage will fail fast if not set

# For hot paths where immediate overwrite happens, skip NaN allocation:
dummy = jnp.zeros(n)
offspring_pop = parent_pop.spawn_offspring(new_genes, fitness=dummy)
# Immediately overwrite fitness in engine's reproduction phase
```

---

## CORRECT SECTIONS (✅ No Discrepancies Found)

1. **Extension Pattern** — Liskov substitutability, casting, generic populations ✓
2. **JAX Integration** — vmap/jit/immutability patterns ✓
3. **Struct-of-Arrays paradigm** — accurate description ✓
4. **Best practices checklist** — all patterns correctly described ✓
5. **BasePopulation slicing examples** — conceptually sound (though slicing API not fully documented) ✓
6. **BinaryGenomeConfig description** — accurate representation ✓
7. **RealGenome methods**: `normalize()`, `magnitude()`, `add_noise()` — all well documented ✓
8. **BinaryGenome methods**: `to_int()`, `count_ones()`, `flip_bit()` — all accurate ✓

---

## RECOMMENDED ACTIONS (Priority Order)

### 🔴 Critical:

1. **Fix RealGenomeConfig usage example** (line ~235)
   - Change `length=10` → `shape=(10,)`
   - Or add note that RealGenomeConfig doesn't support legacy `length` parameter

2. **Remove/clarify Multi-dimensional genome support**
   - Either implement the TODO or add deprecation warning

### 🟡 High:

3. **Standardize distance metrics** across all genome types
   - Use `DistanceMetric` constants consistently
   - Document default metrics per genome type in README

4. **Add CategoricalGenome section** with permutation helpers

5. **Document all public methods** missing from README:
   - `__len__()`, `__getitem__()`, `__iter__()`
   - `distance_matrix()` on populations
   - `create_population()` for vmap initialization

### 🟢 Medium:

6. **Clarify legacy `length` parameter** that only BinaryGenomeConfig supports

7. **Explain NaN sentinel behavior** in `spawn_offspring()` with safety rationale

8. **Add "Population Operations"** subsection with slicing/iteration examples

---

## Files Reviewed

- ✅ `src/malthusjax/core/genome/README.md` (full)
- ✅ `src/malthusjax/core/genome/real_genome.py` (full)
- ✅ `src/malthusjax/core/genome/binary_genome.py` (full)
- ✅ `src/malthusjax/core/genome/categorical_genome.py` (partial)
- ✅ `src/malthusjax/core/base.py` (BaseGenome, BasePopulation — lines 1-300)
- ✅ `src/malthusjax/core/genome/__init__.py` (current state)

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Critical Issues | 2 |
| High Priority | 3 |
| Medium Priority | 3 |
| Missing Sections | 2 |
| Broken Examples | 1 |
| Correct Sections | 8 |
| **Total Discrepancies** | **11** |
