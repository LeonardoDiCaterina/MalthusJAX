# Selection Module README Audit

**Workspace**: `/src/malthusjax/operators/selection/`  
**README**: `README.md`  
**Audit Date**: April 6, 2026  
**Status**: 🔴 **10 Discrepancies Found** (3 Critical, 4 High, 3 Medium)

---

## Overview

The Selection module README provides high-level architectural guidance on selection operators but contains several significant discrepancies with actual implementations. Key issues:

1. **Output contract incomplete**: Docs don't explain that `__call__` returns TWO arrays (parents + elites)
2. **Elite preservation mechanism undocumented**: `n_elites`, `set_n_elites()`, `get_elite_indices()` completely missing
3. **Key budgeting formula incorrect**: README claims `S * K` but code actually returns just `K`
4. **Critical operator behavior not documented**: `typed_keys` handling, population fallback mechanism, elite fusion optimization

---

## Detailed Findings

### 🔴 CRITICAL ISSUES

#### Issue #1: Output Contract Misrepresents Return Type

**Location**: README Sections 3, 8; BaseSelection.__call__ (base.py:373-384)

**README Claims**:
```
Section 3: "Selection produces index arrays via num_selections, which the engine 
           uses to gather parent genomes."
           
Section 8: "Selection accepts fitness values... and outputs integer indices of 
           shape (num_selections,)."
           
Benefits section: "Selection returns indices (not reordered genomes) to keep 
                  operations lightweight..."
```

**Actual Implementation**:
```python
def __call__(
    self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
) -> Tuple[chex.Array, chex.Array]:  # ← Returns TWO arrays!
    """Run a selection pass returning parents and elites."""
    fitness = getattr(population, "fitness", population)
    parent_idx = self._select(keys, fitness, config, **kwargs)
    elite_idx = self.get_elite_indices(fitness)  # ← Second array
    return parent_idx, elite_idx  # ← Tuple return
```

**Impact**: 
- Users expect single array output: `indices = selector(keys, population, config)`
- Actual output: `parent_idx, elite_idx = selector(keys, population, config)`
- This is a breaking API misunderstanding
- Engine docstring at line 251 of genetic_fastengine.py explicitly says: `__call__(key, fitness) -> (parent_idx, elite_idx)`

**Severity**: 🔴 **CRITICAL** — API contract completely wrong

---

#### Issue #2: Elite Preservation Mechanism Not Documented

**Location**: README (missing section); BaseSelection class (base.py:299-384)

**README** mentions selection **zero times** in these contexts:
- `n_elites` field
- `set_n_elites()` method
- `get_elite_indices()` method
- Elite preservation workflow
- `elite_idx` return value

**Actual BaseSelection Interface**:
```python
class BaseSelection(Generic[P, C]):
    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)
    n_elites: int = _field(pytree_node=False, default=0)  # ← Undocumented!
    
    def set_n_elites(self, n: int) -> "BaseSelection[P, C]":  # ← Undocumented!
        """Set elite count for preservation (called once at engine init)."""
        return ...
    
    def get_elite_indices(self, fitness: chex.Array) -> chex.Array:  # ← Undocumented!
        """Return indices of the top n_elites individuals."""
        if self.n_elites == 0:
            return jnp.zeros(0, dtype=jnp.int32)
        pop_size = fitness.shape[0]
        if self.n_elites >= pop_size:
            return jnp.arange(pop_size, dtype=jnp.int32)
        return jnp.argpartition(-fitness, self.n_elites)[: self.n_elites]
```

**Engine Usage** (genetic_fastengine.py:750):
```python
selection = selection.set_n_elites(params.elitism)  # ← Engine sets elites
```

**Impact**: 
- Users don't know selection returns elites separately
- Users don't know how to control elite preservation
- `set_n_elites()` method completely invisible to documentation readers
- Elite tracking semantics (O(N) argpartition for tournament/roulette) undocumented

**Severity**: 🔴 **CRITICAL** — Core mechanism completely hidden

---

#### Issue #3: Key Budgeting Formula Incorrect

**Location**: README Section 1; BaseSelection.num_keys() (base.py:346-348)

**README Claims**:
```
Table — Key budgeting formula

| sel_keys_needed | S * K |

Why static budgeting matters:
- Enables one-time host-side split and deterministic slicing.
- Avoids dynamic allocations...

Where:
- S = num_selections (per call)
- K = num_keys_per_atomic_operation
- sel_keys_needed = S * K
```

**Actual Implementation**:
```python
def num_keys(self, input_shape: Tuple[int, ...]) -> int:
    """Total keys needed for one selection pass (typically ≤ atomic_keys)."""
    return self.num_keys_per_atomic_operation  # ← Just K, not S * K!
```

**All Three Operators Confirm This**:
```python
# tournament.py, roulette.py, elite_pool.py all define:
@property
def num_keys_per_atomic_operation(self) -> int:
    return 1  # ← Single key needed, regardless of num_selections
```

**Why This Matters**:
- README suggests keys scale with `num_selections`: `sel_keys_needed = S * K`
- Actual implementation: keys are constant at `K` (single key shared for all selections)
- Inside `_select()`, a single RNG key is split/reused internally:
  ```python
  random_selections = jax.random.randint(
      rng, shape=(self.num_selections, self.tournament_size), ...
  )
  ```
- This is correct behavior (single key sufficient for vectorized operations)
- But README completely misrepresents the formula

**Impact**: 
- Users get wrong mental model of key budgeting
- Developers following README formula would over-allocate keys
- Confusion about static RNG budgeting workflow

**Severity**: 🔴 **CRITICAL** — Fundamental algorithm documentation wrong

---

### 🟠 HIGH PRIORITY ISSUES

#### Issue #4: `typed_keys` Behavior Completely Undocumented

**Location**: README (zero mentions); BaseSelection class (base.py:326-330)

**README Never Explains**: 
- What `typed_keys` means
- Why it matters
- How it affects operator behavior
- When it's set and when it matters

**Actual Use in Code**:
```python
# elite_pool.py _select() method:
py
if self.typed_keys:
    rng = keys if keys.ndim == 0 else keys[0]
else:
    rng = keys if keys.ndim <= 1 else keys[0]
```

```python
# tournament.py _select() method (same pattern):
if self.typed_keys:
    rng = keys if keys.ndim == 0 else keys[0]
else:
    rng = keys if keys.ndim <= 1 else keys[0]
```

**Actual Behavior**:
- `typed_keys=True`: New-style JAX PRNG (simple scalars)
- `typed_keys=False`: Legacy uint32[2] pairs
- Key extraction logic branches based on dimensionality
- Engine sets this via `set_typed_keys()` based on PRNG backend

**Impact**: 
- Developers implementing new operators need guidance on typed_keys
- Unclear why this branching logic is needed
- Checklist (item #6) mentions implementing operators but doesn't mention typed_keys at all

**Severity**: 🟠 **HIGH** — Implementation details unclear for developers

---

#### Issue #5: ElitePoolSelection.__call__ Optimization Undocumented

**Location**: README (generic description at lines 73-91); elite_pool.py __call__ (lines 127-181)

**README Says**:
```
"Elite pool selection, the algorithm:
1. Identifies the top elite_k individuals by fitness
2. For each selection, uniformly randomly picks from this elite pool"
```

**Actual __call__ Implementation** (elite_pool.py:127-181):
```python
def __call__(
    self,
    keys: chex.Array,
    population: P,
    config: Optional[C] = None,
    **kwargs: Any,
) -> Tuple[chex.Array, chex.Array]:
    """Select parents from the elite pool and simultaneously identify elites.
    
    This implementation avoids a second O(N) scan by performing a single
    argpartition on the combined effect of elite_k and n_elites.
    """
    # Single argpartition for both parent pool AND elite indices
    k = min(max(self.elite_k, self.n_elites), pop_size)
    
    if k >= pop_size:
        top_k_idx = jnp.arange(pop_size)
    else:
        top_k_idx = jnp.argpartition(-fitness, k)[:k]  # ← Single O(N) pass!
    
    # Then extract pool and elites from single result
    if self.n_elites == 0:
        pool = top_k_idx[:pool_k]
        elite_idx = jnp.zeros(0, dtype=jnp.int32)
    elif self.elite_k == self.n_elites:
        pool = top_k_idx[:pool_k]
        elite_idx = top_k_idx[: self.n_elites]
    else:
        sorted_within = jnp.argsort(-fitness[top_k_idx])  # ← Nested sort!
        sorted_top_k = top_k_idx[sorted_within]
        pool = sorted_top_k[:pool_k]
        elite_idx = sorted_top_k[: self.n_elites]
    
    return parent_idx, elite_idx
```

**Key Optimization Not Documented**:
- Fuses argpartition for both parent pool selection AND elite identification
- Avoids second O(N) scan (which tournament/roulette would do)
- Only works when `elite_k` and `n_elites` don't conflict
- Falls back to sorting when `elite_k != n_elites`

**Impact**: 
- Users don't know this optimization exists
- Users don't know when to use ElitePool vs other methods for elite preservation
- Performance characteristics completely undocumented
- Developers won't understand the conditional logic in code

**Severity**: 🟠 **HIGH** — Critical performance optimization hidden

---

#### Issue #6: Population Fallback Mechanism Not Documented

**Location**: README (missing); BaseSelection.__call__ (base.py:381)

**README Claims**:
```
"Selection accepts either a Population object (extracts .fitness) 
or a fitness array directly."
```

But then doesn't explain WHERE this works or WHAT it does.

**Actual Implementation**:
```python
def __call__(
    self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
) -> Tuple[chex.Array, chex.Array]:
    """Run a selection pass returning parents and elites."""
    fitness = getattr(population, "fitness", population)  # ← The magic!
    parent_idx = self._select(keys, fitness, config, **kwargs)
    elite_idx = self.get_elite_indices(fitness)
    return parent_idx, elite_idx
```

**What This Means**:
- If `population` has `.fitness` attribute: use it
- If `population` is a fitness array directly: use it as-is
- Only works in `__call__()`, NOT in `_select()`
- Developers implementing custom selection must handle population fallback too

**Missing Details**:
- This works in `__call__()` but _select() receives fitness directly
- The pattern `getattr(population, "fitness", population)` should be explained
- Elite pool's `__call__()` also uses this pattern independently
- Users wondering "can I pass fitness directly?" need clear guidance

**Impact**: 
- Flexible API completely undocumented
- Developers don't know if they need to implement both code paths
- Users don't know this trick exists

**Severity**: 🟠 **HIGH** — Hidden API flexibility

---

#### Issue #7: Roulette Configuration Parameters Under-Documented

**Location**: README (Roulette section); roulette.py (lines 1-70)

**README Documents**:
- `temperature` ✓
- `use_gumbel_trick` ✓
- `chunk_size` ✓

**BUT Critical Interactions Not Explained**:

1. **Gumbel-Max Optimization Condition Not Clear**:
   ```python
   if self.use_gumbel_trick and self.num_selections == pop_size:
       # Fast parallel path
   else:
       # Slow categorical path
   ```
   README says "If True (default), uses the Gumbel-Max trick **when** `num_selections == population_size`"
   This condition is critical for understanding performance! Users need to know:
   - Gumbel trick only works for full population replacement
   - If num_selections < pop_size, falls back to categorical
   - This affects convergence and diversity significantly

2. **chunk_size Only Used With Gumbel Trick**:
   ```python
   if self.use_gumbel_trick and self.num_selections == pop_size:
       if pop_size <= self.chunk_size:
           # Fast monolithic path
       else:
           # Chunked scan path
   ```
   README documents chunk_size but doesn't explain:
   - It's ONLY used with Gumbel trick and full replacement
   - Chunking is memory vs speed tradeoff
   - Users need to understand when to tune this

3. **Use Case Guidance Missing**:
   - When Gumbel trick is active (speeds up full replacement)
   - When to use categorical (more memory efficient)
   - How chunk_size affects performance on different hardware

**Impact**: 
- Users don't understand performance characteristics
- Unexpected fallback to slower categorical path
- Tuning advice incomplete

**Severity**: 🟠 **HIGH** — Critical performance tuning guidance missing

---

### 🟡 MEDIUM PRIORITY ISSUES

#### Issue #8: TournamentSelection Defaults Ambiguous

**Location**: README Section 3, Docstring; tournament.py

**README Says**:
```
"tournament_size=2: Mild selection pressure, high diversity
tournament_size=3-5: Balanced (recommended for most problems)
tournament_size=7+: Strong selection pressure, lower diversity
Default: 3 (recommended for general-purpose problems)."
```

**But Registry Default** (`__init__.py:24`) is:
```python
("tournament", TournamentSelection, {"num_selections": 4, "tournament_size": 3})
```

**Code Default** (tournament.py:73):
```python
tournament_size: int = _field(pytree_node=False, default=3)
```

**Issues**:
1. Three different places define "default" (README, registry, code)
2. All agree on 3, but why three places?
3. No explanation of why 3 is optimal (theoretical backing)
4. "Recommended" is vague (for what class of problems?)

**Impact**: 
- Minor confusion about defaults
- No theoretical justification provided

**Severity**: 🟡 **MEDIUM** — Clarification needed

---

#### Issue #9: Developer Checklist Incomplete

**Location**: README Section 9

**Checklist Says** (items 1-6):
```
- [ ] Define num_keys_per_atomic_operation (0 for deterministic, ≥1 for stochastic).
- [ ] Implement _select(keys, fitness, config) as a pure function.
- [ ] Return an integer jnp.ndarray of indices with shape (num_selections,)
- [ ] Document the selection logic and any required config attributes.
- [ ] Add unit tests for correctness and shape contracts.
- [ ] Verify num_keys() returns the correct total key budget.
```

**Missing Items**:
1. Handle `typed_keys` branching (required for extraction logic)
2. Consider overriding `get_elite_indices()` if optimization possible (like ElitePool)
3. Consider overriding `__call__()` if elite+parent fusion is possible
4. Handle population fallback mechanism in `__call__()` if custom
5. Document `set_input_length()` and `set_n_elites()` interactions
6. Provide shape contract for elite indices output
7. Document what happens when `n_elites != 0`

**Impact**: 
- Developers implementing new operators miss critical patterns
- Incomplete understanding of optional optimizations

**Severity**: 🟡 **MEDIUM** — Good practices underdocumented

---

#### Issue #10: Elite Indices Not Mentioned in Return Contract

**Location**: README Technical Summary (Section 8)

**Current Text**:
```
"Input/Output Contract: Selection accepts fitness values... 
and outputs integer indices of shape (num_selections,)."
```

**Should Be**:
```
"Input/Output Contract: Selection accepts fitness values (or Population objects)...
and outputs:
- parent_idx: integer array of shape (num_selections,) for parent gathering
- elite_idx: integer array of shape (n_elites,) for elite preservation"
```

**Missing Details**:
- Elite indices can be empty (shape (0,)) when `n_elites=0`
- Elite indices guaranteed to be top N by fitness (not random)
- Elite indices always come from distinct argpartition (ElitePool may fuse)

**Impact**: 
- Users confused about return shape contracts
- Training data misleading

**Severity**: 🟡 **MEDIUM** — Incomplete specification

---

## Summary Table

| # | Issue | Severity | Category | Location |
|---|-------|----------|----------|----------|
| 1 | Return type misrepresents output | 🔴 CRITICAL | API Contract | Sections 3, 8 |
| 2 | Elite preservation undocumented | 🔴 CRITICAL | Missing Feature | Entire module |
| 3 | Key budgeting formula incorrect | 🔴 CRITICAL | Algorithm | Section 1 |
| 4 | typed_keys behavior undocumented | 🟠 HIGH | Implementation | Missing |
| 5 | ElitePool optimization hidden | 🟠 HIGH | Performance | Section 3 |
| 6 | Population fallback undocumented | 🟠 HIGH | API Flexibility | Section 3 |
| 7 | Roulette config interactions unclear | 🟠 HIGH | Tuning Guidance | Section 3 |
| 8 | Tournament defaults ambiguous | 🟡 MEDIUM | Clarification | Section 3 |
| 9 | Developer checklist incomplete | 🟡 MEDIUM | Best Practices | Section 9 |
| 10 | Elite indices not in return contract | 🟡 MEDIUM | Specification | Section 8 |

---

## Recommended Action Priority

### Phase 1 (Critical - Fix API Contract)
1. Add "Selection Returns Tuple: (parent_idx, elite_idx)" section
2. Fix key budgeting formula to match actual code (just K, not S*K)
3. Document n_elites, set_n_elites(), get_elite_indices()

### Phase 2 (High - Complete Implementation Docs)
4. Document typed_keys branching logic and when it's set
5. Explain ElitePool's fused argpartition optimization
6. Clarify population fallback mechanism with examples
7. Document Roulette's conditional Gumbel-Max behavior

### Phase 3 (Medium - Improve Quality)
8. Add theoretical justification for default tournament_size=3
9. Expand developer checklist with missing items
10. Update technical summary with complete return contract

---

## Code Quality Notes

✅ **Strengths**:
- All three operators correctly implement `_select()` as pure functions
- Output indices are properly typed (int32/int64)
- Docstrings are comprehensive and well-written (elite_pool, tournament, roulette)
- Edge cases handled (pop_size < elite_k, n_elites=0, etc.)

⚠️ **Issues**:
- README doesn't match implementation (API contract confusion)
- Silent overrides in ElitePoolSelection.__call__ not explained
- typed_keys branching pattern repeated in three implementations (DRY violation?)
- No cross-references between README and docstrings
