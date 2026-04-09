# Crossover Module README Audit

**Workspace**: `/src/malthusjax/operators/crossover/`  
**README**: `README.md`  
**Audit Date**: April 6, 2026  
**Status**: 🟡 **7 Discrepancies Found** (2 Critical, 2 High, 3 Medium)

---

## Overview

The Crossover module README provides comprehensive architectural guidance on the 3-tier paradigm but contains several significant discrepancies with actual implementations. Key issues relate to:

1. **Injection mode undocumented** — Complete feature not mentioned despite having multiple implementations
2. **Mask semantics inconsistency** — Documentation vs actual code mixed in places
3. **SBX offspring semantics unclear** — Returns single genome, not two as default parameters suggest
4. **Evosax wrapper undocumented** — Alternative pattern not explained in README
5. **Developer checklist incomplete** — Missing guidance on key handling and injection mode patterns

---

## Detailed Findings

### 🔴 CRITICAL ISSUES

#### Issue #1: Injection Mode Completely Undocumented

**Location**: README (zero mentions); __init__.py (lines 8-10, 40-47); real.py (multiple _injection classes); evosax_crossover.py (lines 33-100)

**README Never Mentions**:
- What "injection mode" is
- Why it exists
- How it differs from standard mode
- When to use it
- How to implement it

**Actual Code Evidence**:
```python
# __init__.py exports multiple injection variants:
from .real import (
    BinomialCrossover_injection,
    BlendCrossover_injection,
    SimulatedBinaryCrossover_injection,
    UniformCrossover_injection as RealUniformCrossover_injection,
)

# Registry includes injection variants:
("uniform_real_injection", RealUniformCrossover_injection, {}),
("blend_injection", BlendCrossover_injection, {}),
("simulated_binary_injection", SimulatedBinaryCrossover_injection, {}),
("binomial_injection", BinomialCrossover_injection, {}),
```

**Injection Mode Pattern** (evident from real.py implementations):
```python
@struct.dataclass
class UniformCrossover_injection(
    BaseCrossover_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Uniform Crossover.
    Single key splits into (n_pairs * n_offspring) subkeys; jax.vmap(per_row) generates all masks
    in parallel, returning (n_pairs * n_offspring, d) flattened array for base wrapper to unfold.
    Trade-off: Full noise materialization enables reproducibility without key re-splitting.
    """
    
    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Generate all (pair, offspring) masks upfront. Shape: (n_pairs * n_offspring, d)."""
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n)
        def per_row(k: chex.PRNGKey) -> chex.Array:
            return jax.random.bernoulli(k, p=self.crossover_rate, shape=config.shape)
        return jax.vmap(per_row)(subkeys)  # (n, d) boolean masks
```

**EvosaxUniformCrossoverWrapper With injection_mode Support**:
```python
@struct.dataclass
class EvosaxUniformCrossoverWrapper(BaseCrossover[...]):
    """
    ...
    With ``injection_mode=True`` (default), the engine passes a single key and
    this operator splits internally for maximum performance.
    """
    injection_mode: bool = _field(pytree_node=False, default=True)

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        """Return key budget.
        With ``injection_mode=True``, always returns 1 (single key, split internally).
        """
        if self.injection_mode:
            return 1  # Single key, not pre-allocated
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation
```

**What Injection Mode Is** (from code patterns):
- Single PRNG key passed instead of pre-allocated key blocks
- Operator internally splits key for all (pair, offspring) combinations
- Full noise materialization upfront vs lazy generation
- Trade-off: Memory for determinism + flexibility

**Missing from README**:
- No mention of BaseCrossover_injection class
- No explanation of single-key vs pre-allocated paradigm
- No guidance on when to use injection mode
- No mention of _injection variants in registry
- No developer guidance on implementing injection operators

**Impact**: 
- Users don't know injection mode exists
- Developers don't know alternative pattern for operators
- Evosax wrapper's injection_mode parameter is invisible

**Severity**: 🔴 **CRITICAL** — Entire alternative architecture hidden

---

#### Issue #2: Unclear Offspring Semantics for SBX

**Location**: README "Available Crossover Operators" table; real.py SimulatedBinaryCrossover

**README Claims**:
```markdown
| Real / Exploitation | SBX | η=20-30 | Parent-centric, adaptive |

num_offspring : int, optional
    Number of distinct offspring produced per parent pair.
    Default: 2 (SBX naturally produces two offspring).
```

**What This Suggests**:
- SBX produces 2 offspring by default
- Each parent pair generates exactly 2 genomes
- This is inherent to SBX (not configurable)

**Actual Implementation** (real.py SimulatedBinaryCrossover):
```python
@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    num_offspring: int = struct.field(pytree_node=False, default=2)
    
    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:  # ← Returns SINGLE genome, not tuple!
        """
        Tier 1 — XLA-Fused SBX Kernel (Pure, returns single offspring).
        Per-offspring keys in base class ensure swap_mask differs across num_offspring calls,
        yielding distinct children from identical parents.
        ...
        Returns: Offspring RealGenome with (d,) clipped values
        """
        # ... computation ...
        return cast(RealGenome, cast(Any, p1).replace(values=final_values))
```

**Reality**:
- `_recombine_one()` returns SINGLE genome (not 2)
- `num_offspring=2` is just a default parameter
- Tier 3 calls `_recombine_one()` twice via vmap to produce 2 offspring
- The "2 offspring" is a convenience default, NOT inherent to algorithm
- User can set `num_offspring=1` to get single offspring

**What README Should Say**:
```markdown
num_offspring : int, optional
    Number of distinct offspring produced per parent pair.
    Default: 2 (common for SBX, but configurable).
    Note: Each vmap call to _recombine_one returns one offspring;
    num_offspring controls iteration count in Tier 3 orchestration.
```

**Confusion Points**:
- Readers think "SBX produces 2 offspring" (algorithm inherent)
- Actually: SBX produces 1 per call; num_offspring=2 means call twice
- This is a Tier 3 orchestration detail, not algorithm semantics

**Impact**: 
- Users confused about SBX producing 1 vs 2 offspring
- Users don't understand num_offspring is freely configurable
- May lead to incorrect assumptions about algorithm

**Severity**: 🔴 **CRITICAL** — Fundamental semantics misrepresented

---

### 🟠 HIGH PRIORITY ISSUES

#### Issue #3: Mask Semantics Inconsistency

**Location**: README line 53; real.py implementations

**README Says** (Section: "Mask Semantics"):
```markdown
All crossover operators follow the convention:
- mask=False → inherit from Parent 1
- mask=True → inherit from Parent 2

Implementation pattern:
offspring = jnp.where(mask, p2.values, p1.values)  # True -> p2, False -> p1
```

**But In Actual Code** (`real.py` BlendCrossover, line 199):
```python
def _recombine_one(self, p1, p2, noise_data, config, **kwargs):
    should_cross, random_vals = noise_data
    # ...
    offspring_values = cmin + random_vals * (cmax - cmin)
    min_b, max_b = config.bounds
    offspring_values = jnp.clip(offspring_values, min_b, max_b)
    
    # ← INVERTED! should_cross=True means use BLENDED values, not p2!
    final_values = jnp.where(should_cross, offspring_values, p1.values)
    # This breaks the documented convention where True→p2
```

**Similarly in SBX** (`real.py` SimulatedBinaryCrossover, line 435):
```python
should_cross, u, swap_mask = noise_data
# ...
c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
child_vals = jnp.where(swap_mask, c2, c1)  # ← swap_mask is local, not "p2 selection"

# Then:
final_values = jnp.where(should_cross, child_vals, p1.values)
# Again: True means use computed child, not p2!
```

**What's Really Happening**:
- Blend/SBX use mask semantics differently from Uniform/SinglePoint
- Uniform/SinglePoint: mask directly selects parent (True→p2, False→p1)
- Blend: mask gates whether to apply blend (True→use blend, False→use p1)
- SBX: mask gates whether to apply SBX (True→use sbx child, False→use p1)

**Impact**: 
- Developers implementing custom operators will use WRONG mask convention
- Tests verifying "all-true mask → offspring==p2" would FAIL for Blend/SBX
- Documentation is misleading for non-uniform operators

**Severity**: 🟠 **HIGH** — Mask convention overgeneralized

---

#### Issue #4: Evosax Wrapper Pattern Undocumented

**Location**: README (zero explanation); evosax_crossover.py (detailed implementation)

**README Never Explains**:
- EvosaxUniformCrossoverWrapper exists
- How it differs from standard operators
- Single-key vs pre-allocated modes
- When to use it (benchmarking? compatibility?)
- Implementation pattern and trade-offs

**Actual Implementation** (evosax_crossover.py):
```python
class EvosaxUniformCrossoverWrapper(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Evosax Compatibility Wrapper — Single-Key Mode.
    Consumes single key, splits internally in _cross_fused...
    Design trade-off: Dynamic key splitting vs. static shape stability.
    Use for: Benchmarking evosax compatibility; ablation studies; comparative evolution.
    """
    
    injection_mode: bool = _field(pytree_node=False, default=True)
    
    def _cross_fused(self, keys, p1, p2, config, generation=0):
        """Atomic Crossover Kernel (Single-Key Wrapper Pattern).
        Extracts a PRNG key and calls evosax.crossover directly.
        """
        if self.typed_keys:
            prng_key = keys.reshape(-1)[0]
        else:
            prng_key = keys.reshape((-1, keys.shape[-1]))[0]
        child_vals = evosax_crossover(prng_key, p1.values, p2.values, self.crossover_rate)
        return RealGenome.from_tensor(child_vals, config)
```

**Pattern Elements Not Documented**:
1. **Alternative implementation pattern**: Override `_cross_fused()` instead of `_generate_noise()` + `_recombine_one()`
2. **Dynamic key handling**: Different extraction logic for typed_keys vs legacy
3. **Evosax integration**: Shows how to wrap external implementations
4. **Use cases**: Benchmarking, compatibility validation

**Missing from README**:
- No mention of `_cross_fused()` method override pattern
- No explanation of when this differs from standard Tier 1-2-3
- No guidance on supporting external algorithms
- No typed_keys handling patterns documented

**Impact**: 
- Users can't understand EvosaxUniformCrossoverWrapper
- Developers don't know how to wrap external crossover implementations
- Alternative architecture pattern hidden

**Severity**: 🟠 **HIGH** — Important implementation pattern missing

---

### 🟡 MEDIUM PRIORITY ISSUES

#### Issue #5: Developer Checklist Incomplete

**Location**: README "Developer Checklist" section

**Missing Items**:
1. Handle `typed_keys` for PRNG format extraction (like selection module)
2. Understand num_offspring semantics (affects Tier 3 vmap nesting)
3. Consider injection mode alternative pattern
4. Document num_keys_per_atomic_operation correctly
5. Understand when mask semantics apply vs gates

**Current Checklist Missing**:
- No mention of typed_keys handling
- No guidance on num_offspring and how it affects Tier 3 orchestration
- No explanation of why different operators have different mask meanings (Uniform vs Blend vs SBX)
- No guidance on testing (e.g., "all-true mask → offspring==p2" only works for per-gene-selection operators)
- No mention of injection mode implementation pattern

**Content for Checklist** (should add):
```markdown
### Type Handling
- [ ] Understand typed_keys parameter (engine sets based on PRNG backend)
- [ ] Extract keys correctly for _generate_noise: `k1, k2 = keys[0], keys[1]`

### Offspring Semantics
- [ ] Understand num_offspring in Tier 3 vmap context (affects key reshaping, not algorithm)
- [ ] Each _recombine_one call produces ONE offspring (Tier 3 calls it num_offspring times)

### Mask Semantics
- [ ] Document whether your operator uses per-element mask selection (Uniform/SinglePoint)
  or gate-based selection (Blend/SBX)
- [ ] Per-element: `offspring = jnp.where(mask, p2, p1)` (True→p2)
- [ ] Gate-based: `offspring = jnp.where(gate, computed_values, p1)` (True→use computed)

### Optional Patterns
- [ ] Consider injection mode (`BaseCrossover_injection`) if single-key is beneficial
- [ ] Consider _cross_fused override (like EvosaxUniformCrossoverWrapper) for special cases
```

**Severity**: 🟡 **MEDIUM** — Guidance incomplete but not breaking

---

#### Issue #6: Unclear When Injection Mode Best Used

**Location**: real.py _injection class docstrings; README (missing entirely)

**Documentation in Code** (real.py UniformCrossover_injection):
```python
"""
Injection-mode Uniform Crossover.
Single key splits into (n_pairs * n_offspring) subkeys; jax.vmap(per_row) generates all masks
in parallel, returning (n_pairs * n_offspring, d) flattened array for base wrapper to unfold.
Trade-off: Full noise materialization enables reproducibility without key re-splitting.
"""
```

**What's Unclear**:
- When to use injection vs standard?
- Memory overhead of "full materialization"?
- Performance implications?
- Which mode should users prefer?

**Missing README Section** (should explain):
```markdown
## Tier 2 Variants: Standard vs Injection Mode

**Standard Mode** (BaseCrossover base class):
- Operator consumes pre-allocated keys from ResourceMapper
- Keys reshaped (n_pairs, n_offspring, num_keys_per_atomic_operation, 2)
- Tier 2 (_generate_noise) called once per pair-offspring combination inside vmap
- Lazy generation: noise generated on-demand per kernel call
- Memory: O(num_keys_per_atomic_operation * 2) per vmap call

**Injection Mode** (BaseCrossover_injection subclass):
- Operator receives single key from engine
- Internally splits into (n_pairs * n_offspring) subkeys upfront
- Tier 2 (_generate_noise) returns fully materialized noise array
- Memory: O(n_pairs * n_offspring * genome_shape) upfront
- Trade-off: More memory for exact reproducibility + explicit noise control

**When to Use Each**:
- Standard mode: Default; memory-efficient; recommended for most operators
- Injection mode: When full noise materialization needed; when reproducibility is critical
```

**Severity**: 🟡 **MEDIUM** — Choice unguided but not blocking

---

#### Issue #7: FB-1 Reference Needs Explanation

**Location**: README lines 89, 95 (references FB-1 design note)

**README Says**:
```
# eliminates a physical data copy that would break XLA fusion (see FB-1).

> **Design note (FB-1)**: Earlier versions used an offspring-major ordering via
> `jnp.transpose`. This forced XLA to materialize a physical data copy...
```

**Problem**: 
- What is "FB-1"? Design doc? Issue tracker? Internal reference?
- No context given
- Reader must guess what FB-1 means
- Not explained in README intro or references section

**Better Format**:
```
# eliminates a physical data copy that would have broken XLA fusion 
# (original design used transpose; see architecture notes below)

> **Design note**: Earlier versions used an offspring-major ordering via
> `jnp.transpose`...
```

**Or**:
```
> **Design Evolution**: In Phase 3 refactoring (see DESIGN.md §2.1), 
> we switched from offspring-major to pair-major ordering...
```

**Severity**: 🟡 **MEDIUM** — Obscure reference but not blocking understanding

---

## Summary Table

| # | Issue | Severity | Category | Location |
|---|-------|----------|----------|----------|
| 1 | Injection mode undocumented | 🔴 CRITICAL | Missing Feature | Entire module |
| 2 | SBX offspring semantics unclear | 🔴 CRITICAL | Semantics | Parameters table |
| 3 | Mask semantics inconsistency | 🟠 HIGH | Implementation | Mask semantics |
| 4 | Evosax wrapper pattern undocumented | 🟠 HIGH | Alternative Pattern | Missing section |
| 5 | Developer checklist incomplete | 🟡 MEDIUM | Guidance | Checklist |
| 6 | Injection mode usage unclear | 🟡 MEDIUM | Guidance | Missing section |
| 7 | FB-1 reference unexplained | 🟡 MEDIUM | Clarity | Design note |

---

## Recommended Action Priority

### Phase 1 (Critical - Fix Semantics)
1. Add explanation of injection mode and BaseCrossover_injection pattern
2. Clarify SBX offspring semantics (single per call, num_offspring configurable)
3. Fix/clarify mask semantics (per-element vs gate-based distinction)

### Phase 2 (High - Complete Implementation Docs)
4. Explain EvosaxUniformCrossoverWrapper pattern and use cases
5. Add typed_keys handling to checklist
6. Document offspring semantics in Tier 3 context

### Phase 3 (Medium - Improve Clarity)
7. Add guidance on when to use injection vs standard mode
8. Explain FB-1 reference or replace with clearer phrasing
9. Expand checklist with mask semantics, type handling, alternative patterns

---

## Code Quality Notes

✅ **Strengths**:
- All implementations correctly implement the 3-tier paradigm
- Docstrings in code are detailed and well-written
- Binary operators work correctly (UniformCrossover, SinglePointCrossover)
- Real-valued operators comprehensive (Blend, SBX, Binomial, EvosaxWrapper)
- Injection mode alternatives available and working

⚠️ **Issues**:
- Injection mode completely hidden from README
- SBX docstring misleading about inherent 2-offspring nature
- Mask semantics inconsistent across operators (not clearly documented)
- EvosaxUniformCrossoverWrapper pattern undocumented
- FB-1 reference unexplained
- Incomplete developer guidance (typed_keys, offspring semantics, mask distinctions)

---

## Files Modified/Exported

**Implementations** (all working correctly):
- `binary.py`: UniformCrossover, SinglePointCrossover
- `real.py`: UniformCrossover, Blend, SBX, Binomial, + _injection variants
- `evosax_crossover.py`: EvosaxUniformCrossoverWrapper
- `__init__.py`: Exports + registry includes all operators

**Missing from README**:
- All injection mode variants
- Evosax wrapper pattern
- Mask semantics distinctions
- When/why to use each mode
- Detailed typed_keys handling
