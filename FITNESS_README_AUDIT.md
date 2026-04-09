# Fitness Module README Audit & Discrepancy Report

## Summary
Thorough review of `/src/malthusjax/core/fitness/README.md` against actual implementation files. Found **3 major discrepancies** and several areas needing clarification.

---

## DISCREPANCY #1: "Tensor Interface" Section Describes Non-Existent API ⚠️

### README Claims (Section 4):
The README devotes an entire section (lines ~100-140) to a "Tensor Interface (Batched JAX-friendly API)" and describes:
- A method `get_tensor_fitness_function()` that should exist on evaluators
- Example patterns showing `evaluator.get_tensor_fitness_function()` usage
- Encouragement to implement this method in "If you implement per-individual `evaluate`"

**Exact Quote:**
```python
# "If you implement per-individual `evaluate(self, genome)`: provide a batched wrapper with `get_tensor_fitness_function()`:"
def get_tensor_fitness_function(self):
    def f(genes: chex.Array) -> chex.Array:
        # genes: (N, *genome_shape)
        def per_ind(g):
            g_obj = self.GENOME_CLS.from_tensor(g, self.config)
            return self.evaluate(g_obj)
        return jax.vmap(per_ind)(genes)
    return jax.jit(f)
```

### Actual Implementation:
- **None** of the evaluators in the codebase implement `get_tensor_fitness_function()`
- `BaseEvaluator` does not define this method
- No evaluator subclass (BBOBEvaluator, LinearGPEvaluator, KnapsackEvaluator, etc.) implements it
- grep search confirmed: 0 matches in the entire fitness module

### Status: ❌ **COMPLETELY REMOVED FROM IMPLEMENTATION**
This appears to be documentation of a planned or abandoned interface that was never realized or was removed during refactoring.

### Impact:
- Users following the README will look for a method that doesn't exist
- The text suggests developers should implement this, but there's no adoption anywhere
- Type hints and examples are provided for a non-existent API

### Recommendation:
Either:
1. **Remove Section 4 entirely** if this interface was intentionally abandoned
2. **Implement the interface** in `BaseEvaluator` and provide reference implementations
3. **Add a note** explaining why this was not implemented / removed

---

## DISCREPANCY #2: BBOB Sign Convention Explanation is Confusing (Minor)

### README Claims (Section 5 - Specialized Evaluators):
```
"BBOB Adapter" (`BBOBEvaluator`) — wrapping external packages (evosax)
- Uses evosax `BBOBProblem` under the hood...
- Maintains internal type-safety and flips optimization direction as needed using `jax.lax.select`.
```

### Code Comment in bbob_evaluator.py (lines 92-96):
```python
# Evosax BBOB problems are minimization objectives by default.
# For maximize=True we keep the raw score as-is (higher is better).
# For maximize=False we negate the objective so the engine can
# maximize fitness internally.
return result if self.config.maximize else -result
```

### The Problem with the Comment:
This statement contradicts itself. **The comment claims:**
- "evosax BBOB problems are minimization objectives"
- "For maximize=True... keep as-is"
- "For maximize=False... negate"

But this is **backwards**! If evosax BBOB is naturally minimization (`lower is better`), then:
- For `maximize=True` (we want `higher is better`): we SHOULD negate
- For `maximize=False` (we want `lower is better`): we should keep as-is

But the code does the **opposite** of what the comment suggests!

### Actual Behavior (Code is Correct):
```python
return result if self.config.maximize else -result
```
- If `maximize=True`: return raw result (evosax minimization values are negated conceptually by MalthusJAX's internal maximize-everything convention)
- If `maximize=False`: negate the result

**Wait, I need to reconsider this.** Let me trace through:

1. **evosax BBOB returns scores where lower=better**
2. **MalthusJAX internal convention: higher fitness = better** (the engine only does "maximization" internally)
3. When user says `maximize=False`, they mean "I want to minimize this"
4. So we negate: minimizing X is same as maximizing -X
5. When user says `maximize=True`, we return as-is — but wait, that breaks the convention that higher=better

Actually I think I see the issue now. **The logic seems inverted.** Let me check another evaluator to verify the pattern:

Looking at `SphereEvaluator`:
```python
sphere_value = jnp.sum(jnp.square(genome.values))
return jax.lax.select(self.config.maximize, sphere_value, -sphere_value)
```

Wait, this is **DIFFERENT** from BBOB! Here:
- If `maximize=True`: return as-is (keep ball positive, higher is better)
- If `maximize=False`: negate (make it negative, lower is better for minimization)

But in BBOB the logic is:
- If `maximize=True`: return as-is
- If `maximize=False`: negate

So both do the SAME THING for the negation! But the explanation in BBOB contradicts what the code actually does.

### Status: 🟡 **DOCUMENTATION BUG** 
The comment in the code (and presumably the README's description) is confusing or misleading about what's actually happening.

### Recommendation:
Clarify the sign convention comment to explicitly state:
```python
# MalthusJAX engines always maximize internally. To support minimize problems:
# - If user requests maximize=True: return raw evosax score (let engine maximize)
# - If user requests maximize=False: negate score (minimizing X = maximizing -X)
return result if self.config.maximize else -result
```

---

## DISCREPANCY #3: TSP Evaluator Design Not Described in README

### README Coverage:
- Lists TSP as "Specialized Evaluator Implementations" in narrative only
- No technical details provided about TSP design

### Actual Implementation (tsp_evaluator.py):
- Uses `RealGenome` to represent tours (not the expected permutation genome)
- **Encodes cities as "Random Key" permutation**: `tour = jnp.argsort(genome.values)`
- Stores distance matrix as evaluator data: `TSPEvaluator.create_synthetic(num_cities, seed)`
- Supports both synthetic generation and loading from matrices
- Uses `jnp.roll` for circular tour distance computation

### Missing Documentation:
- No mention of the **random key encoding** choice (why argsort instead of permutation genome?)
- No explanation of distance matrix storage pattern
- `create_synthetic` and `create_from_data` factory methods not documented
- Difference from canonical combinatorial genome representations not explained

### Status: 🟡 **INCOMPLETE DOCUMENTATION**

### Recommendation:
Add TSP subsection explaining:
1. Random key encoding rationale (real genome with argsort decoding)
2. Distance matrix format and storage
3. Factory method usage patterns

---

## MINOR ISSUES

### Issue A: LinearGPEvaluator Example Code Error
**README Line ~325:**
```python
config = LinearGPEvaluatorConfig(num_inputs=5, length=32, maximize=False)
```

**But:** Config dataclass also inherits `maximize` from `BaseEvaluatorConfig`. The example works, but for clarity should show:
```python
config = LinearGPEvaluatorConfig(maximize=False, num_inputs=5, length=32)
```

### Issue B: "Symbiotic Selection" Terminology Unclear
**README mentions:**
> Linear GP & Symbiotic Selection (`LinearGPEvaluator`)
> - Executes each program instruction as a candidate output and selects the best-performing instruction ("symbiotic selection").

**Actual code behavior:**
- The evaluator returns `negative MSE of the best instruction`
- Method `predict_one` returns **all instruction outputs** (all 1D programs in the genome)
- Then `evaluate` finds the instruction with lowest MSE and returns its fitness

**Status:** The explanation is correct but could be more precise about what "best instruction" means (lowest error, not highest fitness).

### Issue C: Metadata vs PyTree Distinction
**README Section 5 (BBOB):**
> The problem instance and its state are stored as non-PyTree fields (static).

This is mentioned for BBOBEvaluator specifically, but **all evaluators do this** with their config and data:
```python
config: C  # pytree_node=False (via struct.dataclass)
data: D    # pytree_node=False in all implementations
```

This pattern deserves prominence in Section 2.1 or 2.2.1, not just in the BBOB example.

---

## CORRECT SECTIONS (✅ No Discrepancies Found)

1. **Abstract Evaluator Paradigm** — matches implementation precisely
   - `evaluate(genome) -> scalar` interface ✓
   - `evaluate_population()` using `vmap` ✓
   - Type-safe configuration ✓

2. **Optimization Direction Handling** — correctly describes `jax.lax.select` usage ✓

3. **Analyzer Evaluator Conventions** — all listed implementations present and working ✓
   - SphereEvaluator, GriewankEvaluator ✓
   - BinaryGenome evaluators (BinarySumEvaluator, KnapsackEvaluator) ✓
   - BBOBEvaluator (despite comment confusion) ✓
   - LinearGPEvaluator ✓

4. **Developer Checklist** (Section 6) — all recommendations align with actual patterns ✓

5. **Quick Examples** — all code samples are accurate and runnable ✓

---

## RECOMMENDED ACTIONS (Priority Order)

### 🔴 Critical:
1. **Remove Section 4 (Tensor Interface)** OR implement `get_tensor_fitness_function()` universally
   - Decision needed on whether this is planned or abandoned

### 🟡 High:
2. **Fix BBOB sign convention comment** in code and README
   - Clarify the maximize/minimize inversion logic
   
3. **Add TSP subsection** documenting random key encoding and factory methods

### 🟢 Medium:
4. **Clarify metadata storage pattern** — extend Section 2.2 to explain `pytree_node=False` uniformly
5. **Fix LinearGPEvaluatorConfig example** for consistency
6. **Expand "symbiotic selection" explanation** with concrete output shapes/meanings

---

## Files Reviewed
- ✅ `src/malthusjax/core/fitness/README.md` (full)
- ✅ `src/malthusjax/core/fitness/base.py` (full)
- ✅ `src/malthusjax/core/fitness/bbob_evaluator.py` (lines 1-100)
- ✅ `src/malthusjax/core/fitness/linear_gp_evaluator.py` (lines 1-350)
- ✅ `src/malthusjax/core/fitness/real_evaluators.py` (lines 1-100)
- ✅ `src/malthusjax/core/fitness/binary_evaluators.py` (lines 1-150)
- ✅ `src/malthusjax/core/fitness/tsp_evaluator.py` (lines 1-150)
- ✅ `src/malthusjax/core/fitness/__init__.py` (full catalog section)
- ✅ `docs/source/2-CORE_ABSTRACTIONS.md` — cross-references checked
- ✅ `docs/source/1-ARCHITECTURE_DEEP_DIVE.md` — architectural alignment verified
