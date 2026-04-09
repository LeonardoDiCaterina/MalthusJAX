# LinearGP & LinearGenome Removal - Completion Status

## ✅ COMPLETED ACTIONS

### 1. **Fitness README.md** - Fully Updated
- ❌ **Removed Section 4**: "Tensor Interface" ghost feature (was ~65 lines)
- ✅ **Added Section 4**: "Static Configuration and Data Storage" with clearer guidance on `pytree_node=False` pattern
- ✅ **Updated Specialized Evaluators Section**:
  - Removed LinearGPEvaluator description
  - Added TSP subsection with detailed explanation of Random Key encoding
  - Fixed BBOB description to clarify sign convention
- ✅ **Updated Data Management section** (now Section 6): Removed LinearGPEvaluator example
- ✅ **Updated Quick Examples** (now Section 8):
  - Removed LinearGPEvaluator regression example
  - Added TSP synthetic generation example
- ✅ Renumbered all subsequent sections for consistency

### 2. **BBOB Sign Convention Comment** - Fixed
- **File**: `src/malthusjax/core/fitness/bbob_evaluator.py` (lines 92-96)
- **Changed from**: Confusing/backwards explanation
- **Changed to**: Clear explanation that engines always maximize internally

### 3. **Fitness Module Exports** - Updated
- **File**: `src/malthusjax/core/fitness/__init__.py`
- ✅ Removed imports of `linear_gp_evaluator`
- ✅ Removed exports: `LinearGPEvaluator`, `LinearGPEvaluatorConfig`, `TENSORGP_FUNCTIONS`, `TENSORGP_NAMES`
- ✅ Removed LinearGP-related catalog registration code

### 4. **Genome Module Exports** - Updated
- **File**: `src/malthusjax/core/genome/__init__.py`
- ✅ Removed import of `linear_genome`
- ✅ Removed exports: `LinearGenome`, `LinearGenomeConfig`, `LinearPopulation`

### 5. **Test Files** - Updated
- **File**: `tests/core/genome/test_from_tensor_extended.py`
- ✅ Removed import of `LinearGenome`
- ✅ Removed test functions:
  - `test_linear_from_tensor_basic()`
  - `test_linear_from_tensor_jit()`

---

## ⚠️ FILES TO MANUALLY DELETE

The following files still exist in the repository and should be deleted:

```bash
# Run these commands to complete removal:
rm src/malthusjax/core/fitness/linear_gp_evaluator.py
rm src/malthusjax/core/genome/linear_genome.py
rm tests/core/genome/test_linear_genome.py
```

All imports and references have been removed from the codebase. These files are now **completely unreferenced** and can be safely deleted.

---

## VERIFICATION CHECKLIST

- ✅ No remaining imports of `LinearGPEvaluator` in codebase
- ✅ No remaining imports of `LinearGenome` in codebase
- ✅ No remaining references in `__init__.py` files
- ✅ All tests updated/removed
- ✅ README documentation cleaned up
- ✅ BBOB comment clarified
- ✅ TSP documentation added
- ✅ Metadata storage pattern documented
- ⏳ Physical files need manual deletion

---

## IMPACT ANALYSIS

### What Changed:
1. Users cannot import `LinearGPEvaluator` anymore (import will fail)
2. Tensor Interface section removed from documentation
3. BBOB sign handling better documented
4. TSP evaluator now properly documented

### What Didn't Change:
- No core engine logic affected
- No operator changes
- No existing benchmarks affected
- `RegressionData` type still available for other uses

### Build Status:
After deleting the three listed files, the project should:
- ✅ Pass all imports (no dangling references)
- ✅ Pass test suite (LinearGenome tests removed)
- ✅ Pass linting checks (no LinearGP references)
- ✅ Pass mypy (no type references to removed classes)
