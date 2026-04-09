# Option C: Backward Compatibility Deep Dive

## Executive Summary

Implementing Option C (DataID Config) **can maintain 100% backward compatibility** if designed carefully. The key insight: **make all new features optional with intelligent fallback logic**.

---

## Current System Architecture

### 1. Call Hierarchy
```
Composer.quick_run(fitness="sphere:dim=10", ...)
    ↓
    Load TOML config (if provided)
    ↓
OperatorCatalog.get("sphere:dim=10")
    ↓
    Factory: _create_bbob_evaluator(dim=10, ...)
    ↓
    -> BBO BEvaluator instance
```

### 2. Current Signatures (Existing)

```python
# Composer.quick_run()
def quick_run(
    self,
    fitness: Optional[str] = None,  # "sphere:dim=10"
    **kwargs
) -> ExperimentResult

# OperatorCatalog.get()
def get(self, spec: str) -> Any:
    """Takes ONLY the spec string"""

# load_experiment_config()
def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Returns (experiment_meta, resolved_pipelines)"""
```

---

## Option C Proposal: What Would Change?

### 1. New TOML Structure (Optional)

```toml
# OLD (still works)
[experiment.shared]
fitness = "sphere:dim=10"

# NEW (optional enhancement)
[data.sphere_10]
source = "synthetic"
generator = "sphere"
dim = 10
seed = 42

[experiment.shared]
fitness = "sphere:data_id=sphere_10"  # Reference by ID
# OR (backward compat)
fitness = "sphere:dim=10"              # Direct params still work
```

### 2. New Optional Parameters

```python
# Composer.quick_run()
def quick_run(
    self,
    fitness: Optional[str] = None,
    data_config: Optional[Dict[str, Any]] = None,  # NEW (optional)
    **kwargs
) -> ExperimentResult

# OperatorCatalog.get()
def get(
    self,
    spec: str,
    data_registry: Optional[Dict[str, Any]] = None,  # NEW (optional kwarg)
) -> Any

# load_experiment_config()
def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> Tuple[
    Dict[str, Any],              # experiment_meta
    Dict[str, Dict[str, Any]],   # resolved_pipelines
    Dict[str, Any],              # NEW: data_registry (optional)
]
```

---

## Breaking Changes & Mitigation

### ❌ Breaking Change 1: OperatorCatalog.get() Signature

**Current:**
```python
factory, defaults = self._registry[operator_type]
merged = {**defaults, **user_params}
return factory(**merged)
```

**If we add data_registry as positional arg → BREAKS existing code**

**Solution: Make it a keyword-only argument with default None**
```python
def get(self, spec: str, data_registry: Optional[Dict[str, Any]] = None) -> Any:
    """data_registry is optional; if None, falls back to direct params"""
    operator_type, user_params = self.parse_spec(spec)
    
    if operator_type in self._evosax_strategies:
        return self._evosax_strategies[operator_type]
    
    factory, defaults = self._registry[operator_type]
    
    # NEW: Resolve data_id if present
    if data_registry and "data_id" in user_params:
        data_id = user_params.pop("data_id")
        if data_id not in data_registry:
            raise KeyError(f"Data ID '{data_id}' not found in registry")
        # Inject resolved data
        user_params["_resolved_data"] = data_registry[data_id]
    
    merged = {**defaults, **user_params}
    return factory(**merged)
```

**✅ Backward Compatible Because:**
- Existing calls `catalog.get("sphere:dim=10")` work unchanged
- New calls `catalog.get("sphere:data_id=...", data_registry={...})` work too
- Old factories ignore `data_registry` parameter

---

### ❌ Breaking Change 2: Fitness Factory Signatures

**Current:**
```python
def _create_bbob_evaluator(**kwargs):
    config = BBOBConfig(
        fn_name=kwargs.get("fn_name", "sphere"),
        num_dims=kwargs.get("dim", 10),
        maximize=kwargs.get("maximize", True),
        seed=kwargs.get("seed", 42),
    )
    return BBOBEvaluator.create(config)
```

**If we add `_resolved_data` kwarg → Current code breaks if it doesn't accept `**kwargs` properly**

**Solution: Ensure all factories use flexible `**kwargs` with `.pop()` for Option C params**

```python
def _create_bbob_evaluator(**kwargs):
    # Remove Option C params before passing to config
    _resolved_data = kwargs.pop("_resolved_data", None)
    
    config = BBOBConfig(
        fn_name=kwargs.get("fn_name", "sphere"),
        num_dims=kwargs.get("dim", 10),
        maximize=kwargs.get("maximize", True),
        seed=kwargs.get("seed", 42),
    )
    return BBOBEvaluator.create(config)
```

**✅ Backward Compatible Because:**
- `.pop()` safely removes `_resolved_data` if present
- Factories that don't use `.pop()` still work (kwargs just has extra key)
- All tests pass unchanged

---

### ❌ Breaking Change 3: load_experiment_config() Return Type

**Current (v1):**
```python
meta, pipelines = load_experiment_config("config.toml")
# Returns: Tuple[Dict, Dict]
```

**If we return 3-tuple → BREAKS unpacking**
```python
meta, pipelines, data = load_experiment_config("config.toml")
# Returns: Tuple[Dict, Dict, Dict]
```

**Solution: Make it backward compatible by returning extended tuple**

```python
@dataclass
class ExperimentLoadResult:
    """Backward compatible config load result."""
    meta: Dict[str, Any]
    pipelines: Dict[str, Dict[str, Any]]
    data_registry: Dict[str, Any] = field(default_factory=dict)
    
    # Allow unpacking like old tuple
    def __iter__(self):
        # Old code: meta, pipelines = result
        yield self.meta
        yield self.pipelines
        # NEW code can do:
        # result.data_registry (or unpack all 3)

def load_experiment_config(path: str, pipelines=None):
    # ... existing logic ...
    data_registry = _build_data_registry(cfg.get("data", {}))
    return ExperimentLoadResult(meta, pipelines_dict, data_registry)
```

**✅ Backward Compatible Because:**
```python
# OLD code still works (implicit unpacking first 2)
meta, pipelines = load_experiment_config("config.toml")

# NEW code can access all
result = load_experiment_config("config.toml")
data_reg = result.data_registry
meta, pipelines = result  # unpacking shorthand
```

---

### ❌ Breaking Change 4: Composer.quick_run() Signature

**Current:**
```python
result = composer.quick_run(
    fitness="sphere:dim=10",
    seeds=(1,2,3),
    ...
)
```

**Adding optional kwarg → NOT breaking**
```python
def quick_run(
    self,
    fitness: Optional[str] = None,
    data_config: Optional[Dict[str, Any]] = None,  # NEW (optional)
    **kwargs
):
    # If data_config provided, build registry
    data_registry = {}
    if data_config:
        data_registry = self._build_data_registry(data_config)
    
    # Existing logic...
    evaluator = self.catalog.get(fitness, data_registry=data_registry)
```

**✅ Backward Compatible Because:**
- `data_config` defaults to `None`
- All existing calls work unchanged
- New calls can pass `data_config={"sphere_10": ...}`

---

## Migration Path: 3 Stages

### Stage 1: Backward Compatible Enhancement (Additive)
**Time: 2-3 hours | Breaking Changes: 0**

1. Add optional `data_registry` kwarg to `OperatorCatalog.get()`
2. Add optional `_resolved_data` handling in all fitness factories
3. Create `_build_data_registry()` helper (no TOML parsing yet)
4. Add optional `data_config` kwarg to `Composer.quick_run()`

**Result:** Code works unchanged, but infrastructure ready for data configs

---

### Stage 2: TOML Data Parsing (Gradual)
**Time: 1-2 hours | Breaking Changes: 0 (with compatibility class)**

1. Enhance `load_experiment_config()` to parse `[data.*]` sections
2. Use `ExperimentLoadResult` dataclass for backward compat unpacking
3. Update examples to show both old and new patterns
4. Add deprecation warning for direct params (optional)

**Result:** Old TOML files work. New files can use `[data.*]` sections.

---

### Stage 3: Best Practices & Documentation (Non-breaking)
**Time: 1 hour | Breaking Changes: 0**

1. Document dual patterns (old vs new)
2. Add examples showing both approaches
3. Migration guide for large projects
4. Linting rules: `prefer_data_id_over_inline_params` (optional, advisory)

**Result:** Clear migration path without forcing anyone

---

## Backward Compatibility Matrix

| Scenario | Old Code | Will It Work? | Notes |
|----------|----------|---|---|
| `quick_run(fitness="sphere:dim=10")` | ✅ Yes | ✅ Unchanged | Direct params still work |
| `catalog.get("sphere:dim=10")` | ✅ Yes | ✅ Unchanged | No data_registry needed |
| `meta, pipelines = load_config()` | ✅ Yes | ✅ Works | `ExperimentLoadResult` unpacks |
| TOML without `[data.*]` | ✅ Yes | ✅ Works | Loads as before |
| Python factories (`_create_*`) | ✅ Yes | ✅ Works | `.pop("_resolved_data")` is safe |
| Pickled evaluators (v2.0 format) | ✅ Yes | ✅ Works | No serialization changes |

---

## Implementation Details: How to Keep It Safe

### 1. Add Type Hints for Clarity
```python
from typing import Optional, Dict, Any

def get(
    self,
    spec: str,
    data_registry: Optional[Dict[str, Any]] = None,  # Type clear
) -> Any:
    """Catalog lookup with optional data resolution."""
    ...
```

### 2. Defensive Programming in Factories
```python
def _create_tsp_evaluator(**kwargs):
    # Remove Option C params FIRST
    _resolved_data = kwargs.pop("_resolved_data", None)
    
    if _resolved_data is not None:
        # NEW: Use resolved data
        distance_matrix = _resolved_data["distance_matrix"]
        return TSPEvaluator.create_from_matrix(distance_matrix)
    
    # OLD: Use direct params (fallback)
    num_cities = kwargs.get("num_cities", 50)
    random_seed = kwargs.get("random_seed", 42)
    return TSPEvaluator.create_synthetic(num_cities, random_seed)
```

### 3. Preserve TOML Structure
```toml
# OLD TOML (still works)
[experiment.shared]
fitness = "sphere:dim=10"

# NEW TOML (opt-in)
[data.sphere_10]
source = "synthetic"
dim = 10

[experiment.shared]
fitness = "sphere:data_id=sphere_10"
```

### 4. Intelligent Fallback Logic
```python
def quick_run(
    self,
    fitness: Optional[str] = None,
    data_config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    # If no data_config provided, works exactly as before
    data_registry = {}
    if data_config:
        data_registry = self._build_data_registry(data_config)
    
    # Pass to catalog
    evaluator = self.catalog.get(fitness, data_registry=data_registry)
    # ... rest unchanged ...
```

---

## Risk Assessment

### Low Risk (✅ Can be implemented safely)
- ✅ Adding optional kwargs (always safe if default to None)
- ✅ Using `.pop()` for optional dict keys
- ✅ `ExperimentLoadResult` unpacking trick
- ✅ Separating data parsing from evaluator logic

### Medium Risk (⚠️ Need careful testing)
- ⚠️ Modifying `_registry` structure (but we only read it, don't change format)
- ⚠️ TOML parsing edge cases (but isolated in `load_experiment_config()`)
- ⚠️ Data format validation (add comprehensive error messages)

### High Risk (❌ Avoid)
- ❌ Changing existing positional argument order
- ❌ Removing old factory functions
- ❌ Breaking TOML structure for `[pipelines.*]`

---

## Testing Strategy

### Unit Tests (Ensure backward compat)
```python
def test_catalog_get_backward_compat():
    """Old way still works"""
    catalog = OperatorCatalog()
    evaluator = catalog.get("sphere:dim=10")
    assert isinstance(evaluator, BBOBEvaluator)

def test_catalog_get_with_data_registry():
    """New way works too"""
    catalog = OperatorCatalog()
    data_reg = {"sphere_10": {...}}
    evaluator = catalog.get("sphere:data_id=sphere_10", data_registry=data_reg)
    assert isinstance(evaluator, BBOBEvaluator)

def test_load_config_unpacking_backward_compat():
    """Old unpacking still works"""
    meta, pipelines = load_experiment_config("examples/mock_binary_experiment.toml")
    assert isinstance(meta, dict)
    assert isinstance(pipelines, dict)

def test_quick_run_without_data_config():
    """Old API unchanged"""
    result = composer.quick_run(
        fitness="sphere:dim=10",
        seeds=(42,)
    )
    assert result is not None
```

### Integration Tests
```python
def test_full_stack_old_way():
    """End-to-end: old TOML file, old API"""
    result = composer.quick_run(
        fitness="sphere:dim=10",
        seeds=(1, 2)
    )
    assert len(result.runs) == 2

def test_full_stack_new_way():
    """End-to-end: new TOML with [data], new API"""
    result = composer.quick_run(
        fitness="sphere:data_id=sphere_10",
        data_config={...},
        seeds=(1, 2)
    )
    assert len(result.runs) == 2
```

---

## Migration Guide for Users

### For Existing Projects
```python
# This still works exactly as before
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament",
    crossover="blend",
    mutation="gaussian",
    seeds=(1, 2, 3)
)
```

### For New Projects (Opting In)
```python
# Define data once
data_config = {
    "sphere_10": {
        "source": "synthetic",
        "generator": "sphere",
        "dim": 10,
        "seed": 42
    }
}

# Reuse across multiple runs
for algo in ["ga", "pso", "de"]:
    result = composer.quick_run(
        fitness="sphere:data_id=sphere_10",  # Reference by ID
        data_config=data_config,
        algorithm=algo,
        seeds=(1, 2, 3)
    )
    print(f"Algorithm {algo}: {result.aggregated_summary()}")
```

### For TOML Files
```toml
# Old way (still works)
[experiment.shared]
fitness = "sphere:dim=10"

# New way (opt-in)
[data.sphere_10]
source = "synthetic"
dim = 10
seed = 42

[experiment.shared]
fitness = "sphere:data_id=sphere_10"
```

---

## Deprecation Strategy (Optional, For Future)

If we want to eventually prefer data IDs, we can add soft deprecation:

```python
def _validate_and_warn_on_inline_params(operator_type, user_params):
    """Warn users about inline params when data_id is available."""
    if (operator_type in DATA_DRIVEN_EVALUATORS and 
        "data_id" not in user_params and
        "file_path" not in user_params):
        warnings.warn(
            f"{operator_type} supports data_id parameter; "
            "consider using data configurations for reproducibility",
            FutureWarning
        )
```

**But this is optional** — we never force users to migrate.

---

## Conclusion

### ✅ Option C CAN be implemented with zero breaking changes if:

1. **All new parameters are optional** with sensible defaults
2. **Old code paths always work** (fallback logic)
3. **Factory functions are flexible** (use `.pop()` for optional keys)
4. **TOML structure is extended, not changed** (backward compat sections)
5. **Return types use clever unpacking tricks** (ExperimentLoadResult)

### 🎯 Recommendation: Go with Option C

- **Backward compatibility: 100% achievable**
- **Implementation effort: 3-4 hours for core + tests**
- **Real-world benefits: Enterprise-grade reproducibility**
- **Risk level: Low with careful testing**
- **Migration cost for users: Zero (no forced changes)**

**The beauty:** Old code and new code coexist peacefully in the same codebase for as long as needed.

