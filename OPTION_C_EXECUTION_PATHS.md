# Option C: Dual-Code-Path Architecture

Visual guide to how Option C maintains backward compatibility through dual execution paths.

---

## Execution Flow Diagram

### Current (v2.0): Single Path - No Data Configs

```
Composer.quick_run(fitness="sphere:dim=10")
    │
    ├─ No data_config provided
    │
    ├─ catalog.get("sphere:dim=10")
    │    │
    │    ├─ parse_spec() → ("sphere", {"dim": 10})
    │    │
    │    ├─ factory = _create_bbob_evaluator
    │    │
    │    └─ factory(dim=10, ...)
    │        │
    │        └─ BBOBConfig(fn_name="sphere", num_dims=10)
    │            │
    │            └─ BBOBEvaluator.create(config)
    │
    └─ Run engine with evaluator
       └─ Result
```

---

### Proposed (v3.0): Dual Paths - Backward Compat + Data Configs

```
Composer.quick_run(
    fitness="sphere:dim=10",
    data_config=None
)
    │
    ├─ PATH A (Old): data_config=None (backward compat)
    │   │
    │   ├─ catalog.get("sphere:dim=10", data_registry={})
    │   │    │
    │   │    ├─ parse_spec() → ("sphere", {"dim": 10})
    │   │    │
    │   │    ├─ No "data_id" in params → use PATH A
    │   │    │
    │   │    ├─ factory = _create_bbob_evaluator
    │   │    │
    │   │    └─ factory(dim=10)
    │   │        │
    │   │        └─ BBOBConfig(fn_name="sphere", num_dims=10)
    │   │
    │   └─ Run engine
    │      └─ Result (same as v2.0)
    │
    └─ OR
    
    Composer.quick_run(
        fitness="sphere:data_id=sphere_10",
        data_config={"sphere_10": {...}}
    )
        │
        ├─ PATH B (New): data_config provided
        │   │
        │   ├─ Build data_registry from data_config
        │   │    └─ {"sphere_10": <resolved data>}
        │   │
        │   ├─ catalog.get("sphere:data_id=sphere_10", data_registry={...})
        │   │    │
        │   │    ├─ parse_spec() → ("sphere", {"data_id": "sphere_10"})
        │   │    │
        │   │    ├─ "data_id" in params → use PATH B
        │   │    │
        │   │    ├─ Lookup in data_registry → found!
        │   │    │
        │   │    ├─ user_params["_resolved_data"] = <resolved>
        │   │    │
        │   │    ├─ factory = _create_bbob_evaluator
        │   │    │
        │   │    └─ factory(data_id="sphere_10", _resolved_data={...})
        │   │        │
        │   │        └─ _resolved_data.pop() extracts it
        │   │            │
        │   │            └─ BBOBEvaluator.create_from_data(_resolved_data)
        │   │
        │   └─ Run engine
        │      └─ Result (same output, auditable inputs)
```

---

## Code Path Comparison

### Path A: Old Style (Unchanged)
```python
# Input
fitness = "sphere:dim=10"
data_config = None

# Catalog logic
params = {"dim": 10}
if "data_id" not in params:
    # Use direct parameters
    factory(dim=10)

# Factory
def _create_bbob_evaluator(**kwargs):
    config = BBOBConfig(fn_name="sphere", num_dims=kwargs["dim"])
    return BBOBEvaluator.create(config)

# Output: BBOBEvaluator instance
```

### Path B: New Style (Data Config)
```python
# Input
fitness = "sphere:data_id=sphere_10"
data_config = {
    "sphere_10": {
        "distance_matrix": [...],  # Precomputed
        "problem_metadata": {...}
    }
}

# Catalog logic
params = {"data_id": "sphere_10"}
if "data_id" in params:
    # Resolve from registry
    params["_resolved_data"] = data_registry["sphere_10"]
    factory(data_id="sphere_10", _resolved_data={...})

# Factory
def _create_bbob_evaluator(**kwargs):
    _resolved_data = kwargs.pop("_resolved_data", None)  # Extract, remove
    if _resolved_data is not None:
        # Use precomputed data
        return BBOBEvaluator.create_from_data(_resolved_data)
    else:
        # Fall back to direct parameters
        config = BBOBConfig(fn_name="sphere", num_dims=kwargs["dim"])
        return BBOBEvaluator.create(config)

# Output: BBOBEvaluator instance (same class, different init path)
```

---

## TOML File Structure: Backward Compat

### Old TOML (v2.0 - Still Works in v3.0)
```toml
[experiment]
name = "sphere_baseline"

[experiment.shared]
fitness = "sphere:dim=10"
selection = "tournament:tournament_size=3"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.ga]
crossover = "blend:alpha=0.5"

# When loaded:
# ✅ No [data.*] -> data_registry = {}
# ✅ fitness = "sphere:dim=10" -> parsed as direct params
# ✅ Evaluator created via Path A (unchanged)
```

### New TOML (v3.0 - Enhanced)
```toml
[experiment]
name = "sphere_with_data_id"

# NEW: Define data once (optional)
[data.sphere_10]
source = "synthetic"
generator = "sphere"
dim = 10
seed = 42
problem_name = "Sphere-10D"

# Or reference external file
[data.sphere_real]
source = "file"
file_path = "data/sphere_benchmark.npz"
format = "npz"

[experiment.shared]
# OLD style (still works)
fitness = "sphere:dim=10"

# NEW style (uses data registry)
fitness = "sphere:data_id=sphere_10"

[pipelines.ga]
crossover = "blend:alpha=0.5"

# When loaded:
# ✅ [data.*] sections parsed into data_registry
# ✅ fitness = "sphere:data_id=sphere_10" -> path B
# ✅ Path B resolves data_id -> _resolved_data
# ✅ Evaluator created with precomputed data
```

---

## Factory Function Evolution

### v2.0 Factory (Existing)
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

**This factory:**
- ✅ Works with Path A (old style)
- ✅ Works with Path B (new style) because of `.get()`
- ✅ Argument `_resolved_data` is simply ignored (harmless)

### v3.0 Factory (Enhancement, Still Backward Compat)
```python
def _create_bbob_evaluator(**kwargs):
    # Extract Option C parameter safely
    _resolved_data = kwargs.pop("_resolved_data", None)
    
    if _resolved_data is not None:
        # NEW: Path B - use precomputed data
        config = BBOBConfig(
            fn_name=_resolved_data.get("fn_name", "sphere"),
            num_dims=_resolved_data.get("num_dims", 10),
            maximize=_resolved_data.get("maximize", True),
        )
        return BBOBEvaluator.create(config)
    else:
        # OLD: Path A - use direct parameters (unchanged behavior)
        config = BBOBConfig(
            fn_name=kwargs.get("fn_name", "sphere"),
            num_dims=kwargs.get("dim", 10),
            maximize=kwargs.get("maximize", True),
            seed=kwargs.get("seed", 42),
        )
        return BBOBEvaluator.create(config)
```

**This factory:**
- ✅ Works with Path A (old style) - `.pop()` removes nothing, behaves as before
- ✅ Works with Path B (new style) - `.pop()` extracts precomputed data
- ✅ Zero breaking changes for existing code

---

## Signature Evolution

### OperatorCatalog.get()

**v2.0:**
```python
def get(self, spec: str) -> Any:
    operator_type, user_params = self.parse_spec(spec)
    factory, defaults = self._registry[operator_type]
    merged = {**defaults, **user_params}
    return factory(**merged)
```

**v3.0 (Backward Compat):**
```python
def get(self, spec: str, data_registry: Optional[Dict[str, Any]] = None) -> Any:
    """
    data_registry is optional keyword-only parameter.
    If None, behaves exactly like v2.0.
    """
    operator_type, user_params = self.parse_spec(spec)
    
    # NEW: Resolve data_id if present and registry provided
    if data_registry and "data_id" in user_params:
        data_id = user_params.pop("data_id")
        if data_id not in data_registry:
            raise KeyError(f"Data ID '{data_id}' not found")
        user_params["_resolved_data"] = data_registry[data_id]
    
    factory, defaults = self._registry[operator_type]
    merged = {**defaults, **user_params}
    return factory(**merged)
```

**All existing calls still work:**
```python
# v2.0 code (no data_registry)
evaluator = catalog.get("sphere:dim=10")  # ✅ Still works

# v3.0 code (with data_registry)
evaluator = catalog.get("sphere:data_id=sphere_10", data_registry={...})  # ✅ New feature
```

---

## Unpacking Trick: ExperimentLoadResult

### Problem
Old code unpacks the config result:
```python
meta, pipelines = load_experiment_config("config.toml")
```

If we return 3-tuple, it breaks:
```python
meta, pipelines, data = load_experiment_config("config.toml")  # Runtime error in old code
```

### Solution: Custom Dataclass with `__iter__`

```python
@dataclass
class ExperimentLoadResult:
    """Backward compatible tuple-like result."""
    meta: Dict[str, Any]
    pipelines: Dict[str, Dict[str, Any]]
    data_registry: Dict[str, Any] = field(default_factory=dict)
    
    def __iter__(self):
        """Allow unpacking like old tuple: meta, pipelines = result"""
        yield self.meta
        yield self.pipelines
        # Note: __iter__ only yields 2 items, so old code works
        # New code accesses .data_registry as attribute
```

**Result:**
```python
# Old code (unpacking)
meta, pipelines = load_experiment_config("config.toml")  # ✅ Works
# result.__iter__() yields: self.meta, self.pipelines (2 items)

# New code (attribute access)
result = load_experiment_config("config.toml")
data_reg = result.data_registry  # ✅ Works
meta, pipelines = result  # ✅ Also works (same as old code)
```

---

## Summary: Zero Breaking Changes

| Point | v2.0 | v3.0 | Compat |
|-------|------|------|--------|
| `catalog.get(spec)` | ✅ Works | ✅ Works | ✅ Adding optional kwarg |
| `load_experiment_config()` unpacking | ✅ Works | ✅ Works | ✅ ExperimentLoadResult magic |
| TOML without `[data.*]` | ✅ Works | ✅ Works | ✅ Data parsing is optional |
| Factory functions | ✅ Works | ✅ Works | ✅ `.pop()` is safe |
| Evaluator instantiation | ✅ Works | ✅ Works | ✅ Fallback logic in factories |
| Serialization (pickle) | ✅ Works | ✅ Works | ✅ No format change |
| Tests | ✅ Pass | ✅ Pass | ✅ Add new tests, keep old |

---

## Testing Strategy

### Regression Tests (Verify v2.0 behavior preserved)
```python
def test_path_a_direct_params():
    """Ensure direct parameter passing still works"""
    evaluator = catalog.get("sphere:dim=10")
    assert isinstance(evaluator, BBOBEvaluator)
    # Verify it evaluates correctly
    ...

def test_load_config_old_unpacking():
    """Ensure old unpacking pattern works"""
    meta, pipelines = load_experiment_config("examples/experiment.toml")
    assert isinstance(meta, dict)
    assert isinstance(pipelines, dict)

def test_quick_run_backward_compat():
    """Full stack: old code works end-to-end"""
    result = composer.quick_run(fitness="sphere:dim=10", seeds=(42,))
    assert len(result.runs) == 1
```

### New Feature Tests (Verify v3.0 enhancements)
```python
def test_path_b_data_id():
    """Ensure data_id resolution works"""
    data_reg = {"sphere_10": {...}}
    evaluator = catalog.get("sphere:data_id=sphere_10", data_registry=data_reg)
    assert isinstance(evaluator, BBOBEvaluator)

def test_load_config_with_data_registry():
    """Ensure [data.*] sections parse correctly"""
    result = load_experiment_config("examples/with_data_config.toml")
    assert result.data_registry is not None
    assert "sphere_10" in result.data_registry
```

### Compatibility Tests (Both paths same output)
```python
def test_both_paths_produce_same_result():
    """Path A and Path B should give same fitness values"""
    # Path A: direct
    evaluator_a = catalog.get("sphere:dim=10")
    fitness_a = evaluator_a.evaluate(genome)
    
    # Path B: via data_id
    data_reg = {"sphere_10": <same problem setup>}
    evaluator_b = catalog.get("sphere:data_id=sphere_10", data_registry=data_reg)
    fitness_b = evaluator_b.evaluate(genome)
    
    assert np.allclose(fitness_a, fitness_b)
```

