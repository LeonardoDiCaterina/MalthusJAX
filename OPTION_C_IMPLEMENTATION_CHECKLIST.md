# Option C: Implementation Roadmap & Checklist

Complete breakdown of what needs to change (and what doesn't) to implement Option C while maintaining 100% backward compatibility.

---

## Files to Modify (17 files)

### Core Changes (5 files)

#### 1. `src/malthusjax/composer/catalog.py`
**Impact: Non-breaking enhancement**

```python
# Change 1: Add data_registry parameter
def get(self, spec: str, data_registry: Optional[Dict[str, Any]] = None) -> Any:
    # Existing code unchanged, just add:
    if data_registry and "data_id" in user_params:
        data_id = user_params.pop("data_id")
        if data_id not in data_registry:
            raise KeyError(f"Data ID '{data_id}' not in registry")
        user_params["_resolved_data"] = data_registry[data_id]

# Status: ✅ BACKWARD COMPATIBLE
# - Optional kwarg with default None
# - Old calls work unchanged
# - New calls enable data resolution
```

**Checklist:**
- [ ] Add type hint: `data_registry: Optional[Dict[str, Any]] = None`
- [ ] Add data_id resolution logic
- [ ] Add error handling for missing data_id
- [ ] Update docstring
- [ ] Add test: `test_catalog_get_with_data_id()`
- [ ] Add test: `test_catalog_get_backward_compat()`

**Lines: ~20 additions, 0 removals**

---

#### 2. `src/malthusjax/composer/config.py`
**Impact: Non-breaking enhancement**

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

# NEW: Backward-compatible result class
@dataclass
class ExperimentLoadResult:
    """Backward compatible config load result."""
    meta: Dict[str, Any]
    pipelines: Dict[str, Dict[str, Any]]
    data_registry: Dict[str, Any] = field(default_factory=dict)
    
    def __iter__(self):
        """Allow unpacking: meta, pipelines = result"""
        yield self.meta
        yield self.pipelines

# Change 1: Parse [data.*] sections
def _parse_data_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract [data.*] sections from TOML."""
    data_registry = {}
    data_section = cfg.get("data", {})
    for data_id, data_config in data_section.items():
        # Validate and store (placeholder for format-specific logic)
        data_registry[data_id] = data_config
    return data_registry

# Change 2: Return ExperimentLoadResult instead of tuple
def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> ExperimentLoadResult:
    """... existing docstring ..."""
    # ... existing code ...
    data_registry = _parse_data_section(cfg)
    return ExperimentLoadResult(experiment_meta, resolved, data_registry)

# Status: ✅ BACKWARD COMPATIBLE
# - Old unpacking: meta, pipelines = load_experiment_config() still works
# - New access: result.data_registry works
# - ExperimentLoadResult.__iter__() yields only 2 items
```

**Checklist:**
- [ ] Add `ExperimentLoadResult` dataclass
- [ ] Implement `__iter__` method for tuple unpacking
- [ ] Add `_parse_data_section()` helper
- [ ] Update `load_experiment_config()` return type
- [ ] Update docstring with backwards compat note
- [ ] Add tests: `test_load_config_unpacking_old_style()`
- [ ] Add tests: `test_load_config_data_registry_access()`

**Lines: ~40 additions, 0 removals**

---

#### 3. `src/malthusjax/composer/composer.py`
**Impact: Non-breaking enhancement**

```python
def quick_run(
    self,
    # ... existing parameters ...
    fitness: Optional[str] = None,
    # ... more existing ...
    data_config: Optional[Dict[str, Any]] = None,  # NEW
    **kwargs: Any,
) -> ExperimentResult:
    """... existing docstring (add data_config)\n"""
    
    # NEW: Build data registry if provided
    data_registry: Dict[str, Any] = {}
    if data_config:
        data_registry = self._build_data_registry(data_config)
    
    # ... existing logic ...
    resolved_crossover = catalog.get(
        crossover or "blend:alpha=0.5",
        data_registry=data_registry,  # NEW kwarg
    )
    
    # Status: ✅ BACKWARD COMPATIBLE
    # - data_config defaults to None
    # - Existing calls work unchanged
    # - New calls can pass data_config
```

**Checklist:**
- [ ] Add `data_config` parameter to signature
- [ ] Add `_build_data_registry()` method
- [ ] Pass `data_registry` to all `catalog.get()` calls for data-driven evaluators
- [ ] Update docstring
- [ ] Add tests

**Lines: ~15 additions, 0 removals**

---

#### 4. `src/malthusjax/core/fitness/__init__.py` (Fitness Factories)
**Impact: Non-breaking enhancement**

```python
def _create_bbob_evaluator(**kwargs):
    # NEW: Extract Option C parameter
    _resolved_data = kwargs.pop("_resolved_data", None)
    
    if _resolved_data is not None:
        # NEW: Use precomputed data (if applicable)
        # For BBOB, we might not use _resolved_data yet
        # but the pattern enables it for future evaluators
        pass
    
    # EXISTING: Direct parameters (unchanged)
    config = BBOBConfig(
        fn_name=kwargs.get("fn_name", "sphere"),
        num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
        maximize=kwargs.get("maximize", True),
        seed=kwargs.get("seed", 42),
    )
    return BBOBEvaluator.create(config)

# Apply same pattern to:
# - _create_knapsack_evaluator
# - _create_binary_sum_evaluator
# - _create_griewank_evaluator
# - _create_sphere_evaluator
# (All get the defensive .pop("_resolved_data", None) at start)

# Status: ✅ BACKWARD COMPATIBLE
# - .pop() is safe (removes nothing if key absent)
# - Existing calls work
# - Future evaluators can use _resolved_data
```

**Checklist:**
- [ ] Add to `_create_bbob_evaluator()`: `_resolved_data = kwargs.pop("_resolved_data", None)`
- [ ] Add to `_create_knapsack_evaluator()`: same
- [ ] Add to `_create_binary_sum_evaluator()`: same
- [ ] Add to `_create_griewank_evaluator()`: same
- [ ] Add to `_create_sphere_evaluator()`: same
- [ ] Add tests for backward compat

**Lines: ~3 additions per factory (15 total)**

---

#### 5. `tests/composer/test_catalog_registry.py` (Existing Tests)
**Impact: Add new tests, keep all old ones**

```python
# NEW TESTS (don't remove existing ones)

def test_catalog_get_with_data_registry():
    """Test data_id resolution"""
    catalog = OperatorCatalog()
    data_reg = {"sphere_10": {...}}
    evaluator = catalog.get("sphere:data_id=sphere_10", data_registry=data_reg)
    assert evaluator is not None

def test_catalog_get_backward_compat_no_registry():
    """Test old behavior unchanged"""
    catalog = OperatorCatalog()
    evaluator = catalog.get("sphere:dim=10")  # No data_registry param
    assert evaluator is not None

def test_data_registry_missing_id_raises():
    """Test error handling"""
    catalog = OperatorCatalog()
    data_reg = {}
    with pytest.raises(KeyError):
        catalog.get("sphere:data_id=missing", data_registry=data_reg)

# Status: ✅ NO BREAKING CHANGES
# - All old tests still run and pass
# - New tests validate new feature
```

**Checklist:**
- [ ] Add `test_catalog_get_with_data_registry()`
- [ ] Add `test_catalog_get_backward_compat_no_registry()`
- [ ] Add `test_data_registry_missing_id_raises()`
- [ ] Add `test_load_config_data_registry_parsing()`
- [ ] Add `test_load_config_unpacking_backward_compat()`

**Lines: ~40 additions, 0 removals**

---

### New Files (3 files)

#### 6. `src/malthusjax/benchmarking/io.py`
**NEW: Data loading module**

```python
"""Data loading utilities for external files."""

from pathlib import Path
from typing import Any, Dict, Union, Optional
import jax.numpy as jnp
import numpy as np

class DataLoader:
    """Universal data loader for various formats."""
    
    @staticmethod
    def load_csv(path: Union[str, Path]) -> jnp.Array:
        """Load CSV file."""
        ...
    
    @staticmethod
    def load_npz(path: Union[str, Path]) -> Dict[str, jnp.Array]:
        """Load .npz archive."""
        ...
    
    @staticmethod
    def load_tsplib(path: Union[str, Path]) -> jnp.Array:
        """Load TSPLib distance matrix."""
        ...
    
    @classmethod
    def load_any(cls, path: Union[str, Path]) -> Union[jnp.Array, Dict[str, jnp.Array]]:
        """Auto-detect format and load."""
        ...
```

**Checklist:**
- [ ] Create file
- [ ] Implement `load_csv()`
- [ ] Implement `load_npz()`
- [ ] Implement `load_tsplib()` (basic)
- [ ] Implement `load_any()`
- [ ] Add error handling
- [ ] Add tests

**Lines: ~100**

---

#### 7. `src/malthusjax/benchmarking/registry.py`
**NEW: Data registry management**

```python
"""Data registry for experiment configurations."""

from typing import Dict, Any, Optional
from pathlib import Path
import json

class DataRegistry:
    """Manage data sources for evaluators."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
    
    def add_synthetic(self, data_id: str, config: Dict[str, Any]) -> None:
        """Register synthetic data source."""
        ...
    
    def add_file(self, data_id: str, file_path: str) -> None:
        """Register file-based data source."""
        ...
    
    def resolve(self, data_id: str) -> Any:
        """Load and return data by ID."""
        ...
```

**Checklist:**
- [ ] Create file
- [ ] Implement `DataRegistry` class
- [ ] Implement `add_synthetic()`
- [ ] Implement `add_file()`
- [ ] Implement `resolve()`
- [ ] Add tests

**Lines: ~80**

---

#### 8. `src/malthusjax/core/fitness/tsp_evaluator.py`
**NEW: Example data-driven evaluator**

```python
"""TSP (Traveling Salesman Problem) fitness evaluator."""

from typing import Any, Optional
import jax.numpy as jnp
import chex
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig

@struct.dataclass
class TSPConfig(BaseEvaluatorConfig):
    """TSP configuration."""
    num_cities: int = 50
    random_seed: int = 42
    maximize: bool = False  # Minimization

@struct.dataclass
class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    """TSP evaluator with synthetic/file support."""
    
    config: TSPConfig
    distance_matrix: chex.Array = struct.field(pytree_node=False)
    
    @classmethod
    def create_synthetic(cls, config: TSPConfig) -> TSPEvaluator:
        """Create synthetic TSP instance."""
        ...
    
    @classmethod
    def create_from_matrix(cls, config: TSPConfig, distance_matrix: chex.Array) -> TSPEvaluator:
        """Create from distance matrix."""
        ...
    
    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate tour distance."""
        ...
```

**Checklist:**
- [ ] Create file
- [ ] Implement `TSPConfig`
- [ ] Implement `TSPEvaluator`
- [ ] Implement `create_synthetic()`
- [ ] Implement `create_from_matrix()`
- [ ] Implement `evaluate()`
- [ ] Add tests

**Lines: ~150**

---

#### 9. `tests/benchmarking/test_io.py`
**NEW: Data loading tests**

```python
def test_load_csv():
    """CSV loading works"""
    ...

def test_load_npz():
    """NPZ loading works"""
    ...

def test_load_auto_detect():
    """Auto-detection works"""
    ...

def test_missing_file_error():
    """Missing files raise error"""
    ...
```

**Checklist:**
- [ ] Create test file
- [ ] Add CSV loading tests
- [ ] Add NPZ loading tests
- [ ] Add format auto-detection tests
- [ ] Add error handling tests

**Lines: ~60**

---

### Filesystem Changes (1 directory)

#### 10. `data/` Directory
**NEW: Sample data files**

```
data/
├── README.md
├── tsp/
│   ├── berlin52.tsp         # TSPLib format
│   └── README.md
├── knapsack/
│   └── kp_100.csv
└── synthetic_examples/
    └── README.md
```

**Checklist:**
- [ ] Create `data/` directory
- [ ] Add sample TSP file (or reference to download)
- [ ] Add `data/README.md` with sources
- [ ] Add license information

---

### Documentation Changes (4 files)

#### 11. `README.md` (Update)
**Impact: Informational, no breaking changes**

```markdown
## Data-Driven Evaluators (v3.0+)

MalthusJAX now supports both:
- **Direct parameters**: `fitness="sphere:dim=10"` (v2.0)
- **Data IDs**: `fitness="sphere:data_id=sphere_10"` (v3.0+)

### Example: Using data configs

```python
result = composer.quick_run(
    fitness="sphere:data_id=sphere_10",
    data_config={...},
)
```

See [Option C Guide](OPTION_C_GUIDE.md) for details.
```

**Checklist:**
- [ ] Add section on data-driven evaluators
- [ ] Add quick example
- [ ] Link to detailed documentation

---

#### 12. `OPTION_C_GUIDE.md`
**NEW: User guide for Option C**

```markdown
# Option C: Data-Driven Evaluators Guide

## Overview

Option C enables reproducible optimization by decoupling data specifications from evaluator logic.

## Usage

### Synthetic (same as before)
```python
result = composer.quick_run(fitness="sphere:dim=10")
```

### Data ID (new feature)
```python
result = composer.quick_run(
    fitness="tsp:data_id=berlin52",
    data_config={"berlin52": {...}}
)
```

## TOML Format

See [OPTION_C_EXECUTION_PATHS.md](OPTION_C_EXECUTION_PATHS.md) for detailed examples.
```

**Checklist:**
- [ ] Create guide
- [ ] Add usage examples
- [ ] Add TOML examples
- [ ] Add migration guide

---

#### 13. `BACKWARD_COMPATIBILITY.md` (Update)
**Summary of compatibility guarantees**

```markdown
# Backward Compatibility Statement

MalthusJAX v3.0 maintains 100% backward compatibility with v2.0.

## Guaranteed to Work

- ✅ All v2.0 TOML files work unchanged
- ✅ All v2.0 Python code works unchanged
- ✅ `composer.quick_run()` API unchanged
- ✅ `catalog.get()` API unchanged (extended with optional kwarg)
- ✅ All fitness evaluators work as before

## New Features (Opt-In)

- 🆕 Data configurations: `[data.*]` sections in TOML
- 🆕 DataID references: `fitness="sphere:data_id=..."`
- 🆕 External data loading: `file_path` parameter support
```

**Checklist:**
- [ ] Update file with v3.0 promises
- [ ] Guarantee old code works
- [ ] Highlight new opt-in features

---

#### 14. `CHANGELOG.md`
**NEW: Version history**

```markdown
# Changelog

## v3.0 (2026-04-XX)

### New Features
- Data-driven evaluators with DataID support
- TOML `[data.*]` section support
- External data loading (CSV, TSPLib, NPZ)
- TSP evaluator
- 100% backward compatible with v2.0

### Changes
- `OperatorCatalog.get()` now accepts optional `data_registry` kwarg
- `load_experiment_config()` returns `ExperimentLoadResult` (backward compat unpacking support)
- New module: `src/malthusjax/benchmarking/io.py`

### Fixes
- None (breaking changes mitigated)

### Deprecations
- None (old API still fully supported)

## v2.0 (2026-03-31)
- Genome catalog
- Binary genome support
- ...
```

**Checklist:**
- [ ] Create file
- [ ] Document v3.0 features
- [ ] Document any deprecations (none planned)

---

### Example/Config Files (2 files)

#### 15. `examples/tsp_experiment.toml`
**NEW: Example TSP config**

```toml
[experiment]
name = "tsp_demo"

[data.berlin52_synthetic]
source = "synthetic"
generator = "tsp"
num_cities = 52
random_seed = 42

[experiment.shared]
fitness = "tsp:data_id=berlin52_synthetic"
genome = "real:dim=52,bounds=(0,1)"
pop_size = 100
generations = 50

[pipelines.ga_baseline]
selection = "tournament:tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"
```

**Checklist:**
- [ ] Create example TOML
- [ ] Make it runnable

---

#### 16. `examples/run_tsp_demo.py`
**NEW: Example runner**

```python
"""Example: Running TSP optimization with Option C."""

from malthusjax.composer import Composer
from malthusjax.composer.config import load_experiment_config

def main():
    # Load config with data sections
    result = load_experiment_config("examples/tsp_experiment.toml")
    
    composer = Composer()
    for pipeline_name, kwargs in result.pipelines.items():
        exec_result = composer.quick_run(
            data_config=result.data_registry,  # NEW
            **kwargs
        )
        print(f"Pipeline {pipeline_name}: {exec_result.aggregated_summary()}")

if __name__ == "__main__":
    main()
```

**Checklist:**
- [ ] Create example script
- [ ] Make it executable with `python examples/run_tsp_demo.py`

---

## Implementation Timeline

### Phase 1: Core Backward-Compatible Infrastructure (3 hours)
**Files: 1, 2, 3, 4, 5**

```
- Modify catalog.py: Add data_registry parameter
- Modify config.py: Add ExperimentLoadResult class
- Modify composer.py: Add data_config parameter
- Modify fitness factories: Add .pop("_resolved_data")
- Add comprehensive tests
```

**Deliverable:** Infrastructure ready, all old tests pass, no breaking changes

**Testing:**
```bash
pytest tests/composer/test_catalog_registry.py -v
pytest tests/composer/test_composer.py -v
# All should pass unchanged
```

---

### Phase 2: New Modules & Examples (2.5 hours)
**Files: 6, 7, 8, 9, 10, 15, 16**

```
- Create benchmarking/io.py: DataLoader
- Create benchmarking/registry.py: DataRegistry
- Create core/fitness/tsp_evaluator.py: TSP
- Create tests for new modules
- Create example config and script
- Add sample data files
```

**Deliverable:** TSP evaluator works with data configs

**Testing:**
```bash
python examples/run_tsp_demo.py  # Should complete successfully
pytest tests/benchmarking/test_io.py -v
```

---

### Phase 3: Documentation (1.5 hours)
**Files: 11, 12, 13, 14**

```
- Update README.md
- Create OPTION_C_GUIDE.md
- Update BACKWARD_COMPATIBILITY.md
- Create CHANGELOG.md
```

**Deliverable:** Complete documentation

---

### Phase 4: Integration & QA (1.5 hours)
**All files**

```
- Run full test suite
- Manual testing of examples
- Verify backward compat with v2.0 configs
- Performance testing (ensure no regressions)
```

**Testing:**
```bash
pytest tests/ -v  # All tests pass
pytest tests/composer/ -v
pytest tests/benchmarking/ -v
python examples/mock_binary_experiment.toml  # Old config works
python examples/run_tsp_demo.py  # New feature works
```

---

## Total Effort Estimate

| Phase | Task | Hours | Files | Complexity |
|-------|------|-------|-------|-----------|
| 1 | Infrastructure | 3.0 | 5 | Medium |
| 2 | New Modules | 2.5 | 7 | Medium |
| 3 | Documentation | 1.5 | 4 | Low |
| 4 | QA/Integration | 1.5 | — | Low |
| **TOTAL** | **Option C Full** | **8.5** | **16+** | **Well-Managed** |

---

## Risk Mitigation Checklist

### Pre-Implementation
- [ ] Review all current tests (ensure they'll still pass)
- [ ] Document current API contracts
- [ ] Create branch: `feat/option-c-data-driven`

### During Implementation
- [ ] Run tests after each phase
- [ ] Commit with atomic changes (one feature per commit)
- [ ] Add tests alongside code (TDD approach)
- [ ] No merges until full test suite passes

### Post-Implementation
- [ ] Run full CI/CD pipeline
- [ ] Manual smoke testing
- [ ] Update version number (v2.0 → v3.0)
- [ ] Tag release
- [ ] Update release notes

---

## Rollback Plan

If issues found:
```bash
git revert <commit-hash>
# All changes are locally isolated, easy to revert
# Old tests would have caught any issues
```

---

## Success Criteria

✅ **Backward Compatibility**
- All v2.0 tests pass unchanged
- Old TOML files work without modification
- Old Python code works without modification

✅ **New Functionality**
- Data IDs can be referenced in fitness specs
- TOML supports `[data.*]` sections
- TSP evaluator works end-to-end
- Examples run successfully

✅ **Code Quality**
- 100+ new tests added
- All tests pass
- Coverage maintained (no regression)
- Documentation complete

