# Composer Data Flow Analysis: Complete Guide

## Overview
This document provides a comprehensive analysis of how the Composer module handles data in MalthusJAX, covering data configuration parameters, TOML parsing, data registry integration, and the complete data flow to fitness evaluators.

---

## 1. How `data_config` Parameter Works in `quick_run()`

### Quick Run Method Signature
```python
# From src/malthusjax/composer/composer.py line ~125

def quick_run(
    self,
    # ... other parameters ...
    data_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ExperimentResult:
```

### Data Config Processing Flow

The `data_config` parameter is passed through the following chain:

```python
# Step 1: quick_run receives data_config
# Step 2: For malthusjax backend, it's passed to _build_real_engine()
if self._has_real_operators(...):
    engine = self._build_real_engine(
        # ...
        data_config=data_config,  # <-- passed here
        **kwargs,
    )
```

**Location**: [src/malthusjax/composer/composer.py](src/malthusjax/composer/composer.py#L425) lines 420-430

---

## 2. How `[data.*]` TOML Sections Are Parsed

### TOML Parsing Function
```python
# From src/malthusjax/composer/config.py

def _parse_data_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract [data.*] sections from TOML."""
    data_registry: Dict[str, Any] = {}
    data_section = cfg.get("data", {})
    if not isinstance(data_section, dict):
        return data_registry

    for data_id, data_config in data_section.items():
        if isinstance(data_config, dict):
            data_registry[data_id] = data_config
    return data_registry
```

**Location**: [src/malthusjax/composer/config.py](src/malthusjax/composer/config.py#L57) lines 57-67

### Load Experiment Config Return Value
```python
def load_experiment_config(
    path: str,
    pipelines: Optional[List[str]] = None,
) -> ExperimentLoadResult:
    """Returns: ExperimentLoadResult with:
    - meta: Dict with experiment metadata
    - pipelines: Dict of pipeline configurations
    - data_registry: Dict of parsed [data.*] sections
    """
```

**Location**: [src/malthusjax/composer/config.py](src/malthusjax/composer/config.py#L76) lines 76-137

### Example TOML Format
```toml
# examples/tsp_experiment.toml
[data.berlin52_synthetic]
source = "synthetic"
generator = "tsp"
num_cities = 52
random_seed = 42

[data.test_file]
source = "file"
path = "data/tsp/test3.tsp"
```

**Supported TOML data section keys:**
- `source`: `"synthetic"` or `"file"` (required)
- `path`: File path (required if `source="file"`)
- `generator`: Evaluator type (optional, for synthetic)
- Additional evaluator-specific parameters (e.g., `num_cities`, `random_seed` for TSP)

---

## 3. Complete Data Flow: `quick_run()` → Fitness Evaluators

### Full Data Flow Diagram

```
quick_run(data_config={...})
    ↓
_build_real_engine(data_config=data_config)
    ↓
_build_data_registry(data_config)
    ├─→ DataRegistry().register() for each data_id
    └─→ DataRegistry().resolve() for each data_id
        ├─→ source="file" → DataLoader.load_any(path)
        └─→ source="synthetic" → return config dict
    ↓
Returns: resolved_data = {data_id: resolved_data_object}
    ↓
catalog.get(spec, data_registry=resolved_data)
    ├─→ Parse fitness spec: "sphere:data_id=berlin52_synthetic"
    ├─→ Extract data_id from params
    ├─→ Look up in data_registry: resolved_data["berlin52_synthetic"]
    └─→ Pass to factory as _resolved_data=<resolved_data>
        ↓
    Factory function (_create_tsp_evaluator, etc.)
    └─→ Receives kwargs with _resolved_data key
        └─→ Creates evaluator with data: _resolved_data
```

### Step-by-Step Implementation

#### Step 1: Data Registry Building
```python
# From src/malthusjax/composer/composer.py lines 803-813

def _build_data_registry(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a configuration of data sources into a resolved data registry."""
    from malthusjax.benchmarking.registry import DataRegistry

    reg = DataRegistry()
    for data_id, data_spec in data_config.items():
        reg.register(data_id, data_spec)

    resolved: Dict[str, Any] = {}
    for data_id in data_config.keys():
        resolved[data_id] = reg.resolve(data_id)  # <-- Loading happens here
    return resolved
```

#### Step 2: DataRegistry Resolution
```python
# From src/malthusjax/benchmarking/registry.py lines 18-36

class DataRegistry:
    def resolve(self, data_id: str) -> Any:
        """Load and return data by ID depending on its source."""
        config = self._registry.get(data_id)
        source = config.get("source", "synthetic")

        if source == "file":
            path = config.get("path")
            return DataLoader.load_any(path)  # <-- File loading
        elif source == "synthetic":
            return config  # <-- Return config for synthetic generation
        else:
            raise ValueError(f"Unknown data source type '{source}'")
```

#### Step 3: Catalog Resolution with Data Registry
```python
# From src/malthusjax/composer/catalog.py lines 182-199

def get(self, spec: str, data_registry: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve spec to a configured operator instance."""
    operator_type, user_params = self.parse_spec(spec)

    if data_registry is not None and "data_id" in user_params:
        data_id = user_params.pop("data_id")
        if data_id not in data_registry:
            raise KeyError(f"Data ID '{data_id}' not in registry")
        user_params["_resolved_data"] = data_registry[data_id]  # <-- Inject data

    factory, defaults = self._registry[operator_type]
    merged = {**defaults, **user_params}
    return factory(**merged)  # <-- Pass to factory
```

#### Step 4: Evaluator Factory Receives Data
```python
# From src/malthusjax/core/fitness/__init__.py lines 124-148

def _create_tsp_evaluator(**kwargs: Any) -> "TSPEvaluator":
    from .tsp_evaluator import TSPEvaluator

    _resolved_data = kwargs.pop("_resolved_data", None)  # <-- Extract data

    if _resolved_data is not None:
        # If it's a dict holding data source specs (synthetic)
        if isinstance(_resolved_data, dict) and _resolved_data.get("source") == "synthetic":
            num_cities = _resolved_data.get("num_cities", kwargs.get("num_cities", 50))
            seed = _resolved_data.get("random_seed", kwargs.get("seed", 42))
            return TSPEvaluator.create_synthetic(num_cities=num_cities, seed=seed)

        # If it's an array (loaded from file)
        distance_matrix = _resolved_data
        if hasattr(distance_matrix, "distance_matrix"):
            distance_matrix = distance_matrix.distance_matrix
        return TSPEvaluator.create_from_data(kwargs, distance_matrix)

    # Fallback: no data provided
    num_cities = kwargs.get("num_cities", 50)
    seed = kwargs.get("seed", 42)
    return TSPEvaluator.create_synthetic(num_cities=num_cities, seed=seed)
```

---

## 4. Examples of Custom Data Being Passed to Evaluators

### Example 1: TSP with Synthetic Data
```python
# Programmatic approach
composer = Composer.create_default()
result = composer.quick_run(
    fitness="tsp:data_id=berlin52",
    data_config={
        "berlin52": {
            "source": "synthetic",
            "num_cities": 52,
            "random_seed": 42,
        }
    },
    genome="real:dim=52,bounds=(0,1)",
    selection="tournament:tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    pop_size=100,
    generations=50,
)
```

### Example 2: TSP with File-Based Data
```python
# From examples/tsp_file_experiment.toml
[data.test_file]
source = "file"
path = "data/tsp/test3.tsp"

# In Python:
result = Composer.from_toml("examples/tsp_file_experiment.toml")
```

**Supported file formats for TSP**: `.tsp`, `.csv`, `.npz`

### Example 3: Knapsack with Direct Config
```python
# Knapsack doesn't need external data (weights/values are in config)
result = composer.quick_run(
    fitness="knapsack:capacity=100,num_items=20",
    genome="binary:length=20",
    # ...
)
```

**Note**: Knapsack parameters are generated in the factory if not provided:
```python
def _create_knapsack_evaluator(**kwargs: Any) -> "KnapsackEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.setdefault("maximize", True)
    config = KnapsackConfig(**kwargs)
    return KnapsackEvaluator(config)
```

### Example 4: Binary Sum (OneMax)
```python
result = composer.quick_run(
    fitness="binary_sum",
    genome="binary:length=100",
    # No data needed; just counts bits
)
```

### Example 5: BBOB Functions
```python
result = composer.quick_run(
    fitness="sphere:dim=10",  # or rastrigin, griewank, etc.
    # No data_config needed; functions are purely computational
)
```

---

## 5. Composer Integration with DataRegistry from Benchmarking Module

### DataRegistry Class

**Location**: [src/malthusjax/benchmarking/registry.py](src/malthusjax/benchmarking/registry.py)

```python
from malthusjax.benchmarking.registry import DataRegistry

class DataRegistry:
    """Manage data sources for evaluators."""
    
    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}
    
    def register(self, data_id: str, config: Dict[str, Any]) -> None:
        """Register a data source configuration."""
        self._registry[data_id] = config
    
    def resolve(self, data_id: str) -> Any:
        """Load and return data by ID depending on its source."""
        config = self._registry.get(data_id)
        source = config.get("source", "synthetic")
        
        if source == "file":
            path = config.get("path")
            return DataLoader.load_any(path)
        elif source == "synthetic":
            return config
        else:
            raise ValueError(f"Unknown data source type '{source}'")
```

### How Composer Uses DataRegistry

1. **In `_build_data_registry()`**: Composer instantiates a DataRegistry, registers all data_config entries, and resolves them
2. **In `_build_real_engine()`**: The resolved data is passed to `catalog.get()` as `data_registry` parameter
3. **In `catalog.get()`**: If a spec includes `data_id=..._`, the catalog looks up the resolved data and injects it as `_resolved_data` to the factory

### Integration Points

| Component | Responsibility |
|-----------|-----------------|
| `DataRegistry` | Manage data source configurations and resolution |
| `DataLoader` | Load data from files (CSV, NPZ, TSP) or return config for synthetic |
| `OperatorCatalog.get()` | Resolve `data_id` specs and inject data into factories |
| `Evaluator Factories` | Receive `_resolved_data` and create evaluators |
| `Evaluators` | Store and use data in `.evaluate()` and `.evaluate_population()` |

---

## 6. Data Types and Structures Expected

### BaseEvaluator Generic Type System

```python
# From src/malthusjax/core/fitness/base.py

@struct.dataclass
class BaseEvaluator(Generic[G, C, D]):
    """JAX-native fitness evaluation interface.
    
    Type Parameters:
        G: Genome type (e.g., RealGenome, BinaryGenome)
        C: Config type (static across vmap)
        D: Data type (e.g., training data, problem parameters; static)
    """
    config: C      # Static configuration
    data: D        # Static data (problem-specific)
```

### Data Type Examples by Evaluator

#### TSPEvaluator
```python
# Type: BaseEvaluator[RealGenome, TSPConfig, chex.Array]

class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    config: TSPConfig
    data: chex.Array = struct.field(pytree_node=False)  # Distance matrix
    
    # Expected data shape: (num_cities, num_cities)
    # Created from: _create_tsp_evaluator receives either
    #   1. dict: {"source": "synthetic", "num_cities": 52, ...}
    #   2. ndarray: computed distance matrix from file
```

#### KnapsackEvaluator
```python
# Type: BaseEvaluator[BinaryGenome, KnapsackConfig, Any]

@struct.dataclass
class KnapsackConfig(BaseEvaluatorConfig):
    weights: chex.Array        # Item weights, shape (n_items,)
    values: chex.Array         # Item values, shape (n_items,)
    capacity: chex.Numeric     # Maximum weight capacity
    penalty_factor: float      # Constraint violation penalty

class KnapsackEvaluator(BaseEvaluator[BinaryGenome, KnapsackConfig, Any]):
    config: KnapsackConfig
    data: Any = struct.field(pytree_node=False, default=None)  # Not used
```

#### SphereEvaluator
```python
# Type: BaseEvaluator[RealGenome, SphereConfig, Any]

class SphereEvaluator(BaseEvaluator[RealGenome, SphereConfig, Any]):
    config: SphereConfig
    data: Any = struct.field(pytree_node=False, default=None)  # No external data
```

### Data Config Structure Expected

#### For Synthetic Data (source="synthetic")
```python
{
    "data_id_1": {
        "source": "synthetic",
        "generator": "tsp",  # optional
        "num_cities": 52,
        "random_seed": 42,
    }
}
```

#### For File-Based Data (source="file")
```python
{
    "data_id_1": {
        "source": "file",
        "path": "data/tsp/test.tsp",  # .tsp, .csv, or .npz
    }
}
```

### DataLoader Supported Formats

**Location**: [src/malthusjax/benchmarking/io.py](src/malthusjax/benchmarking/io.py#L133)

```python
class DataLoader:
    @staticmethod
    def load_csv(path: Path | str) -> chex.Array:
        """Load CSV file into a JAX array."""
        # Returns: ndarray or jax array
    
    @staticmethod
    def load_npz(path: Path | str) -> dict[str, chex.Array]:
        """Load .npz archive into JAX arrays."""
        # Returns: {key: jax_array, ...}
    
    @staticmethod
    def load_tsplib(path: Path | str) -> chex.Array:
        """Load TSPLib distance matrix (EUC_2D format)."""
        # Returns: (num_cities, num_cities) distance matrix
    
    @classmethod
    def load_any(cls, path: Path | str) -> chex.Array | dict[str, chex.Array]:
        """Auto-detect format and load."""
        # Auto-detection based on file extension
```

---

## 7. Complete Data Flow Example: TSP with File Data

### Example Walkthrough

**TOML Configuration**:
```toml
# experiment.toml
[experiment.shared]
fitness = "tsp:data_id=custom_tsp"
genome = "real:dim=52,bounds=(0,1)"

[data.custom_tsp]
source = "file"
path = "data/tsp/berlin52.tsp"
```

**Code Execution**:
```python
result = Composer.from_toml("experiment.toml")
```

**Internal Flow**:

1. **Config Loading** (`config.py:load_experiment_config`):
   ```
   Read TOML → {
     "experiment": [...],
     "data": {"custom_tsp": {"source": "file", "path": "data/tsp/berlin52.tsp"}},
     "pipelines": [...]
   }
   _parse_data_section() extracts:
   data_registry = {"custom_tsp": {"source": "file", "path": "data/tsp/berlin52.tsp"}}
   ```

2. **Composer.quick_run** receives in `compare()`:
   ```
   quick_run(
       fitness="tsp:data_id=custom_tsp",
       data_config={"custom_tsp": {"source": "file", "path": "data/tsp/berlin52.tsp"}},
       ...
   )
   ```

3. **Data Registry Building** (`_build_real_engine` → `_build_data_registry`):
   ```
   DataRegistry().register("custom_tsp", {"source": "file", "path": "..."})
   resolved["custom_tsp"] = DataRegistry().resolve("custom_tsp")
       → DataLoader.load_tsplib("data/tsp/berlin52.tsp")
       → (52, 52) distance matrix as jax array
   
   Returns: {"custom_tsp": <distance_matrix_array>}
   ```

4. **Catalog Resolution** (`catalog.get`):
   ```
   catalog.get(
       "tsp:data_id=custom_tsp",
       data_registry={"custom_tsp": <distance_matrix>}
   )
   
   Parse spec: operator_type="tsp", params={"data_id": "custom_tsp"}
   Extract: user_params["_resolved_data"] = <distance_matrix>
   Call factory: _create_tsp_evaluator(_resolved_data=<distance_matrix>, ...)
   ```

5. **Factory Creates Evaluator** (`_create_tsp_evaluator`):
   ```python
   _resolved_data = <distance_matrix>
   
   if isinstance(_resolved_data, dict) and _resolved_data.get("source") == "synthetic":
       # Synthetic path (not taken)
   else:
       # File path (taken)
       distance_matrix = _resolved_data
       return TSPEvaluator.create_from_data(kwargs, distance_matrix)
           → TSPEvaluator(config=TSPConfig(num_cities=52), data=distance_matrix)
   ```

6. **Evaluator Uses Data** (`evaluate`):
   ```python
   def evaluate(self, genome: RealGenome) -> chex.Numeric:
       tour = jnp.argsort(genome.values)
       tour_shifted = jnp.roll(tour, shift=-1)
       distances = self.data[tour, tour_shifted]  # <-- Uses loaded distance matrix
       total_distance = jnp.sum(distances)
       return total_distance
   ```

---

## 8. Function Reference

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `Composer.quick_run()` | [composer.py:125](src/malthusjax/composer/composer.py#L125) | Entry point for experiments with optional `data_config` |
| `Composer._build_real_engine()` | [composer.py:815](src/malthusjax/composer/composer.py#L815) | Builds engine and resolves data registry |
| `Composer._build_data_registry()` | [composer.py:803](src/malthusjax/composer/composer.py#L803) | Converts data_config to resolved data |
| `load_experiment_config()` | [config.py:76](src/malthusjax/composer/config.py#L76) | Parses TOML and extracts data sections |
| `_parse_data_section()` | [config.py:57](src/malthusjax/composer/config.py#L57) | Extracts `[data.*]` TOML sections |
| `OperatorCatalog.get()` | [catalog.py:182](src/malthusjax/composer/catalog.py#L182) | Resolves operator specs with optional data_registry |
| `DataRegistry.resolve()` | [registry.py:18](src/malthusjax/benchmarking/registry.py#L18) | Loads data from file or returns synthetic config |
| `DataLoader.load_any()` | [io.py:213](src/malthusjax/benchmarking/io.py#L213) | Auto-detects and loads data files |
| `BaseEvaluator.evaluate()` | [base.py:40](src/malthusjax/core/fitness/base.py#L40) | Uses `self.data` for fitness computation |

### Evaluator Creation Factories

| Factory | Location | Supports Data |
|---------|----------|---------------|
| `_create_tsp_evaluator()` | [__init__.py:124](src/malthusjax/core/fitness/__init__.py#L124) | ✓ Yes (file/synthetic) |
| `_create_knapsack_evaluator()` | [__init__.py:102](src/malthusjax/core/fitness/__init__.py#L102) | ✗ No (config-based) |
| `_create_bbob_evaluator()` | [__init__.py:87](src/malthusjax/core/fitness/__init__.py#L87) | ✗ No (computational) |
| `_create_binary_sum_evaluator()` | [__init__.py:110](src/malthusjax/core/fitness/__init__.py#L110) | ✗ No (no external data) |

---

## 9. Important Notes

### Current Limitations

1. **From TOML**: The `data_registry` returned by `load_experiment_config()` is parsed but **not passed to `quick_run()` in the `from_toml()` method**. This appears to be a design gap that should be addressed.

2. **TSP Data Only**: Currently, the primary evaluator that uses external data via `data_id` is TSP. Other evaluators receive their parameters directly.

3. **File Format Support**: Limited to `.csv`, `.npz`, and `.tsp` files. Custom loaders would need to extend `DataLoader`.

### Best Practices

1. **Use `data_id` for TSP**: When using TSP evaluators, always use `data_id` parameter to reference data sources
2. **Provide Full Paths**: File paths in `data_config` or TOML should be relative to where the script runs
3. **Seed Control**: Include `random_seed` in synthetic data configs for reproducibility
4. **Type Consistency**: Ensure data dimensions match genome specifications (e.g., TSP cities = genome dimension)

### JAX Compatibility

All data is stored with `pytree_node=False` to remain static across JAX `vmap` operations:
```python
data: chex.Array = struct.field(pytree_node=False)  # Static, not vmapped
```

This ensures efficient vectorized evaluation without redundant data copying.

---

## 10. Summary Table

| Aspect | Implementation |
|--------|-----------------|
| **Data Config Type** | `Optional[Dict[str, Any]]` |
| **TOML Section** | `[data.*]` subsections |
| **Registry Class** | `DataRegistry` from `benchmarking.registry` |
| **Resolution Pattern** | `data_id` → lookup in registry → resolved data |
| **Evaluator Parameter** | `_resolved_data` injected by `catalog.get()` |
| **Storage Location** | `BaseEvaluator.data` field (static) |
| **Supported Formats** | CSV, NPZ, TSP, synthetic config dicts |
| **Primary Use Case** | TSP and other problem-instance evaluators |
| **Factory Responsibility** | Extract `_resolved_data`, create evaluator with `data=...` |

