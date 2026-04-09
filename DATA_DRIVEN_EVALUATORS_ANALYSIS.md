# Data-Driven Evaluators: Architecture Analysis

This document compares three architectural approaches for integrating data-driven fitness problems (e.g., TSP, ML model training) into the MalthusJAX framework.

---

## Context: Current Architecture

### Existing Evaluators (Stateless/Synthetic)
All current evaluators are **stateless**, purely mathematical functions:
- `SphereEvaluator`: $f(x) = \sum x_i^2$
- `BinarySumEvaluator`: OneMax — count 1-bits
- `KnapsackEvaluator`: 0/1 knapsack with random item weights/values
- `BBOBEvaluator`: Wraps evosax BBOB functions

### JAX Constraint
**JAX JIT compilation prohibits dynamic I/O during fitness evaluation.** All data must be loaded *before* compilation and stored as immutable Pytree nodes.

### Current Registry Pattern
```python
def _create_knapsack_evaluator(**kwargs):
    """Factory function called by OperatorCatalog"""
    kwargs.setdefault("maximize", True)
    config = KnapsackConfig(**kwargs)
    return KnapsackEvaluator(config)

# Registered globally
register_table([
    ("knapsack", _create_knapsack_evaluator, {}),
])

# Usage in TOML
# fitness = "knapsack:capacity=100,num_items=20"
```

---

## Option A: Synthetic Generators (Low Effort, Limited Scope)

### Approach
Generate problem instances programmatically from seed/parameters. No external files needed.

### Implementation
```toml
# examples/tsp_experiment.toml
[experiment.shared]
# TSP with 50 cities, randomly generated
fitness = "tsp:num_cities=50,random_seed=42"

# Or Knapsack (already supported)
genome = "binary:length=100"
fitness = "knapsack:num_items=100,capacity_ratio=0.5,random_seed=42"
```

### New Evaluator Example
```python
# src/malthusjax/core/fitness/tsp_evaluator.py
@struct.dataclass
class TSPConfig(BaseEvaluatorConfig):
    num_cities: int = 50
    random_seed: int = 42
    # Distance matrix is computed deterministically from seed
    maximize: bool = False  # Minimization (shortest tour)

@struct.dataclass
class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    """TSP fitness evaluator with procedurally generated distance matrix."""
    
    config: TSPConfig
    distance_matrix: chex.Array = struct.field(pytree_node=False)  # Static
    
    @classmethod
    def create(cls, config: TSPConfig) -> TSPEvaluator:
        """Generate deterministic TSP instance from seed."""
        rng = jax.random.PRNGKey(config.random_seed)
        # Generate random city coordinates
        coords = jax.random.normal(rng, (config.num_cities, 2))
        # Compute pairwise Euclidean distances
        dist_matrix = euclidean_distance_matrix(coords)
        return cls(config=config, distance_matrix=dist_matrix)
    
    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a permutation-based tour against distance matrix."""
        # genome.values is a permutation [0, 1, 2, ..., n-1]
        tour = jnp.argsort(genome.values)  # or decode as permutation
        tour_with_return = jnp.concatenate([tour, tour[:1]])
        
        # Sum distances along tour edges
        distances = self.distance_matrix[tour_with_return[:-1], tour_with_return[1:]]
        total_distance = jnp.sum(distances)
        
        return total_distance  # minimize
```

### Factory Registration
```python
def _create_tsp_evaluator(**kwargs):
    config = TSPConfig(
        num_cities=kwargs.get("num_cities", 50),
        random_seed=kwargs.get("random_seed", 42),
        maximize=False
    )
    return TSPEvaluator.create(config)

register_table([("tsp", _create_tsp_evaluator, {})], override=True)
```

### Pros
- ✅ **Zero file I/O**: Fully self-contained, no data files
- ✅ **Reproducible**: Seed-based problem generation guarantees reproducibility
- ✅ **Simple integration**: Minimal changes to catalog/registry
- ✅ **Fast prototyping**: 1-2 hours to add a new synthetic problem
- ✅ **JAX-native**: No external dependencies

### Cons
- ❌ **Limited scope**: Cannot solve real-world problems (e.g., actual TSP benchmarks like TSPLIB)
- ❌ **Artificial**: Synthetic data may not reflect real problem structure
- ❌ **Scaling issues**: Large synthetic problems (10k cities) can be slow to generate
- ❌ **No benchmark comparison**: Can't reproduce published results on standard datasets

---

## Option B: DataLoader Module (Balanced, Recommended)

### Approach
Build a lightweight `io.py` module that:
1. Loads external data files (CSV, TSPLib, NPZ) at Composer initialization
2. Pre-processes into JAX arrays
3. Injects into evaluator before engine construction

### Structure
```
src/malthusjax/benchmarking/
├── io.py              # NEW: DataLoader for files
└── loader_registry.py # NEW: File format handlers
```

### Implementation
```python
# src/malthusjax/benchmarking/io.py
from pathlib import Path
import jax.numpy as jnp
import numpy as np

class DataLoader:
    """Universal data loader for external files."""
    
    @staticmethod
    def load_csv(path: str | Path) -> jnp.Array:
        """Load CSV file into JAX array."""
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        return jnp.asarray(data)
    
    @staticmethod
    def load_npz(path: str | Path) -> dict:
        """Load .npz (numpy archive) with multiple arrays."""
        data = np.load(path)
        return {k: jnp.asarray(v) for k, v in data.items()}
    
    @staticmethod
    def load_tsplib_distance_matrix(path: str | Path) -> tuple[jnp.Array, str]:
        """Parse TSPLIB distance matrix format."""
        # TSPLib format parsing logic here
        ...
    
    @classmethod
    def load_any(cls, path: str | Path) -> jnp.Array | dict:
        """Auto-detect format and load."""
        path = Path(path)
        if path.suffix == ".csv":
            return cls.load_csv(path)
        elif path.suffix == ".npz":
            return cls.load_npz(path)
        elif path.suffix in [".tsp", ".txt"]:  # TSPLib
            return cls.load_tsplib_distance_matrix(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
```

### TSP Evaluator with File Support
```python
@struct.dataclass
class TSPConfig(BaseEvaluatorConfig):
    num_cities: int = 50
    random_seed: int = 42
    file_path: Optional[str] = None
    maximize: bool = False

@struct.dataclass
class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    config: TSPConfig
    distance_matrix: chex.Array = struct.field(pytree_node=False)
    
    @classmethod
    def create(cls, config: TSPConfig) -> TSPEvaluator:
        """Create from file OR synthetic."""
        if config.file_path:
            # Load external file
            from malthusjax.benchmarking.io import DataLoader
            dist_matrix = DataLoader.load_tsplib_distance_matrix(config.file_path)
        else:
            # Generate synthetic
            rng = jax.random.PRNGKey(config.random_seed)
            coords = jax.random.normal(rng, (config.num_cities, 2))
            dist_matrix = euclidean_distance_matrix(coords)
        
        return cls(config=config, distance_matrix=dist_matrix)
```

### TOML Usage
```toml
# Use synthetic
fitness = "tsp:num_cities=50,random_seed=42"

# OR use external file
fitness = "tsp:file_path=data/berlin52.tsp"

# OR load distance matrix CSV
fitness = "tsp:file_path=data/tsp_100cities_distances.csv"
```

### Composer Integration
```python
# In Composer.quick_run()
if genome is not None:
    # ... existing genome parsing ...

# NEW: resolv file paths for fitness evaluators
if fitness is not None and ":" in fitness and "file_path=" in fitness:
    from malthusjax.benchmarking.io import DataLoader
    fitness_type, fitness_params = parse_spec(fitness)
    
    if "file_path=" in fitness_params:
        file_path = fitness_params["file_path"]
        # Validate file exists before compilation
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        # Pre-validate file format
        DataLoader.load_any(file_path)
```

### Data Directory Structure
```
data/
├── tsp/
│   ├── berlin52.tsp          # TSPLib format
│   ├── tsp_100.csv           # Distance matrix (CSV)
│   └── README.md             # Format docs
├── knapsack/
│   ├── kp_100items.csv       # weights, values columns
│   └── kp_100items_metadata.txt
└── README.md                 # Data sources and licenses
```

### Pros
- ✅ **Flexibility**: Synthetic + file-based in one evaluator
- ✅ **Real benchmarks**: Can hit TSPLIB, standard ML datasets
- ✅ **Reproducibility**: Both paths are deterministic
- ✅ **Clean separation**: I/O logic isolated in `io.py`
- ✅ **Production-ready**: Handles edge cases (missing files, format errors)
- ✅ **Minimal Composer changes**: Falls through existing catalog pattern

### Cons
- ⚠️ **Medium effort**: ~3-4 hours to build DataLoader + multiple evaluators
- ⚠️ **File management**: Need to organize/document data directory
- ⚠️ **License tracking**: External datasets need attribution/licensing
- ⚠️ **Validation overhead**: Must validate files at init time (slows startup slightly)

---

## Option C: Config-Driven Data Pipeline (Heavy, Extensible)

### Approach
Introduce a new `DataConfig` system that:
1. Declares a data source in TOML separately from evaluator
2. Composer resolves `data_id` → DataLoader → Evaluator
3. Enables rich dataset versioning and curation

### New TOML Structure
```toml
[experiment]
name = "tsp_comparative"

[data.berlin52]
source = "file"
path = "data/tsp/berlin52.tsp"
format = "tsplib"
num_cities = 52

[data.random_tsp_100]
source = "synthetic"
generator = "random_euclidean"
num_cities = 100
seed = 42

[experiment.shared]
fitness = "tsp:data_id=berlin52"  # Reference data by ID
genome = "real:dim=52,bounds=(0,1)"

[pipelines.ga_berlin]
engine_type = "ga"
selection = "tournament:tournament_size=3"
crossover = "blend"
mutation = "gaussian"
```

### Evaluator with DataID Resolution
```python
@struct.dataclass
class TSPConfig(BaseEvaluatorConfig):
    data_id: str  # "berlin52" or "random_tsp_100"
    
@struct.dataclass
class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    config: TSPConfig
    distance_matrix: chex.Array = struct.field(pytree_node=False)
    
    @classmethod
    def create_from_registry(cls, config: TSPConfig, data_registry: dict) -> TSPEvaluator:
        """Resolve data from global registry."""
        if config.data_id not in data_registry:
            raise KeyError(f"Data '{config.data_id}' not registered")
        
        dist_matrix = data_registry[config.data_id]
        return cls(config=config, distance_matrix=dist_matrix)
```

### Composer Integration
```python
class Composer:
    def quick_run(self, ..., data_config: Optional[dict] = None):
        # Parse [data.*] sections from TOML
        data_registry = self._build_data_registry(data_config)
        
        # Pass to evaluator factory
        evaluator = OperatorCatalog().get(
            fitness,
            data_registry=data_registry  # NEW arg
        )
```

### Pros
- ✅ **Enterprise-grade**: Rich declarative config for complex setups
- ✅ **Versioning**: Can track dataset versions in TOML
- ✅ **Reproducibility**: Full audit trail of data source + evaluator pairing
- ✅ **Multi-dataset runs**: Compare algorithms across different problem instances
- ✅ **Extensible**: Easy to add dataset curation workflows later

### Cons
- ❌ **High effort**: ~6-8 hours, requires significant refactoring
- ❌ **Complexity**: Adds a new abstraction layer to Composer
- ❌ **Catalog changes**: Requires modifying OperatorCatalog signature
- ❌ **Parsing overhead**: TOML parsing becomes more complex
- ❌ **Risk**: More moving parts = more bugs

---

## Comparative Scorecard

| Criterion | Option A (Synthetic) | Option B (DataLoader) | Option C (DataID) |
|-----------|---|---|---|
| **Effort** | 1-2 hours | 3-4 hours | 6-8 hours |
| **Real-world problems** | ❌ No | ✅ Yes | ✅ Yes |
| **File I/O** | ❌ None | ✅ Simple | ✅ Advanced |
| **JAX compatible** | ✅ Native | ✅ Via pre-load | ✅ Via pre-load |
| **Reproducibility** | ✅✅ | ✅ | ✅✅ |
| **Catalog changes** | Minimal | Minimal | Significant |
| **Quick prototyping** | ✅✅ | ✅ | ❌ Heavy |
| **Production ready** | For synthesis | ✅ | For enterprises |
| **Extensible** | Fair | Good | Excellent |
| **Maintenance burden** | Low | Low-Medium | Medium-High |

---

## Recommendation

### Use Case → Recommended Option

| Use Case | Option | Rationale |
|----------|--------|-----------|
| **Quick POC**: "Can GA solve TSP?" | **Option A** | Fast, no file management, good for demos |
| **Real benchmarking**: Compare vs TSPLIB | **Option B** | Minimal overhead, unlocks real problems |
| **Production ML pipeline**: Optimize NN weights | **Option B** | Clean I/O, simple integration |
| **Enterprise deployment**: Version + audit trails | **Option C** | If time/resources allow |
| **Interactive research**: Notebook exploration** | **Option A+B** | Start with A, extend to B for real data |

---

## Implementation Priority

### Phase 1 (This Session) — **Option A** ✅
- Implement `TSPEvaluator` with synthetic generation
- Register in fitness catalog
- Add example TOML configs
- **Deliverable**: Can optimize random TSP instances

### Phase 2 (Next) — **Option B** 🆗
- Build `src/malthusjax/benchmarking/io.py` DataLoader
- Add TSPLib file parser
- Enhance `TSPEvaluator` to support `file_path` kwarg
- Create `data/` directory with sample files
- **Deliverable**: Can solve TSPLIB benchmark problems

### Phase 3 (Future) — **Option C** 📋
- Introduce `DataConfig` system
- Refactor Composer to use data registry
- Build dataset curation workflows
- **Deliverable**: Enterprise-grade reproducibility framework

---

## Next Steps

1. **Decision**: Which option appeals to you most?
2. **Quick Win**: Start with Option A (synthetic TSP) to establish pattern
3. **Gradual Expansion**: Move to Option B when needed for real data
4. **Maintain optionality**: Design Option A/B so Option C can extend them later

What sounds best to you?
