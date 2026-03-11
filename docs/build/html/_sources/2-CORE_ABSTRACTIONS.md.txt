# Core Abstractions: Genomes, Populations, and Fitness

This section breaks down the `malthusjax.core` module, which contains the fundamental data structures and evaluation protocols. In a JAX-native framework, these are strictly defined as PyTrees to ensure compatibility with XLA compilation and vmap vectorization.

## 2.1. The Genome System (malthusjax.core.genome)

The Genome represents the genetic material of an individual. In MalthusJAX, Genomes are immutable PyTrees wrapping JAX arrays.

### 2.1.1. The Base Genome PyTree

**Concept:** Abstracting array logic while maintaining JAX tracing capabilities.

A Base Genome acts as the foundational building block for all genetic representations in the framework. It encapsulates a JAX array along with a small amount of structural metadata and provides a uniform interface for downstream components such as operators and evaluators. Because mutation or crossover operators may sometimes produce values that violate domain constraints (e.g. out-of-bounds reals or invalid categorical indices), each genome also exposes an `autocorrect(config)` method. This hook can be invoked at any point in the pipeline – from inside a mutation operator, by a custom engine, or manually in user code – and has access to both the genome and the experiment configuration. Implementations can therefore perform arbitrary repair logic or catch aberrant mutations without assuming when or where the call occurs. By treating genomes as first-class PyTrees, the system ensures that every genome can be passed through `jax.jit`, `jax.vmap`, and other transformations without losing information or causing tracing errors.

*Example (autocorrect focus):*
```python
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
import jax

config = RealGenomeConfig(shape=(6,), bounds=(-5.0, 5.0))
g = RealGenome.random_init(jax.random.PRNGKey(0), config)
# custom autocorrect simply clips out-of-bounds values
g2 = g.autocorrect(config)  # built-in clip behavior
# you could subclass RealGenome and override autocorrect

class MyGenome(RealGenome):
    def autocorrect(self, config):
        # e.g. snap-to-nearest-bound instead of clip
        vals = jax.numpy.where(g.values < config.bounds[0],
                               config.bounds[0],
                               jax.numpy.where(g.values > config.bounds[1],
                                               config.bounds[1],
                                               g.values))
        return MyGenome(values=vals)
```



**Knowledge Point:** How the base Genome class registers itself using `jax.tree_util.register_pytree_node`. Separation of structural metadata (e.g., bounds, lengths) from the active tensor payload.

In practice the class implements the PyTree protocol by splitting the object into a "flattened" list of tensors (the payload) and a static metadata tuple. Registration with `jax.tree_util.register_pytree_node` happens at module import time so that the JAX tracer knows how to traverse the type during compilation. This separation allows the metadata — for example, the permissible bounds of a real-valued genome or the dimensionality of a categorical genome — to remain static while the tensor payload is updated by evolutionary operators.

### 2.1.2. Continuous vs. Discrete Representations

**Concept:** Tailoring evolutionary encodings to specific problem domains.

Different application areas require different genetic encodings: continuous parameter optimization, combinatorial search, or categorical decision variables. MalthusJAX provides specialized genome subclasses that enforce the appropriate domain and offer convenience methods for mutation and initialization. This design keeps client code clean while enabling operators to rely on consistent tensor shapes and container types.

**Knowledge Point:** Implementation details of `RealGenome` (floating-point bounds clipping), `BinaryGenome` (boolean arrays, bit-flipping logic), and `CategoricalGenome` (integer bounds).

Each subclass inherits from the base genome and adds its own invariant checks. `RealGenome` stores a `bounds` tuple and wraps generated or mutated values with `jax.numpy.clip` to remain within the allowed range. `BinaryGenome` represents its data as a boolean array and the associated operators use simple bitwise XOR and random flips to implement mutation. `CategoricalGenome` encodes each gene as an integer index within a specified set of categories; random initialization uses `jax.random.randint` and mutation re-samples from the category set. All of these types participate transparently in batching and JIT compilation.

*Example:*
```python
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
key = jax.random.PRNGKey(42)
bg = BinaryGenome.random_init(key, BinaryGenomeConfig(shape=(16,)))
# mutate flips bits probabilistically
bg2 = bg.mutate(jax.random.PRNGKey(1), rate=0.05)
```

### 2.1.3. Tensor Interoperability and Indexing

**Concept:** Seamless conversion between raw arrays and Genome objects.

Users often need to interoperate with raw JAX arrays, for example when loading data from disk or interfacing with other libraries. The genome classes expose factory methods such as `from_tensor` and `to_tensor` that wrap and unwrap the internal array without copying unnecessarily. Because genomes are PyTrees, these methods are also used by `jax.vmap` and `jax.jit` to transform collections of genomes.

*Example:*
```python
import jax.numpy as jnp
from malthusjax.core.genome.real_genome import RealGenome
arr = jnp.linspace(-1, 1, 10)
g = RealGenome.from_tensor(arr, bounds=(-1,1))
print(g.to_tensor())  # should equal arr
```

**Knowledge Point:** Using factory methods (`from_tensor`) and supporting advanced indexing/slicing on Genomes (e.g., `test_genome_indexing.py`) without breaking PyTree structures.

Implementing `__getitem__` and slicing for genome objects required care to maintain the metadata payload. Tests like `test_genome_indexing.py` ensure that slicing a population returns a new genome with the correct shape and metadata, and that the resulting object still behaves as a PyTree. This allows high-level code to treat populations as arrays of genomes while preserving the benefits of the custom types.

### 2.1.4. Population Metrics and KPIs

**Concept:** Built-in support for population‑level statistics and extensible key performance indicators.

While genomes describe individuals, the `BasePopulation` container aggregates them into a coherent batch and adds space for parallel arrays of derived quantities such as fitness values. To support algorithm diagnostics and analysis, the class provides convenience methods like `distance_matrix()` which computes pairwise distances between all members using nested `jax.vmap` calls. The method simply delegates to `genome.distance(g2, metric=…)`, meaning the distance metric itself lives on the genome class and can be overridden or extended. These operations are JIT‑safe and avoid explicit Python loops, making it trivial to inspect population diversity or clustering at any point in an experiment, and because the metric is customizable the resulting matrix can later be consumed by fitness evaluators or evolutionary engines for tasks such as multi‑objective optimization or novelty search.

**Knowledge Point:** Extending the population object for arbitrary KPIs (e.g. diversity, age, novelty) and accessing them.

The dataclass nature of `BasePopulation` means additional metric fields can be added by subclassing or by using the `.replace()` helper. Any statistic over the population – whether a scalar summary or a full matrix – can be computed in a JAX‑friendly way and stored alongside `genes` and `fitness`. Consumers can then access these values with the usual attribute lookup (`pop.my_metric`) and JIT compilation will carry them through the engine. This pattern generalizes to support custom KPIs required by specific research projects without entangling algorithmic code with bookkeeping logic.

*Example:*
```python
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome
import jax

# compute and attach a simple variance metric
pop = BasePopulation(genes=jax.vmap(lambda k: RealGenome.random_init(k, RealGenomeConfig((5,),(-1,1))))(jax.random.split(jax.random.PRNGKey(0), 10)),
                     fitness=jax.numpy.zeros(10))
dists = pop.distance_matrix(metric="euclidean")
pop2 = pop.replace(distance=dists)
print(pop2.distance.shape)  # (10,10)
```

## 2.2. Fitness Evaluators (malthusjax.core.fitness)

Fitness evaluators define the objective function. MalthusJAX separates the evaluation of a single genome from the batched evaluation of a population.

### 2.2.1. The Evaluator Contract

**Concept:** Defining a pure function `score = evaluator(genome)`.

Fitness evaluators are the only part of the framework aware of problem semantics. They must behave as pure functions: given the same genome and inputs, they always return the same scalar score. This property is essential for reproducibility and for enabling JIT compilation and vectorization across populations. Evaluators thus avoid any hidden state or side effects; any required data, such as training examples for a neural network, is supplied explicitly via additional arguments that are marked static so that JAX can trace correctly.

*Example:*
```python
from malthusjax.core.fitness.real_evaluators import SphereEvaluator
import jax

eval = SphereEvaluator(dim=5)
g = jax.random.normal(jax.random.PRNGKey(0),(5,))
print(eval(g))  # sum of squares
```

**Knowledge Point:** Ensuring evaluators are stateless. Handling external data (e.g., training batches for neuroevolution) by passing them as static arguments or `lax.scan` payloads.

For complex evaluators that iterate over datasets or perform multiple sub-computations, the implementation uses `jax.lax.scan` or `jax.vmap` internally. When external datasets are involved they are passed in as part of the evaluator object but flagged with `pack=True` or declared as static in `__call__` signatures so they do not participate in gradients or get copied unnecessarily. The accompanying test suite verifies that evaluators can be serialized by `jax.jit` and reused across seeds without unexpected behavior.

### 2.2.2. Continuous Benchmarking (BBOBEvaluator)

**Concept:** Integrating standard Black-Box Optimization Benchmarks.

To facilitate fair comparisons with the wider optimization literature, MalthusJAX includes an evaluator that wraps the BBOB benchmark suite. These are well‑known mathematical test functions such as Sphere, Rastrigin, and Rosenbrock that challenge optimization algorithms with multimodality, flat regions, and other pathologies. The `BBOBEvaluator` exposes them through a unified interface and can be configured for dimensionality and noise levels.

**Knowledge Point:** Vectorizing complex mathematical landscapes (Rastrigin, Rosenbrock) using JAX primitives (`jax.numpy`).

The evaluator implementations are written entirely with `jax.numpy` and auxiliary functions so that they can be compiled and run on accelerators. A single call accepts a batch of genomes and returns a batch of scores using `jax.vmap`. This vectorization is critical for high throughput benchmarking and for enabling JIT acceleration of the inner loop of evolutionary engines.

*Example:*
```python
import jax
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator

bbob = BBOBEvaluator(name="rastrigin", dim=10)
pop = jax.random.normal(jax.random.PRNGKey(1),(20,10))
scores = jax.vmap(bbob)(pop)
print(scores.shape)  # (20,)
```

## 2.3. Deterministic Stochasticity (malthusjax.core.random)

Randomness is the motor of evolutionary search, but JAX requires explicit state handling for Pseudo-Random Number Generators (PRNGs). This section deserves special attention because **JAX is a relatively young library undergoing rapid API evolution, and random number generation is one of the most volatile areas**. The PRNG interface has already changed significantly between JAX versions: early versions used global state (`numpy.random.seed`), then introduced legacy-style typed keys (uint32[2] tensors), and now support modern pluggable PRNG backends (Threefry, RBG, etc.). 

Currently, MalthusJAX defaults to **Threefry** for maximum portability and bit-for-bit reproducibility across all devices and clusters. However, **Philox is theoretically superior for GPU workloads** (faster iteration, better statistical properties for hardware), but full Philox support has not yet been integrated into the framework's core operators. Since randomness is absolutely vital to genetic algorithm correctness and reproducibility, the random module is architected with **future-proofing as a first-class design goal**: every key creation, validation, and consumption path is abstracted behind the `PRNGImpl` enum and utility functions, allowing seamless migration to new backends as JAX evolves and as GPU optimization becomes a priority. This forward-thinking design prevents lock-in to deprecated APIs while preserving reproducibility guarantees across versions.

### 2.3.1. JAX PRNG Key Philosophy

**Concept:** Escaping the hidden state of standard random libraries (like `numpy.random`).

JAX treats random number generation as a functional transformation: a PRNG key is an explicit tensor that must be passed to every function needing randomness. This avoids implicit global state and makes the computation graph fully deterministic and reproducible. Users must adopt the practice of splitting keys whenever randomness is consumed, ensuring that each random draw uses a fresh subkey. This design enables several critical benefits:

1. **Reproducibility**: Given the same seed, the entire evolutionary trajectory is deterministic
2. **Traceability**: Every stochastic operation is explicitly coupled to a key, making code auditable
3. **JIT Compilation**: Keys can be traced like any other array, so randomness within compiled loops is straightforward
4. **Multi-device Execution**: Key splitting schemes distribute seeds across GPUs/TPUs without collision

Unlike stateful libraries (e.g., `numpy.random.seed()` which modifies global state), JAX's functional approach makes randomness an input and output of every function, ensuring no hidden dependencies.

**Knowledge Point:** The PRNGKey lifecycle: initializing, splitting (`jax.random.split`), and consuming keys.

The lifecycle follows a strict pattern: initialize a master key, split it into subkeys for specific operations, use each subkey exactly once, and pass the updated master key forward. Within the framework, helpers in `core/random.py` standardize key handling. Engines and operators call `key, subkey = jax.random.split(key)` and return the updated key for later use. A typical evolution step splits the master key into multiple children for population initialization, selection, and mutation.

*Example (key lifecycle in evolutionary pseudocode):*
```python
import jax
import jax.random as jr
from malthusjax.core.random import create_key, validate_key

# Create a master key using the framework utility
master_key = create_key(seed=42, impl="threefry")  # or "rbg", etc.
validate_key(master_key, context="engine initialization")

# Split into subkeys for different operations
master_key, pop_key, select_key, mutate_key = jr.split(master_key, 4)

# Each subkey is used for one specific operation
population = initialize_population(pop_key)              # pop_key consumed here
selected = selection_operator(select_key, fitness)      # select_key consumed here
offspring = mutation_operator(mutate_key, selected)     # mutate_key consumed here

# master_key is fresh and ready for the next generation
```

Note that reusing a key accidentally produces identical random draws. The framework includes validation logic to warn users if keys are reused or if legacy-style keys are detected, catching bugs early in development.

### 2.3.2. Framework Random Utilities

**Concept:** Ergonomic wrappers for common evolutionary random tasks.

To reduce boilerplate and prevent key misuse, the module `core/random.py` provides utilities for key creation, validation, and backend selection. These functions enable explicit control over PRNG implementations (e.g., Threefry vs. Philox) and provide deprecation guidance for legacy-style keys introduced in older JAX versions.

**Knowledge Point:** Creating typed keys with `create_key()`, resolving PRNG backends, and validating keys for proper usage patterns.

The module exposes three main utilities:

1. **`create_key(seed, impl)`**: Creates a JAX PRNG key with an explicit backend specification. Supports `"threefry"` (default, portable), `"philox"` (fast on CPU, reliable on all devices), `"rbg"` (hardware-backed on newer systems), and `"unsafe_rbg"`. The function automatically falls back to legacy `jax.random.PRNGKey()` if the new typed key API is unavailable, with a deprecation warning.

2. **`resolve_prng_impl(name)`**: Converts user-facing PRNG names (strings or enum members) to the internal `PRNGImpl` enum. Supports short aliases (e.g., `"threefry"` → `PRNGImpl.THREEFRY`) and full JAX backend strings (e.g., `"threefry2x32"`).

3. **`validate_key(key, context)`**: Checks whether a key is legacy-style (uint32[2]) or new-style typed. Issues a deprecation warning if legacy keys are detected, guiding users toward `create_key()` for explicit backend control.

*Example (key creation and validation):*
```python
from malthusjax.core.random import create_key, validate_key, resolve_prng_impl
import jax.random as jr

# Create a key with explicit PRNG backend
key = create_key(seed=123, impl="philox")
validate_key(key, context="evolution init")

# Resolve user input to backend enum
impl = resolve_prng_impl("threefry")
key2 = create_key(456, impl=impl)

# Use the key for splitting as normal
key, subkey1, subkey2 = jr.split(key, 3)
print(f"Backend is consistent across splits: {subkey1.shape == subkey2.shape}")
```

**Key Budgeting in Evolutionary Algorithms**: A typical evolution step requires multiple random draws: population initialization, selection, crossover, and mutation. Best practice is to budget keys upfront and pass them explicitly:

*Example (key budgeting in an evolution loop):*
```python
def evolution_step(state, key):
    # Split key for each operation
    key, init_key, select_key, cross_key, mutate_key = jr.split(key, 5)
    
    # Or use fold_in for deterministic derivations
    select_key = jr.fold_in(key, state.generation)  # Depends on generation number
    
    # Each operation uses its dedicated key
    # vmap automatically broadcasts across population
    new_pop = jax.vmap(lambda k, g: initialize_genome(k, config), in_axes=(0, None))(
        jr.split(init_key, len(old_pop)), old_pop
    )
    
    return updated_state, key
```

**Why Multiple PRNG Backends?** Different accelerators and use cases benefit from different algorithms:

- **Threefry**: Portable, identical across all platforms (reproducibility across clusters)
- **Philox**: Faster on modern CPUs, reliable on all GPUs; preferred for GPU-intensive work but not yet integrated into JAX
- **RBG**: Hardware-backed entropy (modern systems); extremely fast but less portable
- **Unsafe RBG**: Hardware entropy without quality checks (risky, not recommended for science)

For benchmarking across frameworks, Threefry ensures bit-for-bit reproducibility with other libraries. For production performance on GPU systems, Philox + typed keys is recommended.

**Batching with vmap and Key Families**: When initializing populations or applying operators across multiple individuals, the framework uses `jax.vmap` with pre-split key families:

*Example (batched initialization):*
```python
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
import jax
import jax.random as jr
from malthusjax.core.random import create_key

config = RealGenomeConfig(shape=(10,), bounds=(-5, 5))
master_key = create_key(seed=42)
master_key, pop_key = jr.split(master_key)

# Generate independent keys for each individual
keys = jr.split(pop_key, pop_size=100)

# vmap initializes entire population in parallel
population = jax.vmap(lambda k: RealGenome.random_init(k, config))(keys)
print(population.values.shape)  # (100, 10)
```

This disciplined approach to key handling keeps the core engine code simple and reduces the risk of subtle duplicate-key bugs. All operators and engines follow this pattern, ensuring that randomness is transparent and auditable.
