# Core Abstractions: Genomes, Populations, and Fitness

This section breaks down the `malthusjax.core` module, which contains the fundamental data structures and evaluation protocols. In a JAX-native framework, these are strictly defined as PyTrees to ensure compatibility with XLA compilation and vmap vectorization.

## 2.1. The Genome and Population System (malthusjax.core.genome)

MalthusJAX cleanly separates the definition of a single individual (**Genome**) from the batch of individuals currently undergoing evolution (**Population**). This distinction is critical because algorithmic state (like Pareto ranks or spatial descriptors) belongs to the population structure, not the genetic payload itself.

### 2.1.1. The Base Genome PyTree

**Concept:** Abstracting array logic while maintaining JAX tracing capabilities.

A Base Genome acts as the foundational building block for all genetic representations in the framework. It encapsulates a genetic payload (which may be a single array or a complex tuple of arrays) along with structural metadata. It provides a uniform interface for downstream components such as operators and evaluators. 

Crucially, **operators interact with genomes by flattening them via `jax.tree_util.tree_leaves`**. This allows operators to mutate or recombine the underlying arrays without needing to know the specific fields (like `.values` or `.weights`) defined by the genome class.

Because mutation or crossover operators may sometimes produce values that violate domain constraints (e.g., out-of-bounds reals or invalid categorical indices), each genome also exposes an `autocorrect(config)` method. This hook encapsulates all domain-specific repair logic. Operators do not clip or constrain values; they simply apply raw arithmetic noise, and the engine invokes `autocorrect()` to snap the payload back to valid domain bounds.

*Example (autocorrect focus):*
```python
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
import jax

config = RealGenomeConfig(shape=(6,), bounds=(-5.0, 5.0))
g = RealGenome.random_init(jax.random.PRNGKey(0), config)

# Operators apply unbounded arithmetic noise...
# Then the engine invokes autocorrect to enforce constraints
g_repaired = g.autocorrect(config)  # built-in clip behavior
```

**Knowledge Point:** Separation of structural metadata from the active tensor payload.

By subclassing `flax.struct.dataclass`, the base genome implements the PyTree protocol. Registration with `jax.tree_util.register_pytree_node` happens automatically. This separation allows metadata—for example, the permissible bounds of a real-valued genome—to remain static (using `pytree_node=False`) while the tensor payload is actively updated by evolutionary operators.

### 2.1.2. Representation Families

**Concept:** Tailoring evolutionary encodings to specific problem domains.

Different application areas require different genetic encodings. MalthusJAX provides specialized genome subclasses grouped by their structural families. 

**The Array Family:**
- **`RealGenome`**: Stores continuous variables. Uses `bounds` and wraps mutated values with `jax.numpy.clip`.
- **`BinaryGenome`**: Represents its data as boolean arrays.
- **`CategoricalGenome`**: Encodes each gene as an integer index within a specified set of categories.
- **`SeriesGenome`**: Encodes 2D time-series or sequence data, where operations can occur along specific time or feature axes.

**The Neuroevolution Family:**
- **`TensorNEATGenome`**: Represents neural network topologies and weights. Unlike simple array genomes, it encapsulates multiple distinct arrays (nodes, edges, activations) and relies on specialized **Emitters** rather than standard arithmetic operators to evolve structurally.

All of these types participate transparently in batching and JIT compilation.

### 2.1.3. The Population Container

**Concept:** Wrapping a batched instance of a Genome alongside engine-level state.

While Genomes describe a single unbatched individual (e.g., payload shape `(D,)`), the `BasePopulation[G]` container aggregates them into a coherent batch. Within a Population object, the `genes` attribute holds a **batched instance** of the Genome PyTree (e.g., payload shape `(N, D)`).

```python
# A single RealGenome
genome = RealGenome(values=jnp.zeros((10,))) 
# genome.values.shape == (10,)

# A Population containing 50 RealGenomes
pop = BasePopulation(
    genes=RealGenome(values=jnp.zeros((50, 10))),
    fitness=jnp.zeros((50,))
)
# pop.genes.values.shape == (50, 10)
```

This design guarantees that any function operating on a `Population` can safely map across the batch dimension (`axis=0`) without breaking the underlying genome structure. 

### 2.1.4. Specialized Populations (MO and QD)

Different evolutionary engines require different state tracking. Instead of polluting the pure `Genome` definition with algorithmic state, MalthusJAX uses specialized Population subclasses.

- **Multi-Objective (MO) Populations**: The `MOPopulation` (`mo/population.py`) adds parallel arrays for `pareto_rank` and `crowding_distance`. The engine updates these fields during non-dominated sorting.
- **Quality-Diversity (QD) Populations**: The `QDPopulation` (`qd/population.py`) adds arrays for behavioral `descriptors` and `cell_indices` to map individuals into the MAP-Elites archive.

By extending the population object, custom KPIs (e.g. diversity, age, novelty) can be seamlessly integrated. Consumers can access these values with the usual attribute lookup (`pop.pareto_rank`) and JIT compilation carries them through the engine efficiently.

## 2.2. Fitness Evaluators (malthusjax.core.fitness)

Fitness evaluators define the objective function. MalthusJAX separates the evaluation of a single genome from the batched evaluation of a population.

### 2.2.1. The Evaluator Contract

**Concept:** Defining a pure function `score = evaluator(genome)`.

Fitness evaluators must behave as pure functions: given the same genome and inputs, they always return the same scalar (or vector for MO) score. This property is essential for reproducibility and for enabling JIT compilation across populations. Evaluators avoid hidden state; any required data, such as training examples for a neural network, is supplied explicitly via additional static arguments.

### 2.2.2. Continuous Benchmarking (BBOBEvaluator)

**Concept:** Integrating standard Black-Box Optimization Benchmarks.

To facilitate fair comparisons with the optimization literature, MalthusJAX includes the `BBOBEvaluator`. It wraps well‑known mathematical test functions (Sphere, Rastrigin, Rosenbrock) that challenge optimization algorithms with multimodality and flat regions.

The implementations are written entirely with `jax.numpy`. A single call accepts a batch of genomes and returns a batch of scores using `jax.vmap`. This vectorization is critical for high throughput benchmarking.

## 2.3. Deterministic Stochasticity (malthusjax.core.random)

Randomness is the motor of evolutionary search, but JAX requires explicit state handling for Pseudo-Random Number Generators (PRNGs). 

Currently, MalthusJAX defaults to **Threefry** for maximum portability and bit-for-bit reproducibility across all devices. However, **Philox is theoretically superior for GPU workloads**. The random module is architected with future-proofing as a first-class design goal: every key creation, validation, and consumption path is abstracted behind the `PRNGImpl` enum.

### 2.3.1. JAX PRNG Key Philosophy

**Concept:** Escaping the hidden state of standard random libraries.

JAX treats random number generation as a functional transformation. A PRNG key is an explicit tensor passed to every function needing randomness. This enables:
1. **Reproducibility**: Given the same seed, the evolutionary trajectory is completely deterministic.
2. **Traceability**: Every stochastic operation is coupled to a specific key.
3. **JIT Compilation**: Keys are traced like any other array.

The lifecycle follows a strict pattern: initialize a master key, split it into subkeys for specific operations, use each subkey exactly once, and pass the updated master key forward. The `ResourceMapper` in the Engine tier automates much of this budgeting.

### 2.3.2. Framework Random Utilities

**Concept:** Ergonomic wrappers for common evolutionary random tasks.

The module `core/random.py` provides utilities for key creation and backend selection:

1. **`create_key(seed, impl)`**: Creates a JAX PRNG key with an explicit backend (e.g., `"threefry"`, `"philox"`).
2. **`resolve_prng_impl(name)`**: Converts user-facing PRNG names to the internal `PRNGImpl` enum.
3. **`validate_key(key, context)`**: Checks whether a key is legacy-style or new-style typed.

This disciplined approach to key handling keeps the core engine code simple and ensures that randomness is transparent and auditable.
