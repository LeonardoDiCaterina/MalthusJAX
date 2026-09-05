# Extending Genomes in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers and developers who want to apply genetic algorithms to custom data structures, such as variable-length graphs, discrete combinatorial structures, or hierarchical neural network policies.

In MalthusJAX, a **Genome** is a structured wrapper around a JAX `chex.Array`. Because MalthusJAX vectorizes evolution using JAX (`vmap`, `jit`, `scan`), your genome must be fully compatible with JAX PyTrees.

This document walks you through creating, registering, and configuring a custom genome.

---

## 1. Creating the Genome Data Structure

Genomes in MalthusJAX are built using **Flax Dataclasses** (`@struct.dataclass`). This ensures that JAX correctly recognizes your genome as a valid PyTree, allowing it to be batched and mapped across a population.

To create a new genome, you should subclass `BaseGenome` (found in `malthusjax.core.base`).

### Example: A Masked Real Genome

Suppose we want to evolve a neural network where every weight has a corresponding binary "mask" bit (1 if the connection is active, 0 if it is pruned). We can encode this in a single flat array and use our genome class to parse it.

Create a new file `src/malthusjax/core/genome/masked_genome.py`:

```python
from typing import cast, Any
import chex
import jax.numpy as jnp
from flax import struct
from malthusjax.core.base import BaseGenome

@struct.dataclass
class MaskedGenome(BaseGenome):
    """
    A custom genome that splits its flat values array into two halves:
    a binary mask and real-valued weights.
    """
    
    # 1. The flat array that standard operators (mutation/crossover) will manipulate.
    values: chex.Array
    
    # 2. Metadata: Let's store how long the mask is.
    # We use struct.field(pytree_node=False) to tell JAX this is static metadata, not a tensor.
    mask_length: int = struct.field(pytree_node=False)

    @property
    def mask(self) -> chex.Array:
        """Extract the mask portion."""
        return self.values[..., :self.mask_length] > 0.0

    @property
    def weights(self) -> chex.Array:
        """Extract the real-valued weights portion."""
        return self.values[..., self.mask_length:]

    # 3. Essential for Predictors (e.g. Jumanji / SKLearn pipelines)
    def unflatten_fn(self) -> tuple[chex.Array, chex.Array]:
        """
        Returns the structured parameters for fitness evaluators.
        Predictors often call this to decode the flat genome values.
        """
        return self.mask, self.weights

    # 4. Factory method used by the Engine Factory
    @classmethod
    def from_tensor(cls, arr: chex.Array, config: Any = None) -> "MaskedGenome":
        """
        Wraps a raw tensor back into a Genome object. 
        This is called internally by MalthusJAX after genetic operators are applied.
        """
        # Note: In a real implementation, config should be a MaskedGenomeConfig 
        # object containing the mask_length. For simplicity, we hardcode it here.
        return cls(values=arr, mask_length=arr.shape[-1] // 2)
```

> [!IMPORTANT]
> Always mark non-array properties (like integers, strings, or configs) with `struct.field(pytree_node=False)`. If you forget this, JAX will attempt to trace them as tensors and crash during `jax.jit`.

---

## 2. Understanding BaseGenome Methods

When inheriting from `BaseGenome`, there are several crucial methods and design patterns you must implement or understand to fully utilize MalthusJAX's architecture.

### Initialization: `random_init` and the `config` Argument
Every genome must implement the `@classmethod random_init`. This method is responsible for generating a **single** genome instance from scratch.
```python
@classmethod
def random_init(cls, key: chex.PRNGKey, config: Any) -> "MaskedGenome":
    """Initialize a single genome using the provided PRNG key and configuration."""
    # Draw from a distribution based on rules defined in your config
    values = jax.random.uniform(key, config.shape, minval=-1.0, maxval=1.0)
    return cls(values=values, mask_length=config.mask_length)
```
> [!NOTE]
> Why does `random_init` take a `config` object? 
> `config` holds the static definition of your search space (like `bounds`, `shape`, or `mask_length`). MalthusJAX separates this static setup from the dynamic `Genome` PyTree to prevent JAX from trying to differentiate or trace static metadata during JIT compilation.

### Buffer Donation and `clone_buffers`
MalthusJAX uses `jax.jit` with the `donate_argnums` flag in several critical execution loops (like the core generation step) to minimize memory allocations. When an argument is "donated", JAX overwrites its memory buffer in-place.
Because of this, `BaseGenome` provides a `clone_buffers()` (or alias `copy()`) method. 
If you ever need to save a specific genome (e.g., storing the best elite of a generation to an external archive) *before* it passes through a donated JIT block, you **must** call `.clone_buffers()` to physically copy the underlying `chex.Array` memory. Otherwise, your archived elite will be overwritten by the next generation's data!

### Pythonic Dunder Methods (Outside JIT)
While `BaseGenome` is primarily designed as a vectorized PyTree for JIT compilation, it implements several standard Python dunder methods (`__len__`, `__getitem__`, `__iter__`) to make it easy to inspect genomes interactively (e.g., in a Jupyter Notebook).

- **`__len__()`**: Returns the size of the population if batched, or the length of the `values` array if it's a single genome.
- **`__getitem__(key)`**: Allows you to extract a specific individual from a batched population PyTree seamlessly: `single_elite = pop.genes[0]`. JAX's `tree_map` is used under the hood to return a perfectly reconstructed `MaskedGenome` object!
- **`__iter__()`**: Lets you iterate over the individuals in a batched genome: `for ind in pop.genes: print(ind.values)`.

> [!WARNING]
> These dunder methods are strictly for **host-side inspection**. Do not use `__iter__` or `__getitem__` inside a `@jax.jit` compiled function, as JAX requires static loop unrolling which will fail or cause massive compilation times. Inside JIT, always use `jax.vmap`!

### "Lifting" Core Operational Methods (`mutate`, `add_noise`, `autocorrect`)
In the same spirit as the dunder methods, you can implement core operational methods directly on your genome class (e.g., a `mutate(self, key)` or `add_noise(self, key)` method). 
Because of the PyTree SoA design, a researcher working at the notebook level can easily perform custom Evolution Strategies (ES) by lifting these methods over a population using `jax.vmap`!

```python
# Notebook-level rapid prototyping
keys = jax.random.split(key, len(population))
batch_mutate = jax.vmap(MaskedGenome.add_noise, in_axes=(0, 0))
mutated_genes = batch_mutate(population.genes, keys)
```

This allows you to bypass the complex `OperatorCatalog` and `GeneticEngine` architectures if you just want to test a very quick custom strategy interactively. 
However, for production experiments, it is **highly encouraged** to go through the official `operators` route (writing and registering a standalone mutation operator). Doing so leverages the optimized, concern-separated experience of the Composer and ensures your code remains modular and scalable.

---

## 3. Creating the Configuration and Population Classes

To make your genome fully compatible with the Composer and standard initialization routines, you should provide a `Config` and a `Population` container.

```python
from malthusjax.core.base import BasePopulation
import jax

@struct.dataclass
class MaskedGenomeConfig:
    """Static configuration for MaskedGenome."""
    shape: tuple[int, ...] = struct.field(pytree_node=False)
    mask_length: int = struct.field(pytree_node=False)
    
    def init_population(self, key: chex.PRNGKey, size: int):
        """Protocol method called by Composer to generate the initial population."""
        # Generate random values for the population
        values = jax.random.normal(key, (size, *self.shape))
        
        genes = MaskedGenome(values=values, mask_length=self.mask_length)
        fitness = jnp.full((size,), -jnp.inf)
        
        return MaskedPopulation(genes=genes, fitness=fitness, config=self)


@struct.dataclass
class MaskedPopulation(BasePopulation[MaskedGenome]):
    """Parallel population container specialized for MaskedGenome."""
    genes: MaskedGenome
    fitness: chex.Array
    config: MaskedGenomeConfig = struct.field(pytree_node=False)
```

---

## 4. Registering the Genome

For MalthusJAX's `Composer` to recognize `"masked_genome"` in your TOML files, you must register it in the `GenomeCatalog`.

Open `src/malthusjax/composer/genome_catalog.py` and modify the `__init__` method of `GenomeCatalog`:

```python
# Inside src/malthusjax/composer/genome_catalog.py

class GenomeCatalog(Singleton):
    def __init__(self):
        super().__init__()
        
        # Register standard genomes...
        from malthusjax.core.genome.real_genome import RealGenomeConfig
        self.register("real", RealGenomeConfig)
        
        # Register your custom genome!
        from malthusjax.core.genome.masked_genome import MaskedGenomeConfig
        self.register("masked", MaskedGenomeConfig)
```

---

## 5. Using the Genome in a TOML File

Now that the genome is implemented and registered, you can immediately use it in any MalthusJAX pipeline!

```toml
[experiment]
name = "custom_genome_experiment"

[pipelines.masked_ga]
engine_type = "ga"
genome_type = "masked"
pop_size = 100
num_generations = 50

[pipelines.masked_ga.genome]
type = "masked"
shape = [128]        # Total size of the flat array
mask_length = 64     # Your custom kwarg! The Composer automatically passes this to MaskedGenomeConfig

[pipelines.masked_ga.fitness]
type = "sklearn"
dataset = "make_regression"
predictor = "my_custom_masked_predictor"
```

> [!WARNING]
> **Operator Compatibility**
> If you are using standard operators (like `gaussian_mutation`), they will blindly mutate the `values` array. If your custom genome requires specific mutation logic (e.g. flipping binary bits for the mask, but adding gaussian noise to the weights), you **must** write and register a custom mutation operator that understands your genome's structure.

---

## 6. Quick-Research Hacks

- **Check JAX Compatibility**: 
  Before running a full evolution, verify your genome is a valid PyTree:
  ```python
  import jax
  genome = MaskedGenome(values=jnp.ones(10), mask_length=5)
  leaves, treedef = jax.tree_util.tree_flatten(genome)
  print(leaves) # Should only show the values array!
  ```
- **Custom Mutation**: 
  If you write a custom operator, make sure the signature accepts and returns your specific genome type: `def mutate(genome: MaskedGenome, key: PRNGKey) -> MaskedGenome:`
