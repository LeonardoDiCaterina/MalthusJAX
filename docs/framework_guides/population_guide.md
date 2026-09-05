# Populations and Genomes in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers extending MalthusJAX with custom algorithms (like Quality Diversity, CMA-ES, or NEAT) or developers writing highly vectorized genetic operators.

Unlike traditional object-oriented genetic algorithms (like DEAP or early PyGAD), MalthusJAX does not maintain a Python list of individual genomes. Instead, it utilizes the **Struct-of-Arrays (SoA)** pattern. This architectural philosophy is the key to unlocking massive hardware acceleration on GPUs/TPUs via JAX.

This document covers how `Population` and `Genome` interact, how to vectorize genome operations, and when you actually need to extend the `BasePopulation`.

---

## 1. The Struct-of-Arrays (SoA) Philosophy

In a traditional Array-of-Structs (AoS) GA, you have `N` independent `Genome` objects. To mutate the population, you write a `for` loop.

In MalthusJAX, the `BasePopulation` is a single **PyTree dataclass** that holds a *single* `Genome` instance. However, every internal leaf array inside that `Genome` has an added leading dimension of size `N` (the population size).

### Why this matters:
1. **Zero-Overhead Vectorization**: By keeping all individuals in contiguous memory, `jax.vmap` can apply operators and fitness evaluators across the entire population in a single accelerated pass.
2. **PyTree Transparency**: Because a `Population` is just a dataclass containing a `Genome` dataclass, JAX natively understands how to flatten and unflatten the entire structure during JIT compilation.

---

## 2. "Lifting" Genome Methods with `vmap`

The true superpower of this architecture is that you write methods (like `autocorrect`, `distance`, or `add_noise`) for a **single** genome. Then, because the `Population` holds the batched PyTree, you "lift" these methods using `jax.vmap` to operate on the entire population simultaneously.

### Example A: Vectorizing `autocorrect`
If a mutation pushes genomes out of bounds, you can correct the entire population at once without `for` loops.

```python
import jax

# We map over the genes (axis 0) but leave the config unbatched (None)
batch_autocorrect = jax.vmap(MyGenome.autocorrect, in_axes=(0, None))

# This returns a newly corrected, batched genome!
corrected_genes = batch_autocorrect(population.genes, config)
population = population.replace(genes=corrected_genes)
```

### Example B: Vectorizing `distance`
If you need to calculate phenotypic or genotypic distances (e.g., for novelty search or speciation).

```python
# 1. Distance of the whole population to a SINGLE elite genome
# in_axes: (0, None, None) -> map over population, broadcast elite and metric string
batch_dist = jax.vmap(MyGenome.distance, in_axes=(0, None, None))
distances_to_elite = batch_dist(population.genes, elite_genome, "euclidean")

# 2. NxN Pairwise Distance Matrix (e.g., for diversity metrics)
# We double-vmap! First over axis 0 of 'self', then over axis 0 of 'other'
pairwise_dist = jax.vmap(jax.vmap(MyGenome.distance, in_axes=(None, 0, None)), in_axes=(0, None, None))
distance_matrix = pairwise_dist(population.genes, population.genes, "euclidean")
```

---

## 3. When (and Why) to Subclass `BasePopulation`

Because of the PyTree SoA pattern, `BasePopulation[G]` can technically hold **any** genome out of the box without requiring a custom subclass. If all your algorithm needs is fitness scores and batched genes, `BasePopulation` is entirely sufficient.

> [!IMPORTANT]
> You only need to extend or subclass `BasePopulation` when your algorithm requires **state or metadata** to be carried alongside the individuals.

### Scenarios requiring a custom Population subclass:

1. **Multi-Objective Evolution (MO)**
   Algorithms like NSGA-II need to track Pareto fronts and crowding distances, and they often implement self-sorting semantics.
   *(Real example from `src/malthusjax/core/genome/mo/population.py`)*
   ```python
   @struct.dataclass
   class MOPopulation(BasePopulation[Any]):
       # We guarantee these are populated for MOPopulation
       pareto_rank: chex.Array = struct.field(default=None)
       crowding_distance: chex.Array = struct.field(default=None)
       maximize: bool = struct.field(pytree_node=False, default=False)

       def select(self, key: chex.Array, batch_size: int) -> "MOPopulation":
           """Performs NSGA-II binary tournament selection based on ranks and crowding."""
           # Implementation uses jnp.where and tree_map to slice the population...
   ```

2. **Quality Diversity (QD)**
   If you are implementing MAP-Elites or Novelty Search, your population might extract descriptors from the `info` dict and compute QD-specific metrics.
   *(Real example from `src/malthusjax/core/genome/qd/population.py`)*
   ```python
   @struct.dataclass
   class QDPopulation(BasePopulation[G]):
       @property
       def descriptors(self) -> chex.Array:
           """Access the behavioral descriptors for the population."""
           return self.info["descriptors"]

       def get_qd_score(self, f_opt: chex.Numeric = 0.0) -> chex.Numeric:
           """Compute the sum of fitnesses of all valid solutions."""
           valid_mask = jnp.isfinite(self.fitness)
           return jnp.sum(jnp.where(valid_mask, self.fitness, 0.0))
   ```

3. **CMA-ES / Evolution Strategies**
   Advanced ES algorithms track global state beyond just candidate solutions.
   ```python
   @struct.dataclass
   class ESPopulation(BasePopulation[RealGenome]):
       covariance_matrix: chex.Array
       step_size: float
       evolution_path: chex.Array
   ```

### KPI Monitoring
You can also subclass `BasePopulation` to add custom property methods that compute real-time statistics (e.g., `population.diversity_score()`) directly on the PyTree without exposing the complex math to the Composer or Evaluator.

---

## 4. Advanced PyTree Manipulations (Slicing)

Because the population is a PyTree containing arrays with a leading `N` dimension, traditional Python slicing (e.g., `pop[:10]`) does not work. Instead, you use `jax.tree_map` to slice every leaf array simultaneously.

```python
def get_elites(population: BasePopulation, indices: chex.Array) -> BasePopulation:
    """Extract a sub-population based on an array of indices."""
    
    # Slice every array inside 'genes' at the given indices
    sliced_genes = jax.tree_util.tree_map(lambda x: x[indices], population.genes)
    
    # Slice fitness
    sliced_fitness = population.fitness[indices]
    
    return population.replace(genes=sliced_genes, fitness=sliced_fitness)
```

---

## 5. Pythonic Inspection (Outside JIT)

While MalthusJAX is heavily optimized for `vmap` and `jit`, it provides standard Python dunder methods (`__len__`, `__getitem__`, `__iter__`) natively on the `BaseGenome` class. 

Because of the PyTree SoA architecture, interacting with `population.genes` from a host-side script feels entirely object-oriented!

### Extracting a Single Individual
When you index a batched genome, `BaseGenome.__getitem__` automatically runs a `tree_map` under the hood. It perfectly reconstructs a single-genome PyTree of the exact correct type (e.g., `RealGenome` or `MaskedGenome`), stripped of the batch dimension!

```python
# Extract the first individual in the population
elite_genome = population.genes[0] 
print(type(elite_genome))  # <class 'MaskedGenome'>

# Pass it to an evaluator (which expects a single genome)
fitness = evaluator.evaluate(elite_genome, jax.random.PRNGKey(42))
```

### Iterating the Population
You can write Pythonic `for` loops over the population in Jupyter Notebooks or analysis scripts.

```python
# Iterate through all individuals in the batch
for i, ind in enumerate(population.genes):
    print(f"Individual {i} Magnitude: {ind.magnitude()}")
```

> [!WARNING]
> **Never use these inside `@jax.jit`!** 
> These dunder methods are intended strictly for host-side inspection and analysis. Attempting to use a Python `for` loop or dynamic integer indexing inside a compiled function forces JAX to unroll the loop, resulting in massive compile times or outright failures. Always use `jax.vmap` inside compiled blocks!
