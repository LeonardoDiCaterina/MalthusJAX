# Extending Genetic Operators in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers who need to implement custom genetic operators (like specific graph crossovers or domain-aware mutation bounds) or Quality-Diversity Emitters that integrate perfectly into MalthusJAX's highly optimized, JIT-compiled pipeline.

Operators (Mutation, Crossover, Selection) and Emitters in MalthusJAX are purely functional PyTrees implemented as Flax Dataclasses. They are designed for maximum XLA optimization, relying heavily on `jax.vmap` and static **PRNG Key Budgeting**.

This guide covers the 3-Tier architecture, how to write and register custom operators, and how Emitters manage resources for Quality-Diversity algorithms.

---

## 1. The 3-Tier Architecture (Mutation & Crossover)

MalthusJAX separates random number generation (RNG) from pure arithmetic to maximize JIT efficiency. When creating a new standard operator, you only implement the lowest tiers, while the base class handles the complex vectorization.

### Tier 1: Pure Arithmetic (`_mutate_one` / `_crossover_one`)
- **Inputs**: A single genome and deterministic values (like pre-generated noise or a boolean crossover mask).
- **Outputs**: The modified genome.
- **Rule**: *Strictly no RNG calls here!* This ensures the arithmetic is perfectly pure for JAX tracing.

### Tier 2: Pure RNG (`_generate_noise` / `_generate_mask`)
- **Inputs**: A single PRNG key and the genome configuration.
- **Outputs**: The random variables (e.g., Gaussian noise) needed by Tier 1.

### Tier 3: The Fused Kernel (`__call__`)
- Implemented by the base class (`BaseMutation` / `BaseCrossover`).
- Fuses Tier 1 and Tier 2 inside a double `vmap` (iterating over both the population size and the number of offspring per parent).

---

## 2. Exposing Keys to the Resource Mapper

Unlike standard JAX code where you might call `jax.random.split` on the fly, MalthusJAX uses **Static Key Budgeting**. 

The `ResourceMapper` runs at engine initialization time. It asks your operator exactly how many keys it needs, and then pre-allocates a massive static array of PRNG keys. By doing this upfront, the XLA graph treats the PRNG dimensionality as a strict constant, entirely avoiding dynamic shape issues during compilation!

To expose your key requirements, you must implement the `num_keys_per_atomic_operation` property.

### Example: A Custom Gaussian Mutation
```python
from typing import Any
import chex
import jax
from flax import struct
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

@struct.dataclass
class CustomGaussianMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    mutation_rate: float = 0.1
    clip_bounds: bool = True

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """
        Expose key requirements to the Resource Mapper.
        We only need 1 key to generate the Gaussian noise array for a single offspring.
        """
        return 1

    def _generate_noise(self, key: chex.PRNGKey, g: RealGenome) -> chex.Array:
        """Tier 2: Generate noise using the exact key provided by Tier 3."""
        return jax.random.normal(key, g.shape) * self.mutation_rate

    def _mutate_one(self, g: RealGenome, noise: chex.Array, config: RealGenomeConfig) -> RealGenome:
        """Tier 1: Apply the noise purely."""
        new_values = g.values + noise
        mutated_genome = g.replace(values=new_values)
        
        if self.clip_bounds:
            return mutated_genome.autocorrect(config)
        return mutated_genome
```

---

## 3. Circumventing the Architecture (Notebook Hacks)

What if you just want to test a quick, custom mutation function and you "really don't feel like" building a full Tier-3 operator with key budgeting?

Because operators are registered via the `@register_operator` decorator, the Composer accepts **factory functions**. You can bypass the `BaseMutation` class entirely if you write a pure function that mimics the Tier 3 signature `(keys, population, config)`.

However, the cleanest way to circumvent the architecture for rapid prototyping is simply lifting a custom method directly onto your Genome class (as discussed in the Genome extension guide) and applying it in a Jupyter notebook with a standard `vmap`, completely bypassing the Engine and Resource Mapper.

> [!WARNING]
> While bypassing the 3-Tier architecture is useful for quick scripts, it is **highly discouraged** for production experiments. Bypassing the Resource Mapper means you lose the guarantees of strict key budgeting, which can lead to silent PRNG correlation bugs or catastrophic JIT compilation failures on TPUs.

---

## 4. Registering the Operator

To use your custom operator from a TOML configuration, register it in the `OperatorCatalog`.

```python
from malthusjax.composer.decorators import register_operator

@register_operator("custom_gaussian")
def build_custom_gaussian(**kwargs) -> CustomGaussianMutation:
    """Factory function intercepted by the Composer."""
    return CustomGaussianMutation(
        mutation_rate=kwargs.get("mutation_rate", 0.1),
        clip_bounds=kwargs.get("clip_bounds", True),
        num_offspring=kwargs.get("num_offspring", 1) # Inherited from BaseMutation
    )
```

**TOML Usage:**
```toml
[pipelines.my_pipeline.mutation]
type = "custom_gaussian"
mutation_rate = 0.05
```

---

## 5. Emitters (Quality-Diversity)

In Quality-Diversity (QD) algorithms (like MAP-Elites), standard linear selection and crossover don't apply. Instead, MalthusJAX relies on **Emitters**.

Emitters (`BaseEmitter` / `AtomicEmitter`) act as the ultimate endpoint for the Resource Mapper:
- **Exposing Keys**: Emitters declare their `batch_size` (offspring per generation) and `num_keys_per_atomic_operation`.
- **The `ask()` Orchestrator**: Instead of a standard Tier-3 `__call__`, Emitters use an `ask(state, repertoire, keys)` method. 
- **Resource Routing**: The Resource Mapper routes a massive flat buffer of keys into `ask()`. The Emitter slices off exactly what it needs for sampling parents from the archive (Tier 2), and beautifully reshapes the remaining keys into `(batch_size, atomic_keys)` to pass down into the `jax.vmap` for the pure atomic generation (Tier 1).

### Example Emitter Structure
```python
from malthusjax.operators.emitters.base import AtomicEmitter

class MyQDEmitter(AtomicEmitter):
    
    @property
    def batch_size(self) -> int:
        return 128
        
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
        
    def _sample_parents(self, state, repertoire, keys):
        # Tier 2: Use keys to sample from MAP-Elites grid
        return parents, metadata, new_state
        
    def _emit_one(self, state, key, *parents, **kwargs):
        # Tier 1: Mutate or crossover the sampled parents
        return mutated_offspring
```
