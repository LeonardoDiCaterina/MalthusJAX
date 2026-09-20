# Extending Genome Interpreters in MalthusJAX

> [!TIP]
> **Who is this for?** Researchers designing novel neural architectures (e.g. Transformers, KANs), custom Genetic Programming decoders, or other mappings from a raw genome PyTree to an executable policy/predictor.

In MalthusJAX's Composable Evaluator architecture, the **Interpreter** is the bridge between a Genome and an Environment. Its job is simple: take a raw genome and decode it into a mathematical function that can act upon the environment.

---

## 1. The Core Architecture: `BaseInterpreter`

All interpreters inherit from `BaseInterpreter[G]` (where `G` is the specific genome type), located in `src/malthusjax/core/fitness/composable/base.py`.

### The Key Properties

1. **Stateless (JIT-friendly)**: All configuration (input/output dimensions, hidden layer sizes) must be compiled in at construction time as `pytree_node=False` fields. The interpreter carries no JAX arrays itself.
2. **Genome-specific**: It expects a specific type of genome (e.g. `RealGenome`, `LinearGenome`).
3. **`apply(genome, inputs) -> outputs`**: The core method. It takes a single genome and a single input vector, and returns the prediction or action.
4. **`num_params`**: A property returning the exact number of scalar values this architecture requires from the genome. 

---

## 2. Writing a Custom Interpreter (Example: Simple Linear Decoder)

Let's build a simple linear layer without biases: $y = Wx$.

### Step 1: Subclass and Define Fields

```python
from typing import Any
import chex
from flax import struct
from malthusjax.core.fitness.composable.base import BaseInterpreter
from malthusjax.core.genome.real_genome import RealGenome

@struct.dataclass
class LinearInterpreter(BaseInterpreter[RealGenome]):
    """A simple linear decoder without biases."""
    
    # Must be marked pytree_node=False so they don't break JAX tracing!
    input_dim: int = struct.field(pytree_node=False)
    output_dim: int = struct.field(pytree_node=False)
```

### Step 2: Implement `num_params`

The Composer uses this property to ensure the user's `genome_length` in their TOML exactly matches the capacity of your architecture.

```python
    @property
    def num_params(self) -> int:
        """W matrix size."""
        return self.input_dim * self.output_dim
```

> [!NOTE]
> If the length of the genome is determined by the environment (e.g. a TSP tour length) rather than the interpreter architecture, return `-1`.

### Step 3: Implement `apply`

```python
    def apply(self, genome: RealGenome, inputs: chex.Array | None = None) -> chex.Array:
        # 1. Unpack the flat genome values into a weight matrix
        W = genome.values.reshape((self.input_dim, self.output_dim))
        
        # 2. Apply to inputs
        return inputs @ W
```

---

## 3. Registering the Interpreter

To use your interpreter in TOML configurations, you need to register it. While you can use the `@register_interpreter` decorator, the recommended approach for the Composable architecture is to let the adapter or the script instantiate it directly if it's complex, or register a simple string alias.

Currently, you instantiate it directly in your python code before passing it to the Evaluator:

```python
interp = LinearInterpreter(input_dim=4, output_dim=2)
```

*(Note: Composer TOML string-based initialization for custom interpreters is coming in a future release).*

---

## 4. When you don't need an Interpreter

If your task is an `OptimizationTask` (like BBOB, TSP, or Knapsack), the genome *is* the solution. There is no `inputs` to map to `outputs`.

In this case, use the built-in `IdentityInterpreter`. It simply ignores the inputs and returns `genome.values` directly to the environment.
