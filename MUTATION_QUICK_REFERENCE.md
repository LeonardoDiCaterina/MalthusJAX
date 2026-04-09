# Mutation Quick Reference

**Fast lookup for common mutation patterns.**

---

## 5-Minute Getting Started

### Installation
```bash
pip install malthus-jax
```

### Use Gaussian Mutation in Evolution Loop
```python
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.core.genome import RealGenome, RealGenomeConfig
import jax.numpy as jnp
import jax

# Create operator
mutation = GaussianMutation(
    mutation_rate=0.1,     # 10% of genes mutated
    mutation_strength=0.1  # Gaussian std dev
)

# Setup
key = jax.random.PRNGKey(0)
config = RealGenomeConfig(shape=(20,), dtype=jnp.float32)
population = RealGenome(values=jax.random.normal(key, (50, 20)))

# Apply
keys = jax.random.split(key, 50)
mutated = mutation(keys, population, config)
```

### Use Binary Mutation
```python
from malthusjax.operators.mutation import BitFlipMutation
from malthusjax.core.genome import BinaryGenome, BinaryGenomeConfig

bit_flip = BitFlipMutation(mutation_rate=0.05)

population = BinaryGenome(values=jax.random.bernoulli(key, 0.5, (50, 100)).astype(jnp.uint8))
config = BinaryGenomeConfig(shape=(100,), dtype=jnp.uint8)

mutated = bit_flip(keys, population, config)
```

---

## Operator Quick Chart

| Operator | Best For | Parameters | Keys | Mode |
|----------|----------|------------|------|------|
| **GaussianMutation** | General continuous | rate, strength | 2 | Std + Inj |
| **BallMutation** | High-dim (d>100) | radius, rate | 3 | Std + Inj |
| **PolynomialMutation** | MOO, fine-tuning | rate, eta | 2 | Std + Inj |
| **BitFlipMutation** | Binary problems | rate | 1 | Std only |
| **ScrambleMutation** | Permutation | rate | 2 | Std only |
| **SwapMutation** | Scheduling | rate | 3 | Std only |
| **EvosaxGaussianWrapper** | Benchmarking | strength | 1 | Inj |

---

## API Quick Reference

### GaussianMutation

```python
mutation = GaussianMutation(
    mutation_rate=0.1,           # float in [0,1]
    mutation_strength=0.1,       # Gaussian std dev
    clip=False,                  # Clip to bounds?
    schedule_type=None,          # ScheduleType enum
    final_strength=0.0           # Decay target
)

# Call signature
mutated = mutation(keys, population, config, generation=0)
```

### BallMutation

```python
mutation = BallMutation(
    radius=0.1,                  # Ball radius
    mutation_rate=1.0,           # Usually 1.0
    clip=False,
    schedule_type=None,
    final_radius=0.0
)

# Same call signature as GaussianMutation
```

### PolynomialMutation

```python
mutation = PolynomialMutation(
    mutation_rate=0.1,           # Per-gene probability
    eta=20.0,                    # Shape parameter
    clip=False
)

# Same call API
```

### BitFlipMutation

```python
mutation = BitFlipMutation(
    mutation_rate=0.1            # Bit flip probability
)

# Same call API
```

### Injection Mode Variant

```python
# Available for: Gaussian, Ball, Polynomial
from malthusjax.operators.mutation import GaussianMutation_injection

mutation_inj = GaussianMutation_injection(
    mutation_rate=0.1,
    mutation_strength=0.1
)

# Single-key interface (differs from standard)
mutated = mutation_inj(single_key, population, config, generation=0)
```

---

## Common Tasks

### Task: Set up GA with Gaussian Mutation

```python
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.crossover import UniformCrossover  # or your choice
from malthusjax.core.selection import TournamentSelection
from malthusjax.core.evolution import EvolutionLoop

ga = EvolutionLoop(
    population_size=50,
    selection_fn=TournamentSelection(tournament_size=3),
    crossover_fn=UniformCrossover(crossover_rate=0.7),
    mutation_fn=GaussianMutation(mutation_rate=0.1, mutation_strength=0.1),
    elite_size=2
)

# Run evolution
for gen in range(100):
    pop = ga.step(gen)
    print(f"Gen {gen}: best = {pop.best_fitness}")
```

### Task: Decay Mutation Strength Over Time

```python
from malthusjax.engine.schedules import ScheduleType

mutation = GaussianMutation(
    mutation_rate=0.1,
    mutation_strength=0.2,                    # Start high
    schedule_type=ScheduleType.LINEAR_DECAY,
    final_strength=0.01,                      # End low
    max_generations=1000
)

# In loop: automatically decays
for gen in range(1000):
    mutated = mutation(keys, pop, config, generation=gen)
```

### Task: Compare MalthusJAX vs Evosax Mutation

```python
from malthusjax.operators.mutation import GaussianMutation, EvosaxGaussianWrapper

malthus = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)
evosax_wrapper = EvosaxGaussianWrapper(mutation_strength=0.1)

# Should produce similar (not identical) results
# Identical order-of-operations differs
```

### Task: High-Dimensional Optimization (d > 100)

```python
from malthusjax.operators.mutation import BallMutation

# Use Ball instead of Gaussian
mutation = BallMutation(
    radius=0.1,        # 10% of search space width
    mutation_rate=1.0  # Apply to all dimensions
)
```

### Task: Fine-Tuning Near Optimum

```python
# Use PolynomialMutation with high eta
mutation = PolynomialMutation(
    mutation_rate=0.05,   # Lower rate
    eta=50.0              # Heavy eta = smaller perturbations
)
```

---

## Mode Selection Decision Table

| Scenario | Recommended Mode |
|----------|------------------|
| Multi-operator pipeline (mutation + crossover + selection) | Standard |
| Single-operator, simple loop | Either |
| Large population (>5,000) | Injection (memory efficient) |
| GPU/TPU with limited memory | Injection |
| XLA compilation critical | Standard (pre-allocated shapes) |
| Reproducibility paramount | Standard (per-individual keys) |
| One-shot evaluation | Injection (simpler) |

---

## Parameter Tuning Quick Guide

### Gaussian Mutation

```python
# Exploration phase (early generations)
mutation = GaussianMutation(mutation_rate=0.2, mutation_strength=0.15)

# Balanced phase (mid generations)
mutation = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)

# Exploitation phase (late generations)
mutation = GaussianMutation(mutation_rate=0.05, mutation_strength=0.05)

# Auto-decay with scheduling
mutation = GaussianMutation(
    mutation_rate=0.1,
    mutation_strength=0.1,
    schedule_type=ScheduleType.LINEAR_DECAY,
    final_strength=0.01,
    max_generations=1000
)
```

### Mutation Rate by Dimension

```python
def recommended_mutation_rate(dimension):
    """Standard recommendation: 1/d"""
    return 1.0 / dimension

# Examples:
d=10 → rate ≈ 0.1
d=50 → rate ≈ 0.02
d=100 → rate ≈ 0.01
d=500 → rate ≈ 0.002
```

### Ball Radius by Problem Scale

```python
# If search range is [-L, L]:
# Use radius ≈ 0.05 * (2*L) to 0.1 * (2*L)

search_range = (-5.0, 5.0)
width = 5.0 - (-5.0)  # 10.0
recommended_radius = 0.05 * width  # 0.5 to 1.0

ball = BallMutation(radius=0.5)
```

---

## Troubleshooting Matrix

| Problem | Symptom | Quick Fix |
|---------|---------|-----------|
| "Shape mismatch" | `ValueError: broadcast shapes` | Check `config.shape == genome.values.shape[1:]` |
| "No mutation effect" | Offspring identical to parents | Increase `mutation_rate` or `strength` |
| "Type promotion" | Unexpected `float64` output | Use `jnp.array(val, dtype=dtype)` explicitly |
| "Inconsistent results" | Different seed → different results | Always pass `generation=gen` to mutation call |
| "Slow compilation" | First run 5-10 seconds | Normal (XLA traces); subsequent <100ms |
| "High memory" | OOM with large populations | Use injection mode: `GaussianMutation_injection` |

---

## File Locations

| What | File |
|------|------|
| Gaussian mutation | `src/malthusjax/operators/mutation/real.py` |
| Ball mutation | `src/malthusjax/operators/mutation/real.py` |
| Polynomial mutation | `src/malthusjax/operators/mutation/real.py` |
| Binary mutations | `src/malthusjax/operators/mutation/binary.py` |
| Evosax wrapper | `src/malthusjax/operators/mutation/evosax_mutation.py` |
| Tests | `tests/operators/mutation/` |
| Full guide | `MUTATION_INTEGRATION.md` |
| README | `src/malthusjax/operators/mutation/README.md` |

---

## Advanced: Custom Mutation Template

```python
from flax import struct
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome import RealGenome, RealGenomeConfig, RealPopulation
import jax
import jax.numpy as jnp

@struct.dataclass
class MyMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    param1: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2  # Adjust as needed
    
    def _generate_noise(self, keys, config, generation=0):
        # Tier 2: Generate noise/masks
        k1, k2 = keys[0], keys[1]
        dtype = config.dtype
        
        mask = jax.random.bernoulli(k1, p=0.1, shape=config.shape).astype(dtype)
        noise = jax.random.normal(k2, shape=config.shape, dtype=dtype)
        
        return noise * mask
    
    def _mutate_one(self, genome, noise_data, config, **kwargs):
        # Tier 1: Apply mutation
        mutated = genome.values + noise_data
        return genome.replace(values=mutated)
```

---

## Resources

- **Full Integration Guide**: `MUTATION_INTEGRATION.md`
- **Mutation README**: `src/malthusjax/operators/mutation/README.md`
- **Crossover Quick Ref**: `CROSSOVER_QUICK_REFERENCE.md`
- **Base Classes**: `src/malthusjax/operators/base.py`
- **Examples**: `examples/`
