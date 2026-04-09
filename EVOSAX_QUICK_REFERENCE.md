# Evosax Integration Quick Reference

**Fast lookup for common Evosax integration patterns.**

---

## 5-Minute Getting Started

### Installation
```bash
pip install malthus-jax evosax>=0.1.6
```

### Use Evosax Crossover in MalthusJAX
```python
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
import evosax

crossover = EvosaxUniformCrossoverWrapper(
    evosax_crossover_fn=evosax.crossover,
    crossover_rate=0.7
)
offspring = crossover(keys, parents1, parents2, config)
```

### Use Evosax Mutation in MalthusJAX
```python
from malthusjax.operators.mutation import EvosaxMutationWrapper
import evosax

mutation = EvosaxMutationWrapper(
    evosax_mutation_fn=evosax.mutation,
    std_dev=0.1
)
mutated = mutation(keys, genomes, config)
```

### Evaluate with BBOB Functions
```python
from evosax.problems import BBOBFitness

fitness = BBOBFitness(num_dims=10, function_id=1)
R, Q = fitness.get_rotation_matrices(key)
scores = fitness.rollout(key, solutions, R, Q)
```

---

## Common Tasks

### Task: Run GA with Evosax Crossover+Mutation

```python
import jax
from malthusjax.composers.ga_composer import GAComposer
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxMutationWrapper
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator
import evosax

# Setup
fitness = BBOBEvaluator(num_dims=10, function_id=1)
crossover = EvosaxUniformCrossoverWrapper(evosax.crossover, rate=0.7)
mutation = EvosaxMutationWrapper(evosax.mutation, std_dev=0.1)

# Compose
ga = GAComposer(
    population_size=50,
    crossover_fn=crossover,
    mutation_fn=mutation,
    fitness_fn=fitness,
    elite_size=2
)

# Evolve
key = jax.random.PRNGKey(0)
pop = ga.initialize(key, num_generations=100)
```

### Task: Compare MalthusJAX vs Direct Evosax

```python
# MalthusJAX approach
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
wrapped = EvosaxUniformCrossoverWrapper(evosax.crossover, rate=0.5)
offspring_malthus = wrapped(keys, pop1, pop2, config)

# Direct Evosax approach
offspring_evosax = evosax.crossover(keys[0], pop1.values[0], pop2.values[0], 0.5)

# Should produce numerically identical results
assert jnp.allclose(offspring_malthus.values[0], offspring_evosax)
```

### Task: Add Evosax Pipeline to Custom Composer

```python
from malthusjax.core.evolution import EvolutionLoop
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxMutationWrapper
from malthusjax.core.selection import TournamentSelection

loop = EvolutionLoop(
    population_size=50,
    selection_fn=TournamentSelection(tournament_size=3),
    crossover_fn=EvosaxUniformCrossoverWrapper(
        evosax_crossover_fn=evosax.crossover,
        crossover_rate=0.7
    ),
    mutation_fn=EvosaxMutationWrapper(
        evosax_mutation_fn=evosax.mutation,
        std_dev=0.1
    )
)
# Use as normal
```

---

## API Quick Reference

### EvosaxUniformCrossoverWrapper

```python
wrapper = EvosaxUniformCrossoverWrapper(
    evosax_crossover_fn=evosax.crossover,    # Required
    crossover_rate=0.5,                      # float in [0,1]
    injection_mode=False,                    # bool
    dtype=jnp.float32                        # JAX dtype
)

# Call signature
offspring = wrapper(
    keys=Array,           # shape (pop_size * num_offspring, 2)
    p1_pop=RealGenome,    # Parent1 population
    p2_pop=RealGenome,    # Parent2 population
    config=GenomeConfig   # Shape/dtype info
)
```

### EvosaxMutationWrapper

```python
wrapper = EvosaxMutationWrapper(
    evosax_mutation_fn=evosax.mutation,  # Required
    std_dev=0.1,                         # float > 0
    dtype=jnp.float32                    # JAX dtype
)

# Call signature
mutated = wrapper(
    keys=Array,           # shape (pop_size, 2)
    genomes=RealGenome,   # Population to mutate
    config=GenomeConfig   # Shape/dtype info
)
```

### BBOBFitness (Evosax)

```python
fitness = BBOBFitness(
    num_dims=10,      # Problem dimension
    function_id=1     # BBOB function (1-55)
)

# Get matrices (once per session)
R, Q = fitness.get_rotation_matrices(key)

# Evaluate (reuse R, Q)
scores = fitness.rollout(
    key=key,
    X=solutions,     # shape (n_solutions, num_dims)
    R=R,
    Q=Q
)  # shape (n_solutions,)
```

### BBOBEvaluator (MalthusJAX)

```python
evaluator = BBOBEvaluator(
    num_dims=10,
    function_id=1,        # or function_name="sphere"
    search_range=(-5, 5),
    instance=0            # BBOB instance
)

# Call on genomes
scores = evaluator(genomes)  # RealGenome -> Array
```

---

## Mode Selection: Standard vs Injection

**Choose Standard (`injection_mode=False`)**:
- Default, traditional crossover
- Every gene independently selected from P1 or P2
- ✓ Use when: swapping operators, benchmarking, standard GA

**Choose Injection (`injection_mode=True`)**:
- Alternative semantics: "cross or don't"
- Full parent 1 returned if not crossing
- ✓ Use when: adaptive operators (SBX, BlendCrossover), exploration control

```python
# Standard
wrapper_std = EvosaxUniformCrossoverWrapper(
    evosax.crossover,
    crossover_rate=0.7,
    injection_mode=False  # <- per-element selection
)

# Injection (gate-based)
wrapper_inj = EvosaxUniformCrossoverWrapper(
    evosax.crossover,
    crossover_rate=0.7,
    injection_mode=True   # <- all-or-nothing crossing
)
```

---

## Troubleshooting Matrix

| Problem | Symptom | Fix |
|---------|---------|-----|
| BBOBFitness not found | `ImportError` | `pip install --upgrade evosax` |
| Wrong evosax version | Ask/tell methods missing | Install GitHub: `pip install git+https://github.com/rjbruin/evosax.git@main` |
| Shape mismatch | `ValueError: could not broadcast` | Check `genome_config.shape` matches evosax expectation |
| Slow compilation | First call takes 10+ seconds | Expected (JAX compilation); subsequent calls fast |
| Key leaks / shape errors | JAX tracer issues | Ensure `num_keys_per_atomic_operation` honored |
| Inconsistent results | Random seed not set | Always use explicit `jax.random.PRNGKey(seed)` |

---

## File Locations

| What | File |
|------|------|
| Crossover wrapper | `src/malthusjax/operators/crossover/evosax_crossover.py` |
| Mutation wrapper | `src/malthusjax/operators/mutation/evosax_mutation.py` |
| Compatibility layer | `src/malthusjax/compat/evosax_mimic.py` |
| BBOB evaluator | `src/malthusjax/core/fitness/bbob_evaluator.py` |
| Evosax adapter | `src/malthusjax/composer/evosax_adapter.py` |
| Crossover tests | `tests/operators/crossover/test_evosax_crossover_parity.py` |
| Mutation tests | `tests/operators/mutation/test_evosax_mutation_parity.py` |
| Full guide | `EVOSAX_INTEGRATION.md` (this repo) |

---

## Advanced: Custom Evosax Function

```python
# Define compatible function
def my_crossover(key, p1, p2, rate):
    """Signature: (key, Array, Array, float) -> Array"""
    mask = jax.random.bernoulli(key, rate, p1.shape)
    return jnp.where(mask, p2, p1)

# Wrap it
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
wrapper = EvosaxUniformCrossoverWrapper(my_crossover, rate=0.5)

# Use normally
```

---

## Resources

- **Evosax docs**: https://evosax.readthedocs.io
- **BBOB functions**: https://coco.gforge.inria.fr
- **Full integration guide**: `EVOSAX_INTEGRATION.md`
- **Crossover architecture**: `src/malthusjax/operators/crossover/README.md`
