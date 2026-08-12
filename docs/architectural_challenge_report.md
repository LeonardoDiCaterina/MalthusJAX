# The XLA PyTree Overhead Challenge

## 1. The Performance Gap
During benchmarking on lightweight, low-dimensional problems (e.g., `num_dims=9`, `pop_size=195`, `generations=387`), we identified a significant execution time gap between **MalthusJAX** and **EvoSAX**.

### The Benchmarked Configurations

1. **EvoSAX SimpleGA (Baseline) [~29 ms]**
   - **Context**: The target performance ceiling. EvoSAX passes a flat dataclass of raw JAX arrays (`fitness`, `population`) directly through its `lax.scan` loop. There are no nested PyTrees, dynamic slice updates, or complex structure reconstructions inside the execution kernel, allowing XLA to fuse the loop aggressively.

2. **MalthusJAX SimpleGAEngine (Ablation) [~200 ms]**
   - **Context**: We built this engine specifically to test if the modular operators (`BaseMutation`, `BaseCrossover`) were causing the overhead. It completely hardcodes EvoSAX's `ask` and `tell` logic (direct `jax.random.choice` and manual masking arrays) but keeps MalthusJAX's `BasePopulation` tracking. Even without any base operators, it is ~7x slower, proving the PyTree state structure inside `lax.scan` is the root bottleneck.

3. **MalthusJAX LightenedGeneticEngine (Fast-Path Attempt) [~250 ms]**
   - **Context**: This was an attempt to optimize the standard pipeline by bypassing standard memory copies (`_merge` phase with `elitism=0`) and using a `use_vectorized_operators=True` flag to hardcode vectorized noise. It significantly improved performance over the standard engine (down from 400ms) but hit a hard floor at 250ms due to the constant `pop.replace(genes=...)` unpack/pack cycles inside the loop.

4. **MalthusJAX GeneticEngine (Standard) [~400+ ms]**
   - **Context**: The fully featured, highly modular MalthusJAX engine. It passes `BasePopulation` PyTrees to Tier 1 operators, manages dynamic `ResourceMap` entropy keys, traces histories, and handles elitism merging. Perfect for complex representations (like NEAT graphs or mixed discrete/continuous spaces) where the evaluation time swamps the loop overhead, but too slow for small dimensional BBOB sweeps.

### The Root Cause: JAX/XLA PyTree Instantiation
Extensive profiling ruled out JAX JIT recompilation, history tracking, and elite merging as the primary bottlenecks. The overhead is structural:
- MalthusJAX's `lax.scan` carries a deeply nested PyTree: `AbstractEvolutionState` -> `BasePopulation` -> `RealGenome` -> `values`.
- Inside the loop, the engine continuously unpacks these arrays to do math, and then **re-instantiates the dataclasses** (e.g., `population.replace(genes=...)`) to pass them between phases (Reproduction -> Evaluation -> HOF).
- **XLA HLO Impact**: XLA struggles to fuse these continuous tuple creations and destructions. The resulting HLO IR for MalthusJAX is **2400 lines** (with over 10 `dynamic-update-slice` calls), compared to EvoSAX's highly fused **1800 lines** (0 dynamic slices). On small problem sizes, this un-optimized graph traversal swamps the actual mathematical compute time, adding a ~200ms fixed overhead.

---

## 2. The 3-Tier Operator Architecture
MalthusJAX currently structures its mutation and crossover operators using a modular 3-tier paradigm. For example, in `GaussianMutation`:

### Tier 1: `__call__` (The PyTree Interface)
The standard user-facing method. It accepts a full `BasePopulation` PyTree, allocates entropy, maps over the population, and returns a new `BasePopulation`.
```python
def __call__(self, keys, population, config, generation):
    noise = self._generate_noise(keys, config, generation)
    new_genomes = jax.tree_map(lambda g: self._mutate_one(g, noise, config), population.genes)
    return population.replace(genes=new_genomes)
```
* **Limitation**: Tightly couples the mathematical operation to the `BasePopulation` dataclass structure.

### Tier 2: `_generate_noise` (The Vectorized Math)
Generates the raw numerical perturbations (e.g., masked Gaussian noise) for the entire population at once, using raw JAX arrays.
```python
def _generate_noise(self, keys, config, generation) -> jax.Array:
    mask = jax.random.bernoulli(keys[0], self.mutation_rate)
    noise = jax.random.normal(keys[1]) * self.mutation_strength
    return noise * mask
```
* **Strength**: Highly optimized, pure array operations.

### Tier 3: `_mutate_one` (The Arithmetic Kernel)
Applies the generated noise to a single genome. Currently, it expects a `RealGenome` PyTree.
```python
def _mutate_one(self, genome: RealGenome, noise: jax.Array, config) -> RealGenome:
    mutated_values = genome.values + noise
    return replace(genome, values=mutated_values)
```
* **Limitation**: Forces PyTree unpacking and repacking at the innermost scalar level.

---

## 3. The Path Forward: "Operator Bypass"
To hit the **29ms EvoSAX parity** without losing the modular operators, we can exploit the 3-Tier architecture via a new `VectorizedEngine`.

Instead of flattening the operators (which would break MalthusJAX's general PyTree support), we will **bypass Tier 1**.

### Step A: Refactor Tier 3 to Pure Arrays
We will modify the Tier 3 arithmetic kernels to operate strictly on raw arrays (e.g., `_mutate_values` instead of `_mutate_one`).
```python
def _mutate_values(self, values: jax.Array, noise: jax.Array, config) -> jax.Array:
    return values + noise
```

### Step B: The Vectorized Engine
We will introduce `VectorizedEngine` (in a new file for clean ablation), which strips the `BasePopulation` at initialization and maintains a purely flat array state in its `lax.scan` loop:
```python
@struct.dataclass
class VectorizedEvolutionState:
    genes: jax.Array       # (pop_size, num_dims)
    fitness: jax.Array     # (pop_size,)
    rng_key: chex.Array
```

### Step C: Direct Tier 2/3 Invocation
Inside `VectorizedEngine.step`, we completely bypass `ops.mutation.__call__` and hit the array-native methods directly:
```python
# Bypass Tier 1: Generate noise directly (Tier 2)
noise = ops.mutation._generate_noise(k_mut, config, state.generation)

# Apply noise using pure array vmap (Tier 3)
mutated_genes = jax.vmap(ops.mutation._mutate_values)(state.genes, noise, config)
```

### Conclusion
By decoupling the mathematical operations (Tiers 2/3) from the PyTree packaging (Tier 1), MalthusJAX can serve two distinct execution paradigms:
1. **The Standard Pipeline (`GeneticEngine`)**: Uses Tier 1 for maximum flexibility and nested structured genomes (e.g., Neural Networks).
2. **The Vectorized Pipeline (`VectorizedEngine`)**: Uses Tiers 2/3 for pure numerical speed, bypassing PyTree instantiation inside the XLA loop to achieve EvoSAX's execution speeds on small continuous landscapes.

---

## 4. Benchmark Execution & Analysis Guide
To validate these findings and test new pipeline architectures, you can use the unified `Makefile` commands which interface directly with our TOML-driven experiment runner.

### Running Benchmarks
To execute a benchmark suite defined in a TOML configuration:
```bash
# Standard blocking execution
make benchmark-run TOML=configs/h1_parity_lhs.toml

# Smoke test (runs a highly truncated version for fast verification)
make benchmark-run-smoke TOML=configs/h1_parity_lhs.toml

# Background execution (runs via nohup, useful for long sweeps on clusters)
make benchmark-run-nohup TOML=configs/h1_parity_lhs.toml
```

### Analyzing Benchmark Data
Once a benchmark finishes (or a smoke test completes), the raw results are saved to the `results/` directory. To process this data into summary tables, comparison metrics, and plots:
```bash
make benchmark-analyze TOML=configs/h1_parity_lhs.toml
```
*Note: The analyzer reads the output directory path directly from the TOML file.*

### Full Unified Sweeps
For convenience, you can run and analyze all major hypothesis benchmarks in one go:
```bash
# Run all smoke tests and immediately analyze them
make smoke-all

# Run all full-scale benchmarks and generate thesis tables
make show-tell
```
