# Guide 5: Quality-Diversity & MAP-Elites in Cartesian GP

Quality-Diversity (QD) algorithms (Pugh et al., 2016) illuminate the search space by discovering a diverse archive of high-performing solutions rather than a single global optimum.

`malthus-gp` provides native components to integrate Cartesian Genetic Programming with **MAP-Elites** (Multi-dimensional Archive of Phenotypic Elites).

---

## 1. Why Quality-Diversity for Symbolic Discovery?

Standard optimization converges on a single formula. However, in scientific modeling and engineering:
- Analysts frequently prefer **interpretable, compact formulas** over marginally more accurate, overly complex models.
- Different operating regimes may require different mathematical structures.
- By mapping programs along behavioral dimensions like **formula complexity** and **effective graph depth**, researchers can explore the full Pareto-like trade-off front in an interactive archive.

---

## 2. Core QD Components in `malthus-gp`

### 2.1 Behavioral Descriptors (`CartesianQDEvaluator`)

`CartesianQDEvaluator` evaluates candidate programs and computes behavioral descriptor coordinates:

1. **Active Node Ratio**: The fraction of nodes actively connected to the output:
   $$d_0 = \frac{\sum \text{active\_mask}}{n_{\text{nodes}}}$$
2. **Effective Graph Depth**: The maximum directed path length from primary inputs to the program output:
   $$d_1 = \max_{\text{paths}} (\text{length})$$

### 2.2 The QD Emitter (`CartesianEmitter`)

`CartesianEmitter` handles population perturbation and variation within the MAP-Elites loop, sampling parent programs uniformly from occupied archive cells and applying `CartesianMicroMutation`.

---

## 3. Configuration & Usage Pattern

```python
import jax.numpy as jnp
from malthus_gp.evaluators.cartesian_qd_evaluator import (
    CartesianQDEvaluator,
    CartesianQDEvaluatorConfig,
)
from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.cartesian_emitter import CartesianEmitter

# 1. Configure Cartesian Grid
genome_cfg = CartesianGenomeConfig(
    num_rows=1,
    num_cols=30,
    num_inputs=2,
    num_outputs=1,
)

# 2. Define Quality-Diversity Evaluator
qd_evaluator = CartesianQDEvaluator(
    config=CartesianQDEvaluatorConfig(
        min_descriptors=jnp.array([0.0, 1.0]),   # [min_active_ratio, min_depth]
        max_descriptors=jnp.array([1.0, 30.0]),  # [max_active_ratio, max_depth]
    ),
    data=(X_train, y_train),
)

# 3. Configure MAP-Elites Emitter
emitter = CartesianEmitter(
    mutation_rate=0.05,
    batch_size=32,
)
```

The resulting MAP-Elites repertoire contains diverse symbolic formulas spanning the spectrum from trivial 1-node approximations to high-order 30-node expressions, allowing post-hoc selection of the optimal simplicity-accuracy trade-off.
