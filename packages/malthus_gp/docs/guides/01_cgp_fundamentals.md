# Guide 1: Cartesian GP Fundamentals & Neutral Search

This guide covers the core concepts, mathematical principles, and practical usage of classical Cartesian Genetic Programming (CGP) in `malthus-gp`.

---

## 1. Cartesian Representation vs. Tree-Based GP

Classical Genetic Programming (Koza, 1992) represents computer programs as LISP-like parse trees. While intuitive, tree GP has well-documented failure modes:
1. **Code Bloat**: Trees grow uncontrollably in size without improving fitness, requiring complex parsimony pressure heuristics.
2. **Dynamic Shapes & Branching**: Variable tree depths prevent efficient hardware acceleration in JAX / XLA.
3. **No Reuse of Sub-expressions**: Trees cannot express Directed Acyclic Graph (DAG) structures where a computed node is reused by multiple downstream operations.

### The Cartesian Solution

Cartesian Genetic Programming (Miller & Thomson, 2000) encodes programs as a 2-D computational grid:
- **Fixed Chromosome Size**: The genome has a fixed number of rows ($n_r$) and columns ($n_c$), eliminating bloat.
- **Intrinsic Subgraph Reuse**: A node's output can be referenced by multiple downstream nodes, enabling natural DAG topologies.
- **Active vs. Non-Coding Genes**: Only nodes that have a directed path to the program outputs are evaluated (the *phenotype*). The remaining nodes represent non-coding *junk DNA*.

```text
       ┌───────────┐         ┌───────────┐
x0 ───►│ Node 2 (+)├────────►│ Node 4 (*)├──► Output 0
       └─────┬─────┘         └─────┬─────┘
             │                     │
x1 ──────────┼─────────────────────┘
             ▼
       ┌───────────┐
       │ Node 3 (-)│  (Non-coding / inactive node)
       └───────────┘
```

---

## 2. The $(1 + \lambda)$ Evolutionary Strategy & Neutral Drift

Cartesian GP typically uses a $(1 + \lambda)$-ES (Evolutionary Strategy) with $\lambda = 4$ offspring:

1. **Initialization**: A single parent program is initialized randomly.
2. **Offspring Generation**: The parent is cloned to produce $\lambda$ offspring, each subjected to point micro-mutation.
3. **Evaluation**: All $1 + \lambda$ programs are evaluated on the dataset.
4. **Neutral Replacement**: The best offspring replaces the parent if its fitness is **equal to or better than** the parent:
   $$\text{fitness}(\text{offspring}) \le \text{fitness}(\text{parent}) \implies \text{parent} \leftarrow \text{offspring}$$

### Why Neutrality Matters

In classical algorithms, mutations that do not improve fitness are rejected. In CGP:
- Mutations frequently hit **non-coding nodes**, resulting in identical fitness.
- By allowing the offspring to replace the parent on fitness ties, the search drifts randomly across a **neutral network** of phenotypically equivalent genotypes.
- When an inactive sub-network is eventually connected to an active output by a single point mutation, large structural innovations appear instantaneously, effectively bypassing local minima.

---

## 3. Practical Example: Solving Koza-1

The Koza-1 benchmark is the quartic polynomial:
$$y = x^4 + x^3 + x^2 + x, \quad x \in [-1, 1]$$

### Code Example

```python
import numpy as np
from malthus_gp.learn import CGPRegressor

# 1. Generate training data
X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
y = (X**4 + X**3 + X**2 + X).flatten()

# 2. Configure Cartesian GP Regressor
model = CGPRegressor(
    n_rows=1,              # 1D linear DAG (equivalent to Linear GP with neutrality)
    n_cols=30,             # 30 computational nodes
    levels_back=-1,        # Full feed-forward connectivity
    pop_size=5,            # (1 + 4) Evolutionary Strategy
    max_generations=500,   # Maximum iterations
    mutation_rate=0.04,    # Target ~3 mutated genes per chromosome
    random_state=42,
)

# 3. Fit model
model.fit(X, y)

# 4. Inspect results
print(f"R² Score:        {model.score(X, y):.5f}")
print(f"Evolved Formula: {model.to_formula()}")
```

### Inspecting Active Phenotypes

You can inspect which nodes in the grid were actively expressed:

```python
active_mask = model.get_active_mask()
print(f"Active Nodes: {np.sum(active_mask)} / {len(active_mask)}")
# Example Output: Active Nodes: 5 / 31 (16.1%)
```

Notice that only **16%** of the chromosome is actively coding! The remaining 84% provides the neutral reservoir for evolutionary drift.

---

## 4. Hyperparameter Recommendations

| Parameter | Recommended Default | Description |
| :--- | :---: | :--- |
| `n_rows` | `1` | Use `1` for standard 1D linear DAG. Use `>1` (e.g. 2–5) to restrict depth. |
| `n_cols` | `25 - 50` | Higher column counts increase capacity and neutral reservoir size. |
| `levels_back` | `-1` | `-1` allows unconstrained connections. Set to e.g. `5` for strict locality. |
| `pop_size` | `5` | Classic $(1+4)$-ES. Increasing $\lambda$ to 10 or 20 increases per-generation exploration. |
| `mutation_rate` | `None` (auto) | Automatically calibrated to mutate $\approx 3$ genes per individual: $3 / \text{total\_genes}$. |
