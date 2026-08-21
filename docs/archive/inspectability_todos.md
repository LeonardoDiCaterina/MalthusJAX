# Future Inspectability Improvements

This document tracks planned convenience patterns and magic methods to make JAX state objects and engines more inspectable, especially for interactive development (Jupyter notebooks and CLI debug runs).

## Planned Changes

### 1. Rich Representations (`__repr__` and `__repr_html__`)
Implement clean, summarized string and HTML representations for complex state objects to avoid dumping massive raw JAX arrays to the console.
- **Target:** `GeneticEvolutionState`
  - Show current generation number.
  - Show population size (e.g. `len(population)`).
  - Show the current best fitness value.
- **Target:** `BasePopulation`
  - Show the population size.
  - Show fitness summary statistics (min, max, mean).
  - Format as a neat HTML table when printed in Jupyter notebooks using `_repr_html_()`.

### 2. Convenience Properties on `GeneticEvolutionState`
Expose read-only properties to quickly fetch key evolution indicators without navigating nested struct hierarchies:
- `state.best_individual`: Returns the current `best_genome` wrapper.
- `state.size`: Returns the population size.

### 3. Guardrails for Implementation
- **No Mutability:** Ensure all custom helpers and properties are read-only to satisfy JAX's stateless requirements.
- **Keep PyTree Registration Intact:** Keep these properties strictly helper-based to ensure they do not interfere with Flax's `@struct.dataclass` PyTree flattening/unflattening behavior.
- **Avoid Performance Anti-Patterns:** Keep magic methods like `__iter__` or loop-inducing helpers strictly out of core engine loops (using them only in interactive outer scopes).
