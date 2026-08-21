# `src/` Refactoring Commit Plan

To ensure clean history, backwards compatibility, and zero regressions, the changes in `src/` will be executed across four atomic, well-scoped commits.

---

## Commit 1: `refactor(core): add clone_buffers() and document buffer donation`

### Scope
- `src/malthusjax/core/base.py`
- `src/malthusjax/core/genome/*.py`
- `src/malthusjax/core/README.md`

### Objectives
- Add `clone_buffers()` method on `BaseGenome` and `BasePopulation` to explicitly signal JAX array buffer duplication for XLA buffer donation (`donate_argnums`).
- Retain `copy()` as an alias of `clone_buffers()` for backwards compatibility.
- Update docstrings across PyTree base classes to explain buffer ownership semantics.

---

## Commit 2: `refactor(engine): make 5-phase step pipeline explicit in GeneticFastEngine`

### Scope
- `src/malthusjax/engine/genetic_fastengine.py`
- `src/malthusjax/engine/genetic_engine.py`

### Objectives
- Refactor `step(self, state)` so the high-level control flow reads like clean pseudocode:
  ```python
  def step(self, state):
      state, keys = self.allocate_entropy(state)
      selected = self.selection_phase(state, keys.selection_key)
      offspring = self.reproduction_phase(state, selected, keys.reproduction_key)
      population = self.merge_phase(state, offspring)
      return self.evaluation_phase(state, population, keys.evaluation_key)
  ```
- Ensure zero behavioral changes or JIT tracing regressions by preserving underlying phase implementations.

---

## Commit 3: `refactor(composer): modularize composer and consolidate adapters`

### Scope
- `src/malthusjax/composer/composer.py` (~77 KB breakdown)
- `src/malthusjax/composer/adapters/`
- `src/malthusjax/composer/comparison.py` (New)

### Objectives
- Move external framework adapters (`evosax`, `qdax`, `tensorneat`, `kozax`) into `src/malthusjax/composer/adapters/`.
- Extract comparison table generator and statistical runner out of `composer.py` into `src/malthusjax/composer/comparison.py`.
- Reduce `composer.py` to a clean high-level orchestration interface (`build()`, `run()`, `compare()`).

---

## Commit 4: `docs(operators): document progressive 3-tier vectorization API`

### Scope
- `src/malthusjax/operators/base.py`
- `src/malthusjax/operators/README.md`

### Objectives
- Add docstrings to base operator classes (`BaseSelection`, `BaseCrossover`, `BaseMutation`) explaining the three vectorization levels:
  - **Tier 1 (Genome Level)**: Single individual, lifted via `jax.vmap`.
  - **Tier 2 (Noise Level)**: Controlled stochastic noise generation.
  - **Tier 3 (Population Level)**: Custom population-wide kernels.
- Add practical guidance on trading abstraction for explicit batching/vectorization control.

---

## Verification Strategy

For each commit:
1. Run `pytest tests/` (or `pytest tests/core/` / `pytest tests/engine/` for specific modules).
2. Run `mypy src/malthusjax/core` to verify strict typing.
3. Verify all entry points (`mjax`) continue to function cleanly.
