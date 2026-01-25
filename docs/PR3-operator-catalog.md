# PR3 — Operator Catalog & String-based Configuration

Status: Draft • Owner: you

## Summary
Implement the operator catalog system with `"operator:param=value"` string parsing to bridge MalthusJAX operators with `Composer.quick_run`. Replace `StubEngine` usage with real operator instances while keeping the product-first string-based UX. This enables real evolutionary computation in the benchmarking pipeline.

---

## Goals
- Build operator catalog with string parsing (`"tournament:selections=3,size=4"`)
- Register all major MalthusJAX operators (selection, crossover, mutation, fitness)
- Update `Composer.quick_run` to use real operators instead of `StubEngine`
- Maintain simple string-based UX while supporting complex operator configurations
- Add comprehensive tests for string parsing and operator instantiation

---

## Files to add or modify

### New
- `src/malthusjax/composer/catalog.py` — core catalog system:
  - `OperatorCatalog` class with string parsing
  - `parse_spec()` method for `"type:param1=val1,param2=val2"` format
  - Default operator type mappings (tournament, gaussian, arithmetic, etc.)
- `src/malthusjax/composer/engine_factory.py` — engine construction:
  - `build_engine()` function that takes parsed operators + creates real engine
  - Engine adapter if needed to match `BenchmarkRunner.Engine` protocol

### Modified
- `src/malthusjax/composer/composer.py` — update `quick_run()`:
  - Replace `StubEngine` with `OperatorCatalog` + `build_engine()`
  - Accept string operator parameters (selection, crossover, mutation, fitness)
  - Default operator strings for common use cases
- `src/malthusjax/composer/__init__.py` — export `OperatorCatalog`

### Tests
- `tests/composer/test_catalog.py` — string parsing and operator instantiation
- `tests/composer/test_engine_factory.py` — real engine construction
- `tests/composer/test_quick_run_real.py` — integration with real operators
- Update existing tests to use real operators where appropriate

---

## String Convention & Examples

### Parsing Format
```
"operator_type:param1=value1,param2=value2,param3=value3"
```

### Usage Examples
```python
# Simple cases (use defaults)
selection="tournament"          # → TournamentSelection() with defaults
mutation="gaussian"             # → GaussianMutation() with defaults

# With parameters
selection="tournament:selections=5,size=3"     # → TournamentSelection(selections=5, size=3)
mutation="gaussian:rate=0.2"                   # → GaussianMutation(rate=0.2)
crossover="arithmetic:alpha=0.5,num_offspring=2"  # → ArithmeticCrossover(alpha=0.5, num_offspring=2)
fitness="sphere:dim=10"                        # → SphereEvaluator(SphereConfig(dim=10))

# Full example
composer.quick_run(
    seeds=[1,2,3],
    fitness="griewank:dim=20",
    selection="tournament:selections=4,size=3",
    crossover="arithmetic:alpha=0.3",
    mutation="gaussian:rate=0.1",
    pop_size=50,
    generations=25,
)
```

---

## Default Operator Mappings

### Selection Operators
- `"tournament"` → `TournamentSelection`
- `"roulette"` → `RouletteWheelSelection`
- `"truncation"` → `TruncationSelection`

### Crossover Operators
- `"arithmetic"` → `ArithmeticCrossover`
- `"uniform"` → `UniformCrossover`
- `"onepoint"` → `OnePointCrossover` (if available)

### Mutation Operators
- `"gaussian"` → `GaussianMutation`
- `"bitflip"` → `BitFlipMutation`
- `"polynomial"` → `PolynomialMutation` (if available)

### Fitness Evaluators
- `"sphere"` → `SphereEvaluator` + `SphereConfig`
- `"griewank"` → `GriewankEvaluator` + `GriewankConfig`
- `"knapsack"` → `KnapsackEvaluator` + `KnapsackConfig`

---

## Implementation Strategy

### Phase 1: Catalog Foundation
1. Create `OperatorCatalog` class with string parsing
2. Add operator type mappings for major operators
3. Unit tests for string parsing edge cases

### Phase 2: Engine Integration
1. Create `build_engine()` that accepts parsed operators
2. Wire catalog → real `GeneticEngine` construction
3. Adapter layer if `GeneticEngine` interface doesn't match `BenchmarkRunner.Engine`

### Phase 3: Composer Integration
1. Update `Composer.quick_run()` to use catalog instead of `StubEngine`
2. Set sensible default operator strings
3. Integration tests with real evolutionary runs

---

## Engine Integration Details

### Current Challenge
`BenchmarkRunner` expects `Engine` protocol:
```python
class Engine(Protocol):
    def run_once(self, key: jr.PRNGKey) -> Dict[str, Any]: ...
```

But `GeneticEngine` likely has different interface. Need adapter:

```python
class GeneticEngineAdapter:
    def __init__(self, genetic_engine, genome_config, fitness_evaluator):
        self.engine = genetic_engine
        self.genome_config = genome_config
        self.fitness_evaluator = fitness_evaluator

    def run_once(self, key: jr.PRNGKey) -> Dict[str, Any]:
        # Run genetic_engine and format results for BenchmarkRunner
        # Return dict with 'history', 'summary', 'timings' keys
        ...
```

---

## Tests & Acceptance Criteria

### Unit Tests
- String parsing handles all formats correctly (`"op"`, `"op:param=val"`, `"op:p1=v1,p2=v2"`)
- Type conversion works (int, float, string parameters)
- Error handling for unknown operators and malformed strings
- All registered operators instantiate correctly with parsed parameters

### Integration Tests
- `Composer.quick_run()` with real operators produces sensible results
- Different operator combinations work together
- Small real evolutionary runs complete successfully (pop_size=10, generations=5)
- Results format matches existing `ExperimentResult` structure

### Performance
- Real operator runs should complete in reasonable time for small problems
- No import-time JAX compilation or device allocation

---

## Commands to run locally
```bash
# Test catalog functionality
PYTEST_ADDOPTS="" pytest -c /dev/null tests/composer/test_catalog.py -q

# Test real operator integration
PYTEST_ADDOPTS="" pytest -c /dev/null tests/composer/test_quick_run_real.py -q

# Full composer + benchmarking test suite
PYTEST_ADDOPTS="" pytest -c /dev/null tests/composer tests/benchmarking -q

# Type checking
mypy --ignore-missing-imports --python-version 3.12 src/malthusjax/composer
```

---

## Commit Plan (incremental)
1. **Commit A** — Add `OperatorCatalog` class + string parsing + tests
2. **Commit B** — Add default operator mappings + registration tests
3. **Commit C** — Add `build_engine()` + `GeneticEngineAdapter` + tests
4. **Commit D** — Update `Composer.quick_run()` to use catalog + integration tests
5. **Commit E** — Polish defaults and error handling

---

## Risks & Mitigations
- **Operator parameter mismatch** — test all registered operators thoroughly
- **Engine interface differences** — create adapter layer to bridge protocols
- **JAX compilation overhead** — keep test problems small and cache engines when possible
- **Complex parameter passing** — start with simple cases, expand gradually

---

## Reviewer Checklist
- String parsing handles edge cases gracefully
- All operator types can be instantiated via catalog
- Real evolutionary runs produce expected result format
- No performance regressions vs. StubEngine for small problems
- Documentation covers string format and available operators

---

## Next Steps
After PR2 is merged, create `feat/operator-catalog` branch and implement Commit A (OperatorCatalog + string parsing). This bridges the gap between stub engines and real MalthusJAX evolutionary computation.
