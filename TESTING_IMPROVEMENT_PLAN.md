# MalthusJAX Testing Improvement Strategy

## Executive Summary

Your testing suite has good foundations (pytest configured, coverage tracking, markers defined) but shows signs of iterative development. This plan addresses three key improvements:

1. **Cleanup**: Merge "_fixed" test artifacts back into parent files
2. **Invariants**: Add property-based tests for operator contracts
3. **Professionalization**: Adopt industry standards for test organization, documentation, and CI/CD

---

## Part 1: Test File Consolidation

### Current State: "_Fixed" Artifacts

You have three files that appear to be fixes applied during development that should be reintegrated:

| File | Context | Lines | Solution |
|------|---------|-------|----------|
| `test_engine_quality_fixed.py` | Real engine execution, statistical properties | ~200 | Merge into `test_genetic_engine.py` as `TestEngineExecutionQuality` |
| `test_engine_edge_cases_fixed.py` | Extreme parameters (pop size, generations, dims) | ~150 | Merge into `test_genetic_engine.py` as `TestEdgeCases` |
| `test_genetic_engine_fixes.py` | Engine dataclass defaults, elitism=0 handling | ~60 | Merge into `test_genetic_engine_core.py` |

### Why This Matters

- **Maintainability**: Related tests should be adjacent (easier code review, documentation reading)
- **Execution clarity**: Single engine test file is self-contained
- **Cleanup**: "_fixed" naming indicates technical debt (even if resolved)

### Merging Strategy

**For test_genetic_engine_fixes.py → test_genetic_engine_core.py:**
```python
# Instead of separate file, add to test_genetic_engine_core.py:
class TestEngineInvariants(unittest.TestCase):
    """Test engine dataclass contracts and edge cases like elitism=0."""
    
    def test_engine_dataclass_defaults_valid(self):
        """Ensure dataclass fields are struct instances, not functions."""
        # [existing test code]
    
    def test_engine_elitism_zero_runs(self):
        """Ensure engine handles elitism=0 edge case correctly."""
        # [existing test code]
```

**For test_engine_quality_fixed.py → test_genetic_engine.py:**
```python
class TestEngineExecutionQuality(unittest.TestCase):
    """Test real engine behavior: execution, shapes, optimization direction."""
    # [existing tests]

class TestEdgeCases(unittest.TestCase):
    """Test extreme but valid parameters (pop size, generations, dimensions)."""
    # [existing tests from test_engine_edge_cases_fixed.py]
```

### Merge Checklist

- [ ] Review both files to ensure no duplicate tests
- [ ] Move test methods + docstrings preserving intent
- [ ] Consolidate `setUp()` methods (extract common patterns to `conftest.py`)
- [ ] Update imports (remove redundancy)
- [ ] Run full test suite: `pytest tests/engine/ -v`
- [ ] Verify coverage doesn't drop: `pytest --cov=src/malthusjax --cov-report=term-missing`
- [ ] Delete orphaned files: `rm tests/engine/test_engine_*_fixed.py`
- [ ] Git commit: "refactor: consolidate engine tests, remove _fixed artifacts"

---

## Part 2: Property-Based Testing with Hypothesis

### The Gap

Your existing tests are **example-based**: "Given these inputs, check these outputs."

Property-based tests **find counterexamples**: "For ANY valid input matching this pattern, this invariant must hold."

### Core Operator Invariants

#### Mutation Operators
**File to create**: `tests/operators/mutation/test_hypothesis_invariants.py`

```python
from hypothesis import given, settings, strategies as st, assume
import jax.numpy as jnp
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.real import GaussianMutation

@given(
    pop_size=st.integers(min_value=2, max_value=100),
    genome_dim=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=500)
def test_mutation_never_escapes_bounds(pop_size, genome_dim):
    """INVARIANT: Mutation respects bounds regardless of input.
    
    For any population within bounds and any mutation_rate/strength,
    all offspring must remain within bounds.
    """
    config = RealGenomeConfig(shape=(genome_dim,), bounds=(-5.0, 5.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    
    mutation = GaussianMutation(
        mutation_rate=0.5,
        mutation_strength=0.3,
        num_offspring=1
    ).set_input_length(pop_size)
    
    offspring = mutation(
        jax.random.split(jax.random.PRNGKey(1), mutation.num_keys((pop_size,))),
        population,
        config
    )
    
    # Core invariant check
    assert jnp.all(offspring.values >= config.bounds[0]), \
        f"Mutation produced values below lower bound: {jnp.min(offspring.values)} < {config.bounds[0]}"
    assert jnp.all(offspring.values <= config.bounds[1]), \
        f"Mutation produced values above upper bound: {jnp.max(offspring.values)} > {config.bounds[1]}"


@given(
    mutation_rate=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_mutation_respects_rate(mutation_rate):
    """INVARIANT: mutation_rate=0 means no change, rate=1 means change element-wise.
    
    With rate=0, offspring == parent (identity operation).
    """
    assume(mutation_rate == 0.0)  # Focus on the edge case
    
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=5)
    original_values = jnp.array(population.values)
    
    mutation = GaussianMutation(mutation_rate=mutation_rate).set_input_length(5)
    offspring = mutation(
        jax.random.split(jax.random.PRNGKey(1), mutation.num_keys((5,))),
        population,
        config
    )
    
    assert jnp.allclose(offspring.values, original_values), \
        f"With rate={mutation_rate}, expected identity but got changes"


@given(
    pop_size=st.integers(min_value=2, max_value=50),
)
def test_mutation_preserves_shape(pop_size):
    """INVARIANT: Output shape matches input shape."""
    config = RealGenomeConfig(shape=(15,), bounds=(-1.0, 1.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    
    mutation = GaussianMutation(num_offspring=1).set_input_length(pop_size)
    offspring = mutation(
        jax.random.split(jax.random.PRNGKey(1), mutation.num_keys((pop_size,))),
        population,
        config
    )
    
    assert offspring.values.shape[0] == pop_size, \
        f"Expected {pop_size} individuals, got {offspring.values.shape[0]}"
```

#### Crossover Operators
**File to create**: `tests/operators/crossover/test_hypothesis_invariants.py`

```python
@given(
    pop_size=st.integers(min_value=2, max_value=50),
)
def test_crossover_offspring_within_parental_range(pop_size):
    """INVARIANT: Offspring alleles come from parental ranges.
    
    For real-valued crossover, each offspring dimension should be
    within [min(parent1[i], parent2[i]), max(parent1[i], parent2[i])].
    """
    config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    
    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    
    # Select pairs and verify
    for i in range(0, pop_size - 1, 2):
        parent1_vals = population.values[i]
        parent2_vals = population.values[i + 1]
        
        min_vals = jnp.minimum(parent1_vals, parent2_vals)
        max_vals = jnp.maximum(parent1_vals, parent2_vals)
        
        # Offspring should be within this range (accounting for mutation extension)
        # For conservative check: use larger tolerance
        offspring_within_range = (offspring.values >= min_vals - 1e-5) & \
                                 (offspring.values <= max_vals + 1e-5)
        assert jnp.all(offspring_within_range)


@given(
    pop_size=st.integers(min_value=2, max_value=50),
)
def test_crossover_produces_correct_count(pop_size):
    """INVARIANT: Produces exactly num_offspring individuals per pair."""
    config = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    
    num_offspring = 2
    crossover = SimulatedBinaryCrossover(num_offspring=num_offspring)
    
    offspring = crossover(
        jax.random.split(jax.random.PRNGKey(1), crossover.num_keys((pop_size,))),
        population,
        config
    )
    
    expected_count = (pop_size // 2) * num_offspring
    assert offspring.values.shape[0] == expected_count
```

#### Selection Operators
**File to create**: `tests/operators/selection/test_hypothesis_invariants.py`

```python
@given(
    pop_size=st.integers(min_value=5, max_value=100),
)
def test_selection_returns_exact_count(pop_size):
    """INVARIANT: Selection always returns exactly num_selections individuals."""
    config = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    population.fitness = jax.random.uniform(
        jax.random.PRNGKey(1), shape=(pop_size,)
    )
    
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=pop_size // 2)
    
    selected = selection(
        jax.random.PRNGKey(2),
        population,
        None
    )
    
    assert len(selected.indices) == pop_size


@given(
    maximize=st.booleans(),
)
def test_elite_selection_chooses_best(maximize):
    """INVARIANT: Elite selection picks highest/lowest fitness depending on maximize."""
    pop_size = 20
    config = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))
    population = RealPopulation.init_random(jax.random.PRNGKey(0), config, size=pop_size)
    
    # Create known fitness landscape
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    population.fitness = fitness
    
    selection = ElitePoolSelection(num_selections=pop_size, elite_k=5)
    selected = selection(jax.random.PRNGKey(1), population, None)
    
    selected_fitness = fitness[selected.indices]
    
    if maximize:
        # Should prefer higher values
        assert jnp.mean(selected_fitness) >= jnp.median(fitness)
    else:
        # Should prefer lower values
        assert jnp.mean(selected_fitness) <= jnp.median(fitness)
```

### Install Hypothesis

Add to `pyproject.toml` dev dependencies:
```toml
dev = [
    # ... existing ...
    "hypothesis>=6.80.0",
]
```

### Run Hypothesis Tests

```bash
# Run with verbose output to see shrinking process
pytest tests/operators/ -v -m invariant

# Generate HTML report of test runs
pytest tests/operators/ --hypothesis-show-statistics
```

---

## Part 3: Making Tests "Professional"

### 3.1 Test Documentation

**Problem**: "Test what?" is hard to answer without reading code.

**Solution**: Adopt clear docstring patterns:

```python
def test_mutation_respects_bounds(self):
    """Verify mutation respects genome bounds [MUST-HAVE INVARIANT].
    
    Reference: PR#42 discovered mutation could escape bounds via
    unclipped Gaussian sampling.
    
    Invariant: For any population within bounds and any params,
    all offspring ∈ bounds.
    """
    # test implementation
```

### 3.2 Fixture Organization

**Current**: Fixtures scattered, `setUp()` methods duplicated.

**Target**: Centralized, reusable fixtures in `conftest.py` hierarchy.

Create `tests/conftest.py` (root):
```python
import pytest
import jax.random as jar
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation

@pytest.fixture
def prng_key_factory():
    """Factory for deterministic PRNG keys."""
    counter = [0]
    def _make_key():
        counter[0] += 1
        return jar.PRNGKey(counter[0])
    return _make_key

@pytest.fixture
def real_genome_config_factory():
    """Factory for RealGenomeConfig with various bounds/shapes."""
    def _make(shape=(10,), bounds=(-5.0, 5.0)):
        return RealGenomeConfig(shape=shape, bounds=bounds)
    return _make

@pytest.fixture
def random_population_factory(prng_key_factory, real_genome_config_factory):
    """Factory for random populations."""
    def _make(size=10, config=None):
        if config is None:
            config = real_genome_config_factory()
        return RealPopulation.init_random(prng_key_factory(), config, size=size)
    return _make
```

Then in tests:
```python
def test_mutation_example(random_population_factory):
    """Instead of setUp, use fixture."""
    population = random_population_factory(size=20)
    mutation = GaussianMutation(...)
    # test code
```

Create `tests/engine/conftest.py`:
```python
import pytest
import jax.random as jar
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.core.fitness.real_evaluators import SphereEvaluator, SphereConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig

@pytest.fixture
def standard_engine_params():
    """Standard engine params for testing."""
    return GeneticEngineParams(
        pop_size=30,
        elitism=2,
        num_generations=10
    )

@pytest.fixture
def sphere_evaluator():
    """Pre-configured Sphere evaluator."""
    return SphereEvaluator(config=SphereConfig(num_dims=5))
```

### 3.3 Test Markers & Organization

Add to `pyproject.toml`:
```toml
markers = [
    "slow: slow tests (>1s), run with -m 'not slow'",
    "integration: integration tests requiring multiple components",
    "benchmark: performance benchmarks",
    "invariant: property-based invariant tests",
    "regression: regression tests for specific issues",
    "performance: performance-sensitive tests",
]
```

Use them:
```python
@pytest.mark.invariant
def test_mutation_bounds_invariant(self):
    """Property-based test using Hypothesis."""
    pass

@pytest.mark.slow
def test_large_population_benchmark(self):
    """Benchmark with 1000+ population."""
    pass

@pytest.mark.regression
def test_issue_42_mutation_bounds(self):
    """Regression test for Issue #42: mutation escaping bounds."""
    pass
```

### 3.4 Type Hints & Static Analysis

**Add to `pyproject.toml`**:
```toml
[tool.mypy]
# ... existing settings ...

[[tool.mypy.overrides]]
module = "tests.*"
# Allow lenient typing in tests
disallow_untyped_defs = false
```

**Apply type hints to test files**:
```python
from typing import Callable
import pytest
from malthusjax.core.genome.real_genome import RealPopulation

def test_mutation_example(
    random_population_factory: Callable[..., RealPopulation]
) -> None:
    """Type-hinted test."""
    population = random_population_factory(size=20)
    assert isinstance(population, RealPopulation)
```

### 3.5 Test Independence & Cleanup

**Problem**: Tests can affect each other through shared state.

**Solution**:
- Always use `@pytest.fixture(autouse=True)` for cleanup if needed
- Never rely on test execution order (set `random_order` in pytest)
- Use immutable test data where possible

```python
import pytest

@pytest.fixture(autouse=True)
def reset_jax_config():
    """Reset JAX config between tests."""
    import jax
    original_config = jax.config.update('jax_disable_jit', True)
    yield
    jax.config.update('jax_disable_jit', original_config)
```

### 3.6 Coverage Targets

Update `pyproject.toml`:
```toml
[tool.coverage.run]
branch = true
source = ["src/malthusjax"]

[tool.coverage.report]
# Fail if coverage drops below this
fail_under = 75.0
precision = 2

exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if not self._debug",
]
```

Check coverage:
```bash
pytest --cov=src/malthusjax --cov-report=html tests/
open htmlcov/index.html
```

### 3.7 Performance Testing Infrastructure

Add to `tests/engine/conftest.py`:
```python
import pytest

@pytest.fixture
def benchmark_config():
    """Config for performance benchmarks."""
    return {
        'pop_size': 100,
        'num_gens': 50,
        'num_seeds': 10,  # average over multiple runs
    }

@pytest.mark.benchmark
def test_engine_throughput_benchmark(benchmark_config):
    """Benchmark: single-step throughput (evals/sec)."""
    # Load real config, verify performance hasn't regressed
    pass
```

### 3.8 Create TESTING.md

Create `TESTING.md` at repo root:

```markdown
# Testing Guide for MalthusJAX

## Quick Start

```bash
# Run all tests
pytest

# Run fast tests only (skip slow)
pytest -m "not slow"

# Run with coverage report
pytest --cov=src/malthusjax --cov-report=html

# Run property-based tests
pytest -m invariant -v
```

## Test Organization

```
tests/
├── conftest.py                    # Root fixtures
├── core/
│   ├── conftest.py                # Core-specific fixtures
│   ├── fitness/
│   │   ├── test_bbob_evaluator.py
│   │   └── ...
│   └── genome/
├── engine/
│   ├── conftest.py                # Engine-specific fixtures
│   ├── test_genetic_engine.py     # All engine tests
│   ├── test_genetic_engine_core.py
│   └── ...
├── operators/
│   ├── mutation/
│   │   ├── test_hypothesis_invariants.py  # NEW
│   │   ├── test_gaussian.py
│   │   └── ...
│   ├── crossover/
│   │   ├── test_hypothesis_invariants.py  # NEW
│   │   └── ...
│   └── selection/
│       ├── test_hypothesis_invariants.py  # NEW
│       └── ...
└── composer/
```

## Writing New Tests

### Pattern 1: Example-Based (Traditional)

```python
def test_mutation_example():
    """Test specific scenario with known inputs."""
    # Arrange
    config = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))
    population = RealPopulation.init_random(JAX.random.PRNGKey(0), config, size=10)
    
    # Act
    mutation = GaussianMutation(mutation_rate=0.5, mutation_strength=0.1)
    offspring = mutation(...)
    
    # Assert
    assert offspring.values.shape == population.values.shape
```

### Pattern 2: Property-Based (Hypothesis)

```python
from hypothesis import given

@given(pop_size=st.integers(2, 100))
def test_mutation_shape_invariant(pop_size):
    """Test that mutation preserves shape for any valid input."""
    # Hypothesis generates pop_size values
    # Your test checks the invariant holds for all of them
    ...
```

### Using Fixtures

```python
def test_with_fixture(random_population_factory):
    population = random_population_factory(size=20)
    # test code
```

## Coverage Targets

- **Core** (malthusjax.core): ≥85%
- **Operators** (malthusjax.operators): ≥80%
- **Engine** (malthusjax.engine): ≥75%
- **Composer** (malthusjax.composer): ≥70%

Run coverage report:
```bash
pytest --cov=src/malthusjax --cov-report=html
open htmlcov/index.html
```

## Test Markers

- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.integration` - Tests requiring multiple components
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.invariant` - Property-based invariant tests
- `@pytest.mark.regression` - Regression tests for specific issues

Run only fast tests:
```bash
pytest -m "not slow"
```

## Continuous Integration

Tests run on:
- Python 3.10, 3.11, 3.12
- Both CPU and GPU backends (if available)
- All commits and PRs

View results: [GitHub Actions](https://github.com/LeonardoDiCaterina/MalthusJAX/actions)
```

---

## Summary: Implementation Roadmap

| Phase | Task | Est. Time | Priority |
|-------|------|-----------|----------|
| 1 | Delete "_fixed" files, merge tests | 2 hours | HIGH |
| 2 | Create `test_hypothesis_invariants.py` files (3 files) | 4 hours | HIGH |
| 3 | Refactor conftest.py hierarchy | 2 hours | MEDIUM |
| 4 | Add test markers & documentation | 2 hours | MEDIUM |
| 5 | Create TESTING.md guide | 1 hour | HIGH |
| **Total** | | **~11 hours** | |

### Quick Wins (Do First)

1. **Delete _fixed files** and merge tests → instant cleanup ✓
2. **Create TESTING.md** → immediately documents intent ✓
3. **Add Hypothesis tests** for mutation bounds → catches bugs, looks professional ✓

---

## Questions to Guide Your Implementation

1. **Are there any test-specific fixture patterns you already use heavily?** Knowing this will help you design conftest.py more effectively.

2. **What are your coverage gaps?** (Check: `pytest --cov-report=term-missing`) This will guide where to add property-based tests.

3. **Do you run tests in CI/CD?** If yes, what's your tolerance for test runtime? (Property-based tests can be slower.)

4. **Are there known operator bugs you want regression tests for?** Reference them in test docstrings.

---

**Next Steps**: 
- Start with Phase 1 (merge files) for immediate cleanup
- Then add 2-3 property-based tests (Phase 2) to demonstrate the pattern
- Finally, document everything (Phase 3) for team alignment
