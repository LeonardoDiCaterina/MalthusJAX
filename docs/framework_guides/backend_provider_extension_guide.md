# Backend Provider Extension Guide 🔌

## 📚 Overview

In MalthusJAX, the **Composer** layer decouples high-level experiment configuration from backend execution via the **Backend Provider** pattern.

A **Backend Provider** (`src/malthusjax/composer/backend_provider.py`) is a factory bridge that knows how to:
1. Parse framework-specific options from an [ExperimentConfig](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/experiment_config.py).
2. Resolve problem evaluators into the format expected by the target engine.
3. Instantiate and wire an engine conforming to the runtime-checkable [Engine](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/engine_protocol.py) protocol.

By writing a custom `BackendProvider`, you can integrate any evolutionary or optimization library (such as EvoX, PyTorch-based ES, Brax, Gymnax, or proprietary algorithms) and execute them natively inside `Composer.quick_run()`, `Composer.compare()`, or declarative `.toml` benchmark sweeps.

---

## 1️⃣ The `BackendProvider` Contract

Every backend provider must subclass [BackendProvider](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/backend_provider.py) and implement three core methods:

```python
from typing import Protocol, runtime_checkable, Any, Optional, Tuple, Sequence
import chex
from malthusjax.composer.engine_protocol import Engine

@runtime_checkable
class BackendProvider(Protocol):
    @property
    def name(self) -> str:
        """Canonical backend name used as the primary registry key."""
        ...

    def default_strategy(self, **user_kwargs: Any) -> Any:
        """Create the default Strategy dataclass or dictionary from user-facing kwargs."""
        ...

    def handles_strategy(self, strategy: Any) -> bool:
        """Return True if this provider can handle the given strategy instance."""
        ...

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        """Resolve a fitness spec into a concrete evaluator callable."""
        ...

    def build_engine(
        self,
        strategy: Any,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Engine:
        """Build and return an Engine conforming to the Engine protocol."""
        ...

    def generate_initial_population(
        self,
        config: Any,
        pop_seed: int,
    ) -> Optional[Any]:
        """Generate a shared initial population for compare() parity."""
        ...
```

---

## 2️⃣ The `Engine` Protocol Contract

Any engine returned by `build_engine()` must conform to the `@runtime_checkable` [Engine](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/engine_protocol.py) protocol:

```python
import chex
from typing import Protocol, Dict, Any, runtime_checkable

@runtime_checkable
class Engine(Protocol):
    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Execute a single run for a given PRNG key and return structured results."""
        ...
```

The returned dictionary must contain three top-level keys:
- `"history"`: A list or sequence of per-generation metric dictionaries (e.g. `[{"generation": 0, "best_fitness": 1.2}, ...]`).
- `"summary"`: A dictionary of final scalar run metrics (e.g. `{"best_fitness": 0.01, "total_evaluations": 5000}`).
- `"timings"`: A dictionary of execution durations in seconds (e.g. `{"total": 0.42, "warmup": 0.05, "execution": 0.37}`).

> [!TIP]
> If you are adapting an external library that does not natively output this dictionary structure, wrap it using [UniversalAdapterEngine](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/adapters/base.py) or write a lightweight 10-line adapter class.

---

## 3️⃣ Registering Your Backend

MalthusJAX provides a centralized [BackendRegistry](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/backend_registry.py) that strictly enforces contract compliance at registration time.

```python
from malthusjax.composer.backend_registry import register_backend

# Instantiate and register your provider
provider = MyCustomProvider()
register_backend("my_custom_backend", provider)
```

> [!IMPORTANT]
> The registry validates that registered providers implement callable `build_engine`, `resolve_evaluator`, and `default_strategy` methods. If any method is missing or not callable, `register_backend()` immediately raises a descriptive `TypeError`.

---

## 4️⃣ Complete Working Example: `RandomSearchProvider`

Here is a complete, production-ready custom backend provider that implements pure random search over continuous spaces:

```python
import time
from typing import Any, Dict, List, Optional
import chex
import jax
import jax.numpy as jnp

from malthusjax.composer.backend_provider import BackendProvider
from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.engine_protocol import Engine


class RandomSearchEngine:
    """Minimal engine conforming to the Engine protocol."""

    def __init__(self, evaluator: Any, dim: int, bounds: tuple, pop_size: int, generations: int):
        self.evaluator = evaluator
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        t0 = time.perf_counter()
        low, high = self.bounds
        best_fitness = float("inf")
        history: List[Dict[str, Any]] = []

        curr_key = key
        for gen in range(self.generations):
            curr_key, subkey = jax.random.split(curr_key)
            candidates = jax.random.uniform(
                subkey, shape=(self.pop_size, self.dim), minval=low, maxval=high
            )
            # Evaluate candidates
            fitnesses = jax.vmap(self.evaluator)(candidates)
            gen_best = float(jnp.min(fitnesses))
            if gen_best < best_fitness:
                best_fitness = gen_best

            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "mean_fitness": float(jnp.mean(fitnesses)),
            })

        elapsed = time.perf_counter() - t0
        return {
            "history": history,
            "summary": {
                "best_fitness": best_fitness,
                "total_evaluations": self.pop_size * self.generations,
            },
            "timings": {"total": elapsed, "warmup": 0.0, "execution": elapsed},
        }


class RandomSearchProvider:
    """Custom backend provider for Random Search."""

    @property
    def name(self) -> str:
        return "random_search"

    def default_strategy(self, **user_kwargs: Any) -> Any:
        return {"name": "uniform_random_search"}

    def handles_strategy(self, strategy: Any) -> bool:
        return isinstance(strategy, dict) and strategy.get("name") == "uniform_random_search"

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        if callable(fitness_spec):
            return fitness_spec
        # Fallback sphere evaluator for demonstration
        return lambda x: jnp.sum(x ** 2)

    def build_engine(
        self,
        strategy: Any,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Engine:
        dim = int(kwargs.get("genome_length", 10))
        return RandomSearchEngine(
            evaluator=evaluator,
            dim=dim,
            bounds=bounds,
            pop_size=pop_size,
            generations=generations,
        )

    def generate_initial_population(self, config: Any, pop_seed: int) -> Optional[Any]:
        return None


# Register an instance of the provider
register_backend("random_search", RandomSearchProvider())
```

---

## 5️⃣ Using Your Backend in Experiments

Once registered, your backend is seamlessly accessible across all composer entry points!

### Via Typed Configuration
```python
from malthusjax.composer.experiment_config import (
    ExperimentConfig, PopulationConfig, ExecutionConfig, GenericBackendConfig
)
from malthusjax.composer.engine_factory import EngineFactory
from malthusjax.benchmarking.runner import BenchmarkRunner

config = ExperimentConfig(
    population=PopulationConfig(size=32, genome_length=10),
    execution=ExecutionConfig(generations=50, seeds=(42, 43)),
    backend=GenericBackendConfig(name="random_search"),
)

engine = EngineFactory.build(config)
runner = BenchmarkRunner(engine=engine, experiment_name="random_search_run")
results = runner.run(seeds=config.execution.seeds)
print(results.aggregated_summary())
```

### Via Interactive `quick_run()`
```python
from malthusjax.composer.composer import Composer

composer = Composer.create_default()
result = composer.quick_run(
    backend="random_search",
    fitness="sphere:dim=10",
    pop_size=32,
    generations=50,
    seeds=[42, 43],
)
print(result.aggregated_summary())
```

### Via Declarative TOML Benchmarks
```toml
[experiment]
name = "benchmark_with_random_search"
output_dir = "results/comparison"

[experiment.shared]
fitness = "sphere:dim=10"
pop_size = 50
generations = 100
genome_length = 10
seeds = [1, 2, 3]

[pipelines.malthus_ga]
backend = "malthusjax"
selection = "tournament:tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.baseline_random]
backend = "random_search"
```

```python
results = composer.from_toml("benchmark.toml")
print(results.summary_table())
```

---

## 6️⃣ Unit Testing Your Custom Provider

To ensure your custom provider integrates seamlessly with the MalthusJAX test suite, add a test using `pytest`:

```python
import jax
from malthusjax.composer.backend_registry import get_backend
from malthusjax.composer.engine_protocol import Engine
from malthusjax.composer.experiment_config import ExperimentConfig, GenericBackendConfig

def test_random_search_provider_registration():
    provider_cls = get_backend("random_search")
    provider = provider_cls()
    
    # Test contract methods
    evaluator = provider.resolve_evaluator("sphere:dim=5")
    strategy = provider.default_strategy()
    engine = provider.build_engine(strategy, evaluator, genome_length=5, pop_size=10, generations=5)
    
    # Verify Engine protocol
    assert isinstance(engine, Engine)
    
    # Verify execution output schema
    out = engine.run_once(jax.random.PRNGKey(0))
    assert "history" in out
    assert "summary" in out
    assert "timings" in out
    assert out["summary"]["total_evaluations"] == 50
```
