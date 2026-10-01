#!/usr/bin/env python3
"""Showcase: Modern Composer 3-Layer Architecture in MalthusJAX.

Demonstrates:
1. Type-safe, immutable ExperimentConfig construction and validation.
2. Bi-directional serialization (to_dict / from_dict).
3. The EngineFactory instantiation boundary.
4. Compliance with the runtime-checkable Engine protocol.
5. Multi-seed execution via BenchmarkRunner.
6. Registering and executing a custom third-party BackendProvider.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp

from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.engine_factory import EngineFactory
from malthusjax.composer.engine_protocol import Engine
from malthusjax.composer.experiment_config import (
    ExecutionConfig,
    ExperimentConfig,
    GenericBackendConfig,
    LoggingConfig,
    MalthusJAXBackendConfig,
    OutputConfig,
    PopulationConfig,
)


def showcase_typed_experiment() -> None:
    print("=" * 70)
    print("1. Typed ExperimentConfig & EngineFactory Boundary")
    print("=" * 70)

    # 1. Build an immutable, type-safe configuration tree
    config = ExperimentConfig(
        population=PopulationConfig(
            size=32,
            genome_type="real",
            genome_length=5,
            bounds=(-5.0, 5.0),
        ),
        execution=ExecutionConfig(
            generations=20,
            seeds=(42, 100),
            maximize=False,
        ),
        backend=MalthusJAXBackendConfig(
            fitness="sphere:dim=5",
            selection="tournament:tournament_size=3",
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1",
            elitism=2,
        ),
        logging=LoggingConfig(log_interval=5),
        output=OutputConfig(experiment_name="showcase_typed_run"),
    )

    # 2. Validate configuration
    config.validate()
    print("✓ Configuration validated successfully.")

    # 3. Serialization round-trip
    config_dict = config.to_dict()
    reconstructed = ExperimentConfig.from_dict(config_dict)
    assert reconstructed == config
    print("✓ Serialization round-trip (to_dict -> from_dict) verified.")

    # 4. Single-point instantiation via EngineFactory
    print("\nBuilding engine via EngineFactory.build(config)...")
    engine = EngineFactory.build(config)

    # 5. Protocol verification
    assert isinstance(engine, Engine), "Engine must satisfy Engine protocol"
    print("✓ Engine satisfied @runtime_checkable Engine protocol.")

    # 6. Execution via BenchmarkRunner
    print("Executing across seeds (42, 100)...")
    runner = BenchmarkRunner(engine=engine, experiment_name=config.output.experiment_name)
    results = runner.run(seeds=config.execution.seeds)

    summary = results.aggregated_summary()
    print(f"✓ Execution finished across {len(config.execution.seeds)} seeds.")
    print(f"  Best fitness mean: {summary.get('best_fitness', {}).get('mean', 'N/A'):.6f}")
    print(f"  Total evaluations: {summary.get('total_evaluations', {}).get('mean', 'N/A')}")


# =====================================================================
# Custom Backend Provider Demonstration
# =====================================================================


class CustomHillClimberEngine:
    """Minimal local search engine conforming to Engine protocol."""

    def __init__(self, dim: int, step_size: float, iterations: int):
        self.dim = dim
        self.step_size = step_size
        self.iterations = iterations

    def run_once(self, key: chex.Array, **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        k1, k2 = jax.random.split(key)
        # Initial point
        current = jax.random.uniform(k1, shape=(self.dim,), minval=-5.0, maxval=5.0)
        curr_fitness = float(jnp.sum(current**2))
        history: List[Dict[str, Any]] = []

        curr_key = k2
        for i in range(self.iterations):
            curr_key, subkey = jax.random.split(curr_key)
            step = jax.random.normal(subkey, shape=(self.dim,)) * self.step_size
            candidate = current + step
            cand_fitness = float(jnp.sum(candidate**2))
            if cand_fitness < curr_fitness:
                current = candidate
                curr_fitness = cand_fitness

            history.append({"generation": i, "best_fitness": curr_fitness})

        elapsed = time.perf_counter() - t0
        return {
            "history": history,
            "summary": {"best_fitness": curr_fitness, "total_evaluations": self.iterations},
            "timings": {"total": elapsed, "warmup": 0.0, "execution": elapsed},
        }


class ShowcaseHillClimberProvider:
    """Custom backend provider registering showcase_climber."""

    @property
    def name(self) -> str:
        return "showcase_climber"

    def default_strategy(self, **user_kwargs: Any) -> Any:
        return {"algorithm": "hill_climber"}

    def handles_strategy(self, strategy: Any) -> bool:
        return isinstance(strategy, dict) and strategy.get("algorithm") == "hill_climber"

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        return lambda x: jnp.sum(x**2)

    def build_engine(
        self,
        strategy: Any,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: tuple[float, float] = (-5.0, 5.0),
        history_metrics: Any = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Engine:
        dim = int(kwargs.get("genome_length", 5))
        iterations = generations
        return CustomHillClimberEngine(dim=dim, step_size=0.1, iterations=iterations)

    def generate_initial_population(self, config: Any, pop_seed: int) -> Any:
        return None


register_backend("showcase_climber", ShowcaseHillClimberProvider())


def showcase_custom_backend() -> None:
    print("\n" + "=" * 70)
    print("2. Extending MalthusJAX with a Custom BackendProvider")
    print("=" * 70)

    # Use GenericBackendConfig pointing to our registered provider
    config = ExperimentConfig(
        population=PopulationConfig(genome_length=5),
        execution=ExecutionConfig(generations=40, seeds=(1, 2)),
        backend=GenericBackendConfig(name="showcase_climber"),
        output=OutputConfig(experiment_name="showcase_climber_run"),
    )

    print("Building engine for registered 'showcase_climber' backend...")
    engine = EngineFactory.build(config)
    assert isinstance(engine, Engine)

    runner = BenchmarkRunner(engine=engine, experiment_name=config.output.experiment_name)
    results = runner.run(seeds=config.execution.seeds)

    summary = results.aggregated_summary()
    print("✓ Custom backend successfully resolved and executed:")
    print(f"  Best fitness mean: {summary.get('best_fitness', {}).get('mean', 'N/A'):.6f}")
    print(f"  Total evaluations: {summary.get('total_evaluations', {}).get('mean', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    showcase_typed_experiment()
    showcase_custom_backend()
