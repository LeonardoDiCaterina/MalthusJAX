"""Adapter for Island Models to conform with both GeneticEngine execution and BenchmarkRunner Engine protocol."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.logger import StepLoggingConfig, get_logger
from malthusjax.engine.island_model.base import BaseIslandModel

logger = get_logger("engine.island_model.adapter")


@struct.dataclass
class IslandEvolutionState:
    """Evolution state wrapper returned by IslandEngineAdapter.run().

    Provides consistent attribute access (best_genome, best_fitness, population, generation)
    compatible with GeneticEvolutionState.
    """

    multi_state: Any
    best_genome: Any
    best_fitness: Any
    population: Any
    generation: int = struct.field(pytree_node=False, default=0)


class IslandEngineAdapter:
    """Adapts a BaseIslandModel to expose .run(), .init_state(), .step(), and .run_once()."""

    def __init__(
        self,
        island_model: BaseIslandModel[Any],
        generations: int = 100,
        genome_config: Any = None,
        maximize: bool = False,
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Optional[StepLoggingConfig] = None,
    ) -> None:
        self.island_model = island_model
        self.generations = generations
        self.genome_config = genome_config
        self.maximize = maximize
        self.history_metrics = history_metrics
        self.step_logging = step_logging

    @property
    def engine(self) -> Any:
        return self.island_model.engine

    def init_state(self, key: chex.PRNGKey) -> Any:
        return self.island_model.init_state(key)

    def step(self, multi_state: Any) -> Tuple[Any, Any]:
        return self.island_model.step(multi_state)

    def run(self, init_state: Any, **kwargs: Any) -> Tuple[IslandEvolutionState, Any, Any]:
        """Runs the island model for total generations, executing outer migration intervals."""
        mig_interval = max(1, int(self.island_model.migration_interval))
        num_outer_steps = max(1, int(self.generations // mig_interval))

        def train_loop(carry_state: Any, _: Any) -> Tuple[Any, Any]:
            next_state, history_output = self.island_model.step(carry_state)
            return next_state, history_output

        final_multi_state, history = jax.lax.scan(
            train_loop, init_state, None, length=num_outer_steps
        )

        # Extract global best genome and fitness across all islands and individuals
        # final_multi_state.population.fitness shape: (num_islands, pop_size)
        fitness_matrix = final_multi_state.population.fitness
        if self.maximize:
            best_idx = jnp.argmax(fitness_matrix)
        else:
            best_idx = jnp.argmin(fitness_matrix)

        island_idx, ind_idx = jnp.unravel_index(best_idx, fitness_matrix.shape)
        global_best_fitness = fitness_matrix[island_idx, ind_idx]
        global_best_genome = jax.tree_util.tree_map(
            lambda arr: arr[island_idx, ind_idx], final_multi_state.population.genes
        )

        final_state = IslandEvolutionState(
            multi_state=final_multi_state,
            best_genome=global_best_genome,
            best_fitness=global_best_fitness,
            population=final_multi_state.population,
            generation=self.generations,
        )

        return final_state, history, None

    def run_once(
        self, key: chex.Array, step_logging: Optional[StepLoggingConfig] = None
    ) -> Dict[str, Any]:
        """BenchmarkRunner protocol implementation."""
        t_start = time.perf_counter()
        init_state = self.init_state(key)
        final_state, history, _ = self.run(init_state)
        t_end = time.perf_counter()

        best_fit = float(final_state.best_fitness)
        sign = -1.0 if self.maximize else 1.0
        reported_best = sign * best_fit

        summary = {
            "best_fitness": reported_best,
            "final_generation": self.generations,
        }
        timings = {
            "total": t_end - t_start,
            "execution": t_end - t_start,
            "warmup": 0.0,
        }
        return {
            "history": [],
            "summary": summary,
            "timings": timings,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.island_model, name)
