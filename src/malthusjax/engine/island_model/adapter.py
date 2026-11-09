"""Adapter for Island Models to conform with both GeneticEngine execution and BenchmarkRunner Engine protocol."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.logger import StepLoggingConfig, get_logger
from malthusjax.core.timing import TimingCapture, block_until_ready, execution_metadata
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
        self._compiled_run = None
        self._compiled_spec = None

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
        capture = TimingCapture()
        init_state = self.init_state(key)
        block_until_ready(init_state, effects=True)
        capture.initialized(init_state, self)
        spec = (self.generations, id(self.island_model))
        if self._compiled_spec != spec:
            self._compiled_run = jax.jit(self.run)
            self._compiled_spec = spec
        prepared = self._compiled_run.lower(init_state).compile()
        capture.prepared()
        t_warmup_end = time.perf_counter()
        t_execution_start = time.perf_counter()
        final_state, history, _ = prepared(init_state)
        block_until_ready((final_state, history), effects=True)
        t_end = time.perf_counter()
        capture.executed()

        best_fit = float(final_state.best_fitness)
        sign = -1.0 if self.maximize else 1.0
        reported_best = sign * best_fit

        summary = {
            "best_fitness": reported_best,
            "final_generation": self.generations,
        }
        timings = {
            "total": t_end - t_start,
            "warmup": t_warmup_end - t_start,
        }
        interval = int(self.island_model.migration_interval)
        unavailable = None
        if self.generations <= 0 or interval <= 0 or self.generations % interval:
            unavailable = (
                "Island horizon/interval contract requires P6; execution timing unavailable"
            )
        elif step_logging is not None or self.step_logging is not None:
            unavailable = "Island step logging is not implemented by this adapter"
        else:
            timings["execution"] = t_end - t_execution_start
        return {
            "history": [],
            "summary": summary,
            "timings": timings,
            "timing_metadata": capture.metadata(unavailable),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.island_model, name)
