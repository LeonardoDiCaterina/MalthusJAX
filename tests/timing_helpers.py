"""Small production-protocol engine for deterministic timing contract tests."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.engine.base import AbstractEngine, AbstractEvolutionState
from malthusjax.engine.qd.map_elites import QDGenerationOutput


@struct.dataclass
class TimingEngine(AbstractEngine):
    trace_hook: Any = struct.field(pytree_node=False, default=None)

    def init_state(self, key, population=None, centroids=None):
        if population is None:
            values = jax.random.normal(key, (self.engine_params.pop_size, 2))
            population = BasePopulation(genes=values, fitness=jnp.sum(values**2, axis=1))
        return AbstractEvolutionState(
            population=population,
            best_genome=population.genes[0],
            generation=jnp.int32(0),
            best_fitness=jnp.min(population.fitness),
            rng_key=key,
        )

    def step(self, state):
        if self.trace_hook is not None:
            self.trace_hook()
        next_key, noise_key = jax.random.split(state.rng_key)
        score = state.best_fitness + jax.random.uniform(noise_key)
        next_state = state.replace(
            generation=state.generation + 1, best_fitness=score, rng_key=next_key
        )
        return next_state, QDGenerationOutput(
            best_fitness=score,
            mean_fitness=score,
            std_fitness=jnp.float32(0),
            generation=next_state.generation,
            qd_score=score,
            coverage=jnp.float32(100),
        )
