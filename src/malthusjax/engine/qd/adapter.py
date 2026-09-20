"""Adapter for Quality-Diversity MAP-Elites and Island MAP-Elites Engines.

Provides uniform .init_state(), .step(), and .run() execution interfaces compatible
with MalthusJAX Composer and Scikit-Learn estimators.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from qdax.core.containers.mapelites_repertoire import (
    compute_cvt_centroids,
)

from malthusjax.core.logger import StepLoggingConfig, get_logger
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesState

logger = get_logger("engine.qd.adapter")


@struct.dataclass
class MapElitesEvolutionState:
    """Standardized evolution state wrapper for MAP-Elites."""

    state: MapElitesState[Any, Any]
    best_genome: Any
    best_fitness: Any
    repertoire: Any
    population: Any
    generation: int = struct.field(pytree_node=False, default=0)


@struct.dataclass
class IslandMapElitesEvolutionState:
    """Standardized evolution state wrapper for Island MAP-Elites."""

    multi_state: Any
    best_genome: Any
    best_fitness: Any
    repertoire: Any
    generation: int = struct.field(pytree_node=False, default=0)


class MapElitesEngineAdapter:
    """Adapts MapElitesEngine to expose standard .init_state(), .step(), and .run()."""

    def __init__(
        self,
        engine: MapElitesEngine[Any, Any],
        genome_config: Any,
        generations: int = 100,
        num_centroids: int = 50,
        num_init_cvt_samples: int = 5000,
        minval: Sequence[float] = (0.0, 0.0),
        maxval: Sequence[float] = (1.0, 1.0),
        maximize: bool = False,
        step_logging: Optional[StepLoggingConfig] = None,
        centroids_key: int = 42,
    ) -> None:
        self.engine = engine
        self.genome_config = genome_config
        self.generations = generations
        self.num_centroids = num_centroids
        self.num_init_cvt_samples = num_init_cvt_samples
        self.minval = list(minval)
        self.maxval = list(maxval)
        self.maximize = maximize
        self.step_logging = step_logging

        # Precompute CVT Voronoi centroids for static repertoire layout
        k_cent = jax.random.PRNGKey(centroids_key)
        self.centroids: jnp.ndarray = compute_cvt_centroids(
            num_descriptors=len(self.minval),
            num_init_cvt_samples=self.num_init_cvt_samples,
            num_centroids=self.num_centroids,
            minval=self.minval,
            maxval=self.maxval,
            key=k_cent,
        )

    def init_state(self, key: chex.PRNGKey) -> MapElitesState[Any, Any]:
        """Initializes the MAP-Elites state with initial population and repertoire."""
        k_pop, k_state = jax.random.split(key)
        pop_size = getattr(self.engine.engine_params, "pop_size", 20)
        initial_population = self.genome_config.init_population(k_pop, pop_size)
        return self.engine.init_state(k_state, initial_population, self.centroids)

    def step(self, state: MapElitesState[Any, Any]) -> Tuple[MapElitesState[Any, Any], Any]:
        """Steps one generation of MAP-Elites."""
        return self.engine.step(state)

    def run(
        self, init_state: MapElitesState[Any, Any], key: Optional[chex.PRNGKey] = None
    ) -> Tuple[MapElitesEvolutionState, Any, int]:
        """Executes MAP-Elites evolution across self.generations using jax.lax.scan."""
        start_time = time.perf_counter()

        def _loop(
            cur_state: MapElitesState[Any, Any], _: Any
        ) -> Tuple[MapElitesState[Any, Any], Any]:
            next_state, kpi = self.engine.step(cur_state)
            return next_state, kpi

        final_inner_state, kpis = jax.lax.scan(_loop, init_state, None, length=self.generations)

        elapsed = time.perf_counter() - start_time
        logger.debug(
            "MapElitesEngine completed %d generations in %.3fs (Best Fitness: %s)",
            self.generations,
            elapsed,
            final_inner_state.best_fitness,
        )

        final_state = MapElitesEvolutionState(
            state=final_inner_state,
            best_genome=final_inner_state.best_genome,
            best_fitness=final_inner_state.best_fitness,
            repertoire=final_inner_state.repertoire,
            population=final_inner_state.population,
            generation=final_inner_state.generation,
        )

        return final_state, kpis, self.generations


class IslandMapElitesAdapter:
    """Distributed Quality-Diversity Island Model across isolated MAP-Elites repertoires."""

    def __init__(
        self,
        engine: MapElitesEngine[Any, Any],
        genome_config: Any,
        num_islands: int = 4,
        migration_interval: int = 20,
        num_migrants: int = 2,
        generations: int = 100,
        num_centroids: int = 50,
        num_init_cvt_samples: int = 5000,
        minval: Sequence[float] = (0.0, 0.0),
        maxval: Sequence[float] = (1.0, 1.0),
        maximize: bool = False,
        topology: str = "ring",
        centroids_key: int = 42,
    ) -> None:
        self.engine = engine
        self.genome_config = genome_config
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.num_migrants = num_migrants
        self.generations = generations
        self.num_centroids = num_centroids
        self.num_init_cvt_samples = num_init_cvt_samples
        self.minval = list(minval)
        self.maxval = list(maxval)
        self.maximize = maximize
        self.topology = topology

        # Precompute centroids shared across all islands
        k_cent = jax.random.PRNGKey(centroids_key)
        self.centroids: jnp.ndarray = compute_cvt_centroids(
            num_descriptors=len(self.minval),
            num_init_cvt_samples=self.num_init_cvt_samples,
            num_centroids=self.num_centroids,
            minval=self.minval,
            maxval=self.maxval,
            key=k_cent,
        )

    def init_state(self, key: chex.PRNGKey) -> Any:
        """Initializes 2D multi-island MAP-Elites states across islands."""
        keys = jax.random.split(key, self.num_islands)
        pop_size = getattr(self.engine.engine_params, "pop_size", 20)

        def _init_one(k: chex.PRNGKey) -> MapElitesState[Any, Any]:
            k_pop, k_state = jax.random.split(k)
            pop = self.genome_config.init_population(k_pop, pop_size)
            return self.engine.init_state(k_state, pop, self.centroids)

        return jax.vmap(_init_one)(keys)

    def _island_loop(self, island_state: Any) -> Tuple[Any, Any]:
        """Runs an individual island's MAP-Elites loop for migration_interval steps."""

        def _step(cur_s: Any, _: Any) -> Tuple[Any, Any]:
            next_s, kpi = self.engine.step(cur_s)
            return next_s, kpi

        return jax.lax.scan(_step, island_state, None, length=self.migration_interval)

    def step(self, multi_state: Any) -> Tuple[Any, Any]:
        """Evolves all islands in parallel for migration_interval steps, then migrates elites."""
        # 1. Evolve each island independently via jax.vmap
        evolved_state, history = jax.vmap(self._island_loop)(multi_state)

        # 2. Sample migrants from each island's repertoire
        # Extract rng keys from islands to sample
        first_key = evolved_state.rng_key[0]
        k_mig, k_next_0 = jax.random.split(first_key)
        island_mig_keys = jax.random.split(k_mig, self.num_islands)

        selections = jax.vmap(lambda s, k: s.repertoire.select(k, self.num_migrants))(
            evolved_state, island_mig_keys
        )

        # 3. Permute migrants across island topology
        shift = 1
        rolled_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.roll(x, shift=shift, axis=0), selections.genotypes
        )
        rolled_descriptors = jnp.roll(selections.descriptors, shift=shift, axis=0)
        rolled_fitnesses = jnp.roll(selections.fitnesses, shift=shift, axis=0)

        # 4. Add incoming migrants to each recipient island's repertoire
        new_repertoires = jax.vmap(lambda rep, g, d, f: rep.add(g, d, f))(
            evolved_state.repertoire, rolled_genotypes, rolled_descriptors, rolled_fitnesses
        )

        next_state = evolved_state.replace(repertoire=new_repertoires)
        new_rng_keys = next_state.rng_key.at[0].set(k_next_0)
        next_state = next_state.replace(rng_key=new_rng_keys)

        return next_state, history

    def run(
        self, init_state: Any, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[IslandMapElitesEvolutionState, Any, int]:
        """Executes distributed Island MAP-Elites evolution."""
        start_time = time.perf_counter()

        mig_interval = max(1, self.migration_interval)
        num_outer_steps = max(1, self.generations // mig_interval)

        current_state = init_state
        all_histories: List[Any] = []

        for _ in range(num_outer_steps):
            current_state, history = self.step(current_state)
            all_histories.append(history)

        elapsed = time.perf_counter() - start_time

        # Global best reduction across all islands and all repertoire cells
        # current_state.repertoire.fitnesses has shape (num_islands, num_centroids, 1) or (num_islands, num_centroids)
        rep_fitnesses = current_state.repertoire.fitnesses.squeeze()
        if self.maximize:
            flat_best_idx = jnp.argmax(rep_fitnesses)
        else:
            # Repertoire stores maximization scores, so maximum is highest (best)
            flat_best_idx = jnp.argmax(rep_fitnesses)

        island_idx, centroid_idx = jnp.unravel_index(flat_best_idx, rep_fitnesses.shape)

        best_genome = jax.tree_util.tree_map(
            lambda x: x[island_idx, centroid_idx], current_state.repertoire.genotypes
        )
        best_fitness = rep_fitnesses[island_idx, centroid_idx]
        if not self.maximize:
            best_fitness = -best_fitness

        logger.debug(
            "IslandMapElites completed in %.3fs across %d islands. Global Best Fitness: %s (Island %d, Cell %d)",
            elapsed,
            self.num_islands,
            best_fitness,
            int(island_idx),
            int(centroid_idx),
        )

        final_state = IslandMapElitesEvolutionState(
            multi_state=current_state,
            best_genome=best_genome,
            best_fitness=best_fitness,
            repertoire=current_state.repertoire,
            generation=self.generations,
        )

        return final_state, all_histories, self.generations


__all__ = [
    "MapElitesEngineAdapter",
    "IslandMapElitesAdapter",
    "MapElitesEvolutionState",
    "IslandMapElitesEvolutionState",
]
