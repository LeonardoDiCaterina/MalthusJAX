from typing import Any, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
)
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

# Import QDAX repertoire for internal Map representation
try:
    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
except ImportError:
    MapElitesRepertoire = Any

G = TypeVar("G", bound=BaseGenome)
P = TypeVar("P", bound=BasePopulation[Any])

_field: Any = struct.field


@struct.dataclass
class MapElitesEngineParams(AbstractEngineParams):
    """
    Configuration for the Native MAP-Elites engine.
    Inherits pop_size (batch_size) and num_generations.
    """

    key_derivation: str = "fold_in"  # "fold_in" or "split"
    maximize: bool = False


@struct.dataclass
class MapElitesState(AbstractEvolutionState[G, P]):
    """
    Mutable state container for Native QD Evolution across generations.
    """

    repertoire: Any  # MapElitesRepertoire
    emitter_state: Optional[EmitterState]


@struct.dataclass
class QDGenerationOutput(AbstractGenerationOutput):
    """KPI payload returned at every QD evolution step."""

    qd_score: chex.Array
    coverage: chex.Array


@struct.dataclass
class MapElitesEngine(AbstractEngine[G, P]):
    """
    Native MalthusJAX MAP-Elites Engine.
    Orchestrates the interplay between the BaseEmitter, BaseQDEvaluator, and the QDAX Repertoire grid.
    """

    emitter: BaseEmitter = _field(pytree_node=False)
    evaluator: BaseQDEvaluator[Any, Any, Any] = _field(pytree_node=False)
    engine_params: MapElitesEngineParams = _field(pytree_node=False)

    def init_state(  # type: ignore[override]
        self, rng_key: chex.Array, initial_population: P, centroids: chex.Array
    ) -> MapElitesState[G, P]:
        """
        Initializes the MapElites state.

        Args:
            rng_key: Master PRNG key.
            initial_population: A population already initialized with starting genotypes.
        """
        k1, k2 = jax.random.split(rng_key)

        # 1. Evaluate the initial population to get fitness and descriptors
        eval_pop = self.evaluator.evaluate_population(initial_population)

        # In MalthusJAX, minimization tasks return lower raw fitnesses (lower is better).
        # However, MapElitesRepertoire ALWAYS maximizes. Therefore, we must flip the sign
        # of the fitness if we are minimizing.
        repertoire_fitnesses = (
            eval_pop.fitness if self.engine_params.maximize else -eval_pop.fitness
        )

        # 2. Initialize the QDAX MapElitesRepertoire
        # Notice that MapElitesRepertoire expects genotypes as a PyTree
        # We pass the underlying genome values instead of the wrapper for cleaner JAX tree structures.
        repertoire = MapElitesRepertoire.init(
            genotypes=getattr(eval_pop.genes, "values", eval_pop.genes),
            fitnesses=repertoire_fitnesses,
            descriptors=eval_pop.info["descriptors"],
            centroids=centroids,
        )

        # 3. Initialize the Emitter state
        emitter_state = self.emitter.init(k2, eval_pop)

        # Extract best fitness from the repertoire. Since we flipped the sign if minimizing,
        # we flip it back to log the true raw fitness correctly.
        best_fitness = jnp.max(repertoire.fitnesses)
        if not self.engine_params.maximize:
            best_fitness = -best_fitness

        best_genome_idx = jnp.argmax(repertoire.fitnesses)
        best_genome_values = jax.tree_util.tree_map(
            lambda x: x[best_genome_idx], getattr(eval_pop.genes, "values", eval_pop.genes)
        )
        if hasattr(eval_pop.genes, "replace"):
            best_genome = eval_pop.genes.replace(values=best_genome_values)
        else:
            best_genome = best_genome_values

        return MapElitesState(
            population=cast(P, eval_pop),
            best_genome=best_genome,
            generation=0,
            best_fitness=best_fitness,
            rng_key=k1,
            repertoire=repertoire,
            emitter_state=emitter_state,
        )

    def step(  # type: ignore[override]
        self, state: MapElitesState[G, P]
    ) -> Tuple[MapElitesState[G, P], QDGenerationOutput]:
        """
        Performs a single generation of MAP-Elites.
        """
        # --- Centralized RNG Management ---
        emitter_keys = self.emitter.num_keys()

        if self.engine_params.key_derivation == "qdax_replica":
            # Replicate QDAX's exact nested key-splitting chain:
            # 1. adapter_step:  randkey, subkey1 = split(randkey)
            # 2. update():      key2, subkey2 = split(subkey1)   → subkey2 goes to ask()
            # 3. ask():         key3, subkey3 = split(subkey2)   → subkey3 goes to emitter.emit()
            # 4. update() cont: key4, subkey4 = split(key2)      → subkey4 goes to scoring
            k_next, subkey1 = jax.random.split(state.rng_key)
            key2, subkey2 = jax.random.split(subkey1)
            _key3, subkey3 = jax.random.split(subkey2)
            _key4, k_eval = jax.random.split(key2)

            # The emitter gets a single-element array containing the exact key
            # that QDAX's MixingEmitter.emit() would receive.
            k_ask = jnp.expand_dims(subkey3, axis=0)
            k_tell = k_eval  # Not used by stateless emitters
        else:
            total_rng_budget = emitter_keys + 3  # ask(emitter_keys), eval(1), tell(1), next(1)

            if self.engine_params.key_derivation == "fold_in":
                indices = jnp.arange(total_rng_budget)
                all_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(state.rng_key, indices)
            else:
                all_keys = jax.random.split(state.rng_key, total_rng_budget)

            k_ask = all_keys[:emitter_keys]
            k_eval = all_keys[emitter_keys]
            k_tell = all_keys[emitter_keys + 1]
            k_next = all_keys[emitter_keys + 2]
        # ----------------------------------

        # 1. Ask the emitter for a new batch of offspring
        offspring_pop, new_emitter_state = self.emitter.ask(
            state.emitter_state,
            state.repertoire,
            k_ask,
            generation=state.generation,
            params=self.engine_params,
        )

        # 2. Evaluate the offspring
        eval_pop = self.evaluator.evaluate_population(offspring_pop)

        # 3. Add to the repertoire (returns a new updated repertoire)
        repertoire_fitnesses = (
            eval_pop.fitness if self.engine_params.maximize else -eval_pop.fitness
        )
        new_repertoire = state.repertoire.add(
            eval_pop.genes.values, eval_pop.info["descriptors"], repertoire_fitnesses
        )

        # 4. Tell the emitter the results
        new_emitter_state = self.emitter.tell(
            new_emitter_state,
            new_repertoire,
            eval_pop,
            eval_pop.fitness,
            eval_pop.info["descriptors"],
            k_tell,
        )

        # 5. Extract KPI metrics
        if self.engine_params.track_metrics:
            # The repertoire fitnesses are stored maximized. So if minimize, they are -raw_fitness.
            # We extract best_fitness from the repertoire and flip it back if necessary.
            best_fitness = jnp.max(new_repertoire.fitnesses)
            if not self.engine_params.maximize:
                best_fitness = -best_fitness

            mean_fitness = jnp.mean(
                jnp.where(new_repertoire.fitnesses > -jnp.inf, new_repertoire.fitnesses, 0.0)
            )
            if not self.engine_params.maximize:
                mean_fitness = -mean_fitness

            std_fitness = jnp.std(
                jnp.where(new_repertoire.fitnesses > -jnp.inf, new_repertoire.fitnesses, 0.0)
            )
            # std is strictly positive, no need to negate

            coverage = (
                100
                * jnp.sum(new_repertoire.fitnesses > -jnp.inf)
                / new_repertoire.centroids.shape[0]
            )

            qd_score = jnp.sum(
                new_repertoire.fitnesses, where=(new_repertoire.fitnesses > -jnp.inf)
            )
            if not self.engine_params.maximize:
                # If minimizing, raw fitnesses are negative.
                qd_score = -qd_score

            jnp.sum(new_repertoire.fitnesses > -jnp.inf)
        else:
            best_fitness = cast(chex.Array, jnp.nan)
            mean_fitness = cast(chex.Array, jnp.nan)
            std_fitness = cast(chex.Array, jnp.nan)
            coverage = cast(chex.Array, jnp.nan)
            qd_score = cast(chex.Array, jnp.nan)

        best_genome_idx = jnp.argmax(new_repertoire.fitnesses)
        best_genome_values = jax.tree_util.tree_map(
            lambda x: x[best_genome_idx], new_repertoire.genotypes
        )
        best_genome = state.best_genome.replace(values=best_genome_values) if hasattr(state.best_genome, "replace") else best_genome_values

        kpi = QDGenerationOutput(
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            std_fitness=std_fitness,
            generation=state.generation + 1,
            qd_score=qd_score,
            coverage=coverage,
        )

        new_state = state.replace(  # type: ignore[attr-defined]
            population=eval_pop,
            best_genome=best_genome,
            generation=state.generation + 1,
            best_fitness=best_fitness,
            rng_key=k_next,
            repertoire=new_repertoire,
            emitter_state=new_emitter_state,
        )

        return new_state, kpi
