"""Multi-Objective Native Engine."""

from typing import Any, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.mo.population import MOPopulation
from malthusjax.engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
)
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

G = TypeVar("G", bound=BaseGenome)
P = TypeVar("P", bound=MOPopulation)

_field: Any = struct.field


@struct.dataclass
class MOEngineParams(AbstractEngineParams):
    """
    Configuration for the Multi-Objective engine.
    """

    key_derivation: str = "fold_in"  # "fold_in" or "split"


@struct.dataclass
class MOState(AbstractEvolutionState[G, P]):
    """
    Mutable state container for Multi-Objective Evolution.
    """

    emitter_state: Optional[EmitterState]


@struct.dataclass
class MOGenerationOutput(AbstractGenerationOutput):
    """KPI payload returned at every MO evolution step."""

    num_pareto_optimal: chex.Numeric
    max_crowding_distance: chex.Numeric


@struct.dataclass
class MOEngine(AbstractEngine[G, P]):
    r"""
    Native MalthusJAX Multi-Objective Engine (NSGA-II paradigm).

    Orchestrates the interplay between the BaseEmitter and the smart MOPopulation
    which natively handles $\mu+\lambda$ non-dominated sorting elitism.
    """

    emitter: BaseEmitter = _field(pytree_node=False)
    evaluator: BaseMOEvaluator[Any, Any, Any] = _field(pytree_node=False)
    engine_params: MOEngineParams = _field(pytree_node=False)

    def init_state(  # type: ignore[override]
        self, rng_key: chex.Array, initial_population: BasePopulation[G]
    ) -> MOState[G, P]:
        """
        Initializes the MO state.

        Args:
            rng_key: Master PRNG key.
            initial_population: A population already initialized with starting genotypes.
        """
        k1, k2 = jax.random.split(rng_key)

        # 1. Evaluate the initial population
        # BaseMOEvaluator natively upgrades it to a sorted MOPopulation
        mo_pop = self.evaluator.evaluate_population(initial_population)

        # 2. Initialize the Emitter state
        # The emitter is passed the mo_pop, allowing it to perform tournament selection on ranks
        emitter_state = self.emitter.init(k2, mo_pop)

        # Extract best fitness (For MO, we just grab the first individual's first objective as a dummy best,
        # since "best" is ambiguous in pareto fronts. Alternatively, we could pick the one with highest crowding distance in rank 0)
        best_fitness = mo_pop.fitness[0, 0]
        best_genome = jax.tree_util.tree_map(lambda x: x[0], mo_pop.genes)

        return MOState(
            population=cast(P, mo_pop),
            best_genome=cast(G, best_genome),
            generation=0,
            best_fitness=best_fitness,
            rng_key=k1,
            emitter_state=emitter_state,
        )

    def step(self, state: MOState[G, P]) -> Tuple[MOState[G, P], MOGenerationOutput]:  # type: ignore[override]
        """
        Performs a single generation of MO Evolution (NSGA-II).
        """
        # --- Centralized RNG Management ---
        emitter_keys = self.emitter.num_keys()
        total_rng_budget = emitter_keys + 3  # ask(emitter_keys), eval(1), tell(1), next(1)

        if self.engine_params.key_derivation == "fold_in":
            indices = jnp.arange(total_rng_budget)
            all_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(state.rng_key, indices)
        else:
            all_keys = jax.random.split(state.rng_key, total_rng_budget)

        k_ask = all_keys[:emitter_keys]
        k_tell = all_keys[emitter_keys + 1]
        k_next = all_keys[emitter_keys + 2]
        # ----------------------------------

        # 1. Ask the emitter for a new batch of offspring
        # Emitters designed for MO can natively call `state.population.select(key, batch)` to get parents.
        offspring_pop, new_emitter_state = self.emitter.ask(
            state.emitter_state,
            state.population,
            k_ask,
            generation=state.generation,
            params=self.engine_params,
        )

        # 2. Evaluate the offspring
        # Returns an MOPopulation
        eval_pop = self.evaluator.evaluate_population(offspring_pop)

        # 3. Survival Mechanism (NSGA-II $\mu+\lambda$ Elitism)
        # The MOPopulation natively handles merging and truncating via non-dominated sorting!
        new_mo_pop = state.population.merge(eval_pop).truncate(self.engine_params.pop_size)

        # 4. Tell the emitter the results (optional for basic mutation, useful for CMA-ES etc)
        new_emitter_state = self.emitter.tell(
            new_emitter_state,
            state.population,  # old repertoire
            eval_pop,  # offspring
            eval_pop.fitness,
            None,
            k_tell,
        )

        # 5. Extract KPI metrics
        # Track how many individuals are in the first Pareto front (rank 0)
        num_pareto_optimal = jnp.sum(new_mo_pop.pareto_rank == 0)
        max_crowding_distance = jnp.max(new_mo_pop.crowding_distance)

        best_fitness = new_mo_pop.fitness[0, 0]
        best_genome = jax.tree_util.tree_map(lambda x: x[0], new_mo_pop.genes)

        kpi = MOGenerationOutput(
            best_fitness=best_fitness,
            mean_fitness=jnp.mean(new_mo_pop.fitness[:, 0]),
            std_fitness=jnp.std(new_mo_pop.fitness[:, 0]),
            generation=state.generation + 1,
            num_pareto_optimal=num_pareto_optimal,
            max_crowding_distance=max_crowding_distance,
        )

        new_state = state.replace(  # type: ignore[attr-defined]
            population=cast(P, new_mo_pop),
            best_genome=cast(G, best_genome),
            generation=state.generation + 1,
            best_fitness=best_fitness,
            rng_key=k_next,
            emitter_state=new_emitter_state,
        )

        return new_state, kpi
