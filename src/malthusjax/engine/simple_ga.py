from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.random import create_key, validate_key
from malthusjax.engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
)


@struct.dataclass
class SimpleGAEngineParams(AbstractEngineParams):
    crossover_rate: float = 0.0
    elite_ratio: float = 0.5
    mutation_std: float = 1.0


@struct.dataclass
class SimpleGAEvolutionState(AbstractEvolutionState[BaseGenome, BasePopulation[Any]]):
    pass


@struct.dataclass
class SimpleGAEngine(AbstractEngine[BaseGenome, BasePopulation[Any]]):
    """
    A pure JAX genetic engine that hardcodes EvoSAX's SimpleGA logic,
    bypassing the modular MalthusJAX pipeline to achieve maximum performance.
    """

    genome_config: Any = struct.field(default=None, pytree_node=False)
    evaluator: Any = struct.field(default=None, pytree_node=False)

    def __hash__(self) -> int:
        """Make engine hashable for JIT static_argnums."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other

    def init_state(
        self, rng_key: Union[int, jax.Array], initial_population: Any = None
    ) -> SimpleGAEvolutionState:
        params = self.engine_params

        if isinstance(rng_key, int):
            rng_key = create_key(rng_key, impl=getattr(params, "prng_impl", None))
        else:
            validate_key(rng_key, context="SimpleGAEngine.init_state()")

        init_pop_key, rng_key = jax.random.split(rng_key)

        if initial_population is None:
            initial_population = self.genome_config.init_population(init_pop_key, params.pop_size)

        eval_pop = self.evaluator.evaluate_population(initial_population)
        best_fitness = jnp.min(eval_pop.fitness)
        best_idx = jnp.argmin(eval_pop.fitness)

        best_genome = jax.tree_util.tree_map(lambda x: x[best_idx], eval_pop.genes)

        return SimpleGAEvolutionState(
            population=eval_pop,
            best_genome=best_genome,
            generation=0,
            best_fitness=best_fitness,
            rng_key=rng_key,
        )

    def step(
        self, state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]]
    ) -> Tuple[SimpleGAEvolutionState, AbstractGenerationOutput]:
        params = self.engine_params
        pop = state.population

        # 1. Elitism / Selection
        idx = jnp.argsort(pop.fitness)
        sorted_genes = jax.tree_util.tree_map(lambda x: x[idx], pop.genes)

        num_elites = int(params.pop_size * getattr(params, "elite_ratio", 0.1))
        p = jnp.arange(params.pop_size) < num_elites

        rng_key, k_cross, k_mut, k_1, k_2 = jax.random.split(state.rng_key, 5)
        k_cross_keys = jax.random.split(k_cross, params.pop_size)
        k_mut_keys = jax.random.split(k_mut, params.pop_size)

        parents_1_idx = jax.random.choice(k_1, jnp.arange(params.pop_size), (params.pop_size,), p=p)
        parents_2_idx = jax.random.choice(k_2, jnp.arange(params.pop_size), (params.pop_size,), p=p)

        parents_1 = jax.tree_util.tree_map(lambda x: x[parents_1_idx], sorted_genes)
        parents_2 = jax.tree_util.tree_map(lambda x: x[parents_2_idx], sorted_genes)

        # 2. Crossover & Mutation (Native single-array ops vmap'd over population)
        def recombine_and_mutate(key_c, key_m, p1, p2):
            # crossover mask
            c_mask = jax.random.uniform(key_c, p1.shape) < params.crossover_rate
            offspring = jnp.where(c_mask, p2, p1)
            # mutation
            return offspring + params.mutation_std * jax.random.normal(key_m, p1.shape)

        next_genes = jax.tree_util.tree_map(
            lambda p1_leaf, p2_leaf: jax.vmap(recombine_and_mutate)(
                k_cross_keys, k_mut_keys, p1_leaf, p2_leaf
            ),
            parents_1,
            parents_2,
        )

        # 3. Evaluation
        new_pop = pop.replace(genes=next_genes)  # type: ignore[attr-defined]
        evaluated_pop = self.evaluator.evaluate_population(new_pop)

        # 4. Hall of Fame Update
        gen_best_fitness = jnp.min(evaluated_pop.fitness)
        new_best_fitness = jnp.minimum(gen_best_fitness, state.best_fitness)

        # Using dummy output class as generic metric tracker
        from malthusjax.engine.genetic_fastengine import GeneticGenerationOutput

        metrics = GeneticGenerationOutput(
            best_fitness=new_best_fitness,
            mean_fitness=jnp.mean(evaluated_pop.fitness),
            std_fitness=jnp.std(evaluated_pop.fitness),
            generation=state.generation,
            random_key=state.rng_key,
        )

        new_state = state.replace(  # type: ignore[attr-defined]
            population=evaluated_pop,
            best_fitness=new_best_fitness,
            generation=state.generation + 1,
            rng_key=rng_key,
        )
        return new_state, metrics

    def run(
        self,
        initial_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
        return_history: bool = True,
    ) -> Tuple[
        AbstractEvolutionState[BaseGenome, BasePopulation[Any]], Any, Optional[float]
    ]:
        """Execute full evolution and extract final best genome at the end."""
        final_state, history, elapsed_time = super().run(
            initial_state, time_it=time_it, compile=compile, verbose=verbose, return_history=return_history
        )
        best_idx = jnp.argmin(final_state.population.fitness)
        final_best_genome = jax.tree_util.tree_map(
            lambda x: x[best_idx], final_state.population.genes
        )
        final_state = final_state.replace(best_genome=final_best_genome)  # type: ignore[attr-defined]
        return final_state, history, elapsed_time
