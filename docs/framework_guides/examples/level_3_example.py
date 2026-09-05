import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Level 1 Core
from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator

# Level 2 Operators
from malthusjax.operators.base import BaseMutation, BaseSelection

# Level 3 Engines
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams

# ==============================================================================
# 1. Level 1 & 2 Components (From previous examples)
# ==============================================================================
@struct.dataclass
class ContinuousGenomeConfig:
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    @property
    def dtype(self) -> Any:
        return jnp.float32

    def init_population(self, key: chex.PRNGKey, size: int) -> "ContinuousPopulation":
        return ContinuousPopulation.init_random(key, self, size)

@struct.dataclass
class ContinuousGenome(BaseGenome):
    values: chex.Array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig) -> "ContinuousGenome":
        values = jax.random.uniform(key, shape=config.shape, minval=-5.12, maxval=5.12)
        return cls(values=values)

@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    genes: ContinuousGenome
    fitness: chex.Array
    config: ContinuousGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: ContinuousGenomeConfig, size: int) -> "ContinuousPopulation":
        keys = jax.random.split(key, size)
        batched_genes = jax.vmap(ContinuousGenome.random_init, in_axes=(0, None))(keys, config)
        return cls(genes=batched_genes, fitness=jnp.full((size,), jnp.inf), config=config)

@struct.dataclass
class SphereEvaluatorConfig:
    maximize: bool = struct.field(pytree_node=False, default=False)

@struct.dataclass
class SphereEvaluator(BaseEvaluator[ContinuousGenome, ContinuousGenomeConfig, SphereEvaluatorConfig]):
    def evaluate(self, genome: ContinuousGenome, **kwargs: Any) -> chex.Array:
        return jnp.sum(genome.values ** 2)

@struct.dataclass
class ContinuousGaussianMutation(BaseMutation[ContinuousGenome, ContinuousGenomeConfig]):
    mutation_rate: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0) -> Tuple[chex.Array]:
        noise = jax.random.normal(keys[0], shape=config.shape)
        return (noise,)

    def _mutate_one(self, genome: ContinuousGenome, noise_data: Tuple[chex.Array], config: ContinuousGenomeConfig, **kwargs: Any) -> ContinuousGenome:
        noise = noise_data[0]
        new_values = genome.values + (noise * self.mutation_rate)
        return genome.replace(values=new_values)

# We need a Level 2 Selection operator for the Engine!
@struct.dataclass
class TournamentSelection(BaseSelection[ContinuousGenome, ContinuousGenomeConfig]):
    tournament_size: int = struct.field(pytree_node=False, default=3)
    maximize: bool = struct.field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config: Any = None, **kwargs: Any) -> chex.Array:
        # We need num_selections tournaments of size tournament_size
        participants = jax.random.randint(keys[0], shape=(self.num_selections, self.tournament_size), minval=0, maxval=fitness.shape[0])

        # Get fitness of all participants: shape (num_selections, tournament_size)
        participant_fitness = fitness[participants]

        # Select best from each tournament
        if self.maximize:
            winner_indices = jnp.argmax(participant_fitness, axis=1)
        else:
            winner_indices = jnp.argmin(participant_fitness, axis=1)

        # Get the actual population indices of the winners
        # Use advanced indexing to pluck the winner index for each tournament
        row_indices = jnp.arange(self.num_selections)
        return participants[row_indices, winner_indices]


from malthusjax.operators.base import BaseCrossover

@struct.dataclass
class ContinuousNoOpCrossover(BaseCrossover[ContinuousGenome, ContinuousGenomeConfig]):
    num_offspring: int = struct.field(pytree_node=False, default=1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def _generate_noise(self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0) -> Tuple[chex.Array]:
        return (jnp.zeros(0),)

    def _recombine_one(
        self,
        p1: ContinuousGenome,
        p2: ContinuousGenome,
        noise_data: Tuple[chex.Array],
        config: ContinuousGenomeConfig,
        **kwargs: Any,
    ) -> ContinuousGenome:
        return p1

# ==============================================================================
# 3. Level 3: The Engine Orchestrator
# ==============================================================================
def run_level_3_evolution():
    POP_SIZE = 128
    NUM_GENERATIONS = 100

    # 1. Initialize our components
    config = ContinuousGenomeConfig(shape=(10,))
    evaluator = SphereEvaluator(config=SphereEvaluatorConfig(maximize=False), data=None)
    selection = TournamentSelection(num_selections=POP_SIZE, tournament_size=3, maximize=False)
    crossover = ContinuousNoOpCrossover()

    print("--- Engine 1: Small Mutation, No Elitism ---")
    mutator_1 = ContinuousGaussianMutation(mutation_rate=0.05)

    engine_1 = GeneticEngine(
        genome_config=config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutator_1,
        engine_params=GeneticEngineParams(
            pop_size=POP_SIZE,
            num_generations=NUM_GENERATIONS,
            elitism=0 # No elites
        )
    )

    # Look how simple the execution loop is now!
    master_key = jax.random.PRNGKey(42)
    state_1 = engine_1.init_state(master_key) # ResourceMapper calculates budget here

    t0 = time.time()
    final_state_1, history_1, _ = engine_1.run(state_1)
    t1 = time.time()

    print(f"Engine 1 completed in {t1-t0:.4f} seconds!")
    print(f"Best fitness over time: {history_1.best_fitness[-5:]}")
    print(f"Final best fitness: {final_state_1.best_fitness}")

    print("\n--- Engine 3: Custom Population & History Tracking ---")
    from malthusjax.engine.genetic_fastengine import GeneticGenerationOutput, GeneticEvolutionState

    # 1. Extend the Population to hold a custom metric
    @struct.dataclass
    class CustomPopulation(ContinuousPopulation):
        diversity_metric: chex.Array = struct.field(default_factory=lambda: jnp.zeros(()))

    @struct.dataclass
    class CustomGenomeConfig(ContinuousGenomeConfig):
        def init_population(self, key: chex.PRNGKey, size: int) -> "CustomPopulation":
            pop = super().init_population(key, size)
            return CustomPopulation(
                genes=pop.genes,
                fitness=pop.fitness,
                config=self,
                diversity_metric=jnp.zeros(())
            )

    # 2. Extend the History KPI object
    @struct.dataclass
    class CustomGenerationOutput(GeneticGenerationOutput):
        population_diversity: chex.Array

    # 3. Extend the Engine to compute the metric
    class CustomGeneticEngine(GeneticEngine):
        def step(self, state: GeneticEvolutionState) -> Tuple[GeneticEvolutionState, CustomGenerationOutput]:
            # Run the standard 5-phase loop
            final_state, metrics = super().step(state)

            # Compute our custom metric (e.g. population-wide standard deviation)
            diversity = jnp.std(final_state.population.genes.values)

            # Update the population
            new_pop = CustomPopulation(
                genes=final_state.population.genes,
                fitness=final_state.population.fitness,
                config=final_state.population.config,
                diversity_metric=diversity
            )
            final_state = final_state.replace(population=new_pop)

            # Return our extended KPI object so lax.scan tracks it!
            custom_metrics = CustomGenerationOutput(
                best_fitness=metrics.best_fitness,
                mean_fitness=metrics.mean_fitness,
                std_fitness=metrics.std_fitness,
                generation=metrics.generation,
                random_key=metrics.random_key,
                population_diversity=diversity
            )
            return final_state, custom_metrics

    custom_config = CustomGenomeConfig(shape=(10,))
    mutator_2 = ContinuousGaussianMutation(mutation_rate=0.5)
    engine_3 = CustomGeneticEngine(
        genome_config=custom_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutator_2,
        engine_params=GeneticEngineParams(
            pop_size=POP_SIZE,
            num_generations=NUM_GENERATIONS,
            elitism=5
        )
    )

    state_3 = engine_3.init_state(jax.random.PRNGKey(77))

    # We must overwrite the starting population to be a CustomPopulation so the types match in lax.scan
    initial_pop = custom_config.init_population(jax.random.PRNGKey(77), POP_SIZE)
    state_3 = state_3.replace(population=initial_pop)

    final_state_3, history_3, _ = engine_3.run(state_3)

    print("Engine 3 completed!")
    print(f"Custom History tracking! Diversity over time: {history_3.population_diversity[-5:]}")
    print(f"Final Population's Diversity Metric: {final_state_3.population.diversity_metric}")

if __name__ == "__main__":
    run_level_3_evolution()
