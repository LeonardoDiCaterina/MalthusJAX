"""Level 3 Example: Handing off to the GeneticEngine.

This script demonstrates Level 3: combining Levels 1 & 2 components and
handing them to the GeneticEngine, which eliminates the manual lax.scan
loop, PRNG budgeting, and history tracking boilerplate.

Key things demonstrated:
  - Composable evaluator stack (no monolithic evaluator subclass).
  - Custom Level 2 Operators (Mutation, Selection, Crossover).
  - Plugging everything into GeneticEngine.run() — zero boilerplate loops.
  - Extending the Engine to inject custom metrics (diversity tracking) into
    the lax.scan history payload.
"""

import time
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Level 1 Core
from malthusjax.core.base import BaseGenome, BasePopulation

# Composable evaluator stack (the new paradigm)
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import SphereEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

# Level 2 Operators
from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection

# Level 3 Engine
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
    GeneticGenerationOutput,
)


# ==============================================================================
# 1. Level 1 Components
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


@struct.dataclass
class ContinuousPopulation(BasePopulation[ContinuousGenome]):
    genes: ContinuousGenome
    fitness: chex.Array
    config: ContinuousGenomeConfig = struct.field(pytree_node=False)

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: ContinuousGenomeConfig, size: int
    ) -> "ContinuousPopulation":
        keys = jax.random.split(key, size)

        def init_one(k: chex.PRNGKey) -> ContinuousGenome:
            return ContinuousGenome(
                values=jax.random.uniform(k, shape=config.shape, minval=-5.12, maxval=5.12)
            )

        return cls(
            genes=jax.vmap(init_one)(keys),
            fitness=jnp.full((size,), jnp.inf),
            config=config,
        )


# Composable evaluator — no subclassing required
evaluator = OptimizationEvaluator(
    env=SphereEnv(),
    transform=IdentityTransform(),
    interpreter=IdentityInterpreter(),
    output=ScalarOutput(maximize=False),
)


# ==============================================================================
# 2. Level 2 Operators
# ==============================================================================

@struct.dataclass
class ContinuousGaussianMutation(BaseMutation[ContinuousGenome, ContinuousGenomeConfig]):
    mutation_rate: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array]:
        noise = jax.random.normal(keys[0], shape=config.shape)
        return (noise,)

    def _mutate_one(
        self,
        genome: ContinuousGenome,
        noise_data: Tuple[chex.Array],
        config: ContinuousGenomeConfig,
        **kwargs: Any,
    ) -> ContinuousGenome:
        (noise,) = noise_data
        return genome.replace(values=genome.values + noise * self.mutation_rate)


@struct.dataclass
class TournamentSelection(BaseSelection[ContinuousGenome, ContinuousGenomeConfig]):
    tournament_size: int = struct.field(pytree_node=False, default=3)
    maximize: bool = struct.field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Any = None, **kwargs: Any
    ) -> chex.Array:
        participants = jax.random.randint(
            keys[0],
            shape=(self.num_selections, self.tournament_size),
            minval=0,
            maxval=fitness.shape[0],
        )
        participant_fitness = fitness[participants]
        winner_idx = (
            jnp.argmax(participant_fitness, axis=1)
            if self.maximize
            else jnp.argmin(participant_fitness, axis=1)
        )
        return participants[jnp.arange(self.num_selections), winner_idx]


@struct.dataclass
class ContinuousNoOpCrossover(BaseCrossover[ContinuousGenome, ContinuousGenomeConfig]):
    num_offspring: int = struct.field(pytree_node=False, default=1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def _generate_noise(
        self, keys: chex.Array, config: ContinuousGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array]:
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

    config = ContinuousGenomeConfig(shape=(10,))
    selection = TournamentSelection(num_selections=POP_SIZE, tournament_size=3, maximize=False)
    crossover = ContinuousNoOpCrossover()

    # --- Engine 1: Standard — no boilerplate loop, no manual PRNG budgeting ---
    print("--- Engine 1: Standard GeneticEngine ---")
    engine_1 = GeneticEngine(
        genome_config=config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=ContinuousGaussianMutation(mutation_rate=0.05),
        engine_params=GeneticEngineParams(
            pop_size=POP_SIZE,
            num_generations=NUM_GENERATIONS,
            elitism=0,
        ),
    )

    state_1 = engine_1.init_state(jax.random.PRNGKey(42))

    t0 = time.time()
    final_state_1, history_1, _ = engine_1.run(state_1)
    t1 = time.time()

    print(f"Engine 1 completed in {t1 - t0:.4f}s")
    print(f"Best fitness (last 5 gens): {history_1.best_fitness[-5:]}")
    print(f"Final best fitness: {final_state_1.best_fitness}\n")

    # --- Engine 2: Custom — extend Engine to inject diversity metric into history ---
    print("--- Engine 2: Custom Engine with diversity tracking ---")

    # Extend the population to carry a custom metric
    @struct.dataclass
    class CustomPopulation(ContinuousPopulation):
        diversity_metric: chex.Array = struct.field(default_factory=lambda: jnp.zeros(()))

    # Extend the config to instantiate our custom population
    @struct.dataclass
    class CustomGenomeConfig(ContinuousGenomeConfig):
        def init_population(self, key: chex.PRNGKey, size: int) -> "CustomPopulation":
            pop = super().init_population(key, size)
            return CustomPopulation(
                genes=pop.genes,
                fitness=pop.fitness,
                config=self,
                diversity_metric=jnp.zeros(()),
            )

    # Extend the history KPI object so lax.scan tracks diversity over time
    @struct.dataclass
    class CustomGenerationOutput(GeneticGenerationOutput):
        population_diversity: chex.Array

    # Extend the Engine: inject custom math after the standard 5-phase step
    class DiversityTrackingEngine(GeneticEngine):
        def step(
            self, state: GeneticEvolutionState
        ) -> Tuple[GeneticEvolutionState, CustomGenerationOutput]:
            final_state, metrics = super().step(state)

            # Compute population diversity (std-dev of all gene values)
            diversity = jnp.std(final_state.population.genes.values)

            new_pop = CustomPopulation(
                genes=final_state.population.genes,
                fitness=final_state.population.fitness,
                config=final_state.population.config,
                diversity_metric=diversity,
            )
            final_state = final_state.replace(population=new_pop)

            custom_metrics = CustomGenerationOutput(
                best_fitness=metrics.best_fitness,
                mean_fitness=metrics.mean_fitness,
                std_fitness=metrics.std_fitness,
                generation=metrics.generation,
                random_key=metrics.random_key,
                population_diversity=diversity,
            )
            return final_state, custom_metrics

    custom_config = CustomGenomeConfig(shape=(10,))
    engine_2 = DiversityTrackingEngine(
        genome_config=custom_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=ContinuousGaussianMutation(mutation_rate=0.5),
        engine_params=GeneticEngineParams(
            pop_size=POP_SIZE,
            num_generations=NUM_GENERATIONS,
            elitism=5,
        ),
    )

    state_2 = engine_2.init_state(jax.random.PRNGKey(77))
    # Seed the state with our CustomPopulation so lax.scan types match
    state_2 = state_2.replace(
        population=custom_config.init_population(jax.random.PRNGKey(77), POP_SIZE)
    )

    final_state_2, history_2, _ = engine_2.run(state_2)

    print("Engine 2 completed!")
    print(f"Diversity over time (last 5): {history_2.population_diversity[-5:]}")
    print(f"Final population diversity: {final_state_2.population.diversity_metric:.4f}")


if __name__ == "__main__":
    run_level_3_evolution()
