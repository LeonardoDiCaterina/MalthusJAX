"""Level 4 Example: Custom Orchestration via the Composer.

This script demonstrates Level 4: using the Composer API to declaratively
wire up standard operators (selection, crossover, mutation, fitness), then
extending the Engine below the Composer to inject custom metrics into the
lax.scan history — accessible directly from the Engine's run() output.

Key things demonstrated:
  - Use the Composer for operator/fitness wiring (no boilerplate).
  - Subclass GeneticEngine to track a custom population-diversity metric.
  - Register a custom genome and engine via @register_genome / @register_engine.
  - Run multi-seed experiments via composer.quick_run() for standard metrics.
  - Read custom metrics from the raw engine history (DetGenerationOutput).
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from flax import struct

# MalthusJAX core components
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEvolutionState,
    GeneticGenerationOutput,
    GeneticEngineParams,
)

# Composer components
from malthusjax.composer.decorators import register_genome, register_engine
from malthusjax.composer.composer import Composer

print("=========================================================")
print("  Level 4 Example: Custom Orchestration via the Composer ")
print("=========================================================\n")


# ==============================================================================
# 1. Define the Custom Tracking Architecture
# ==============================================================================

# Extend Population to carry a custom metric computed each generation
@struct.dataclass
class DetPopulation(RealPopulation):
    distance_variance: chex.Array = struct.field(default_factory=lambda: jnp.zeros(()))


# Extend Config to instantiate our custom Population
@struct.dataclass
class DetGenomeConfig(RealGenomeConfig):
    def init_population(self, key: chex.PRNGKey, size: int) -> "DetPopulation":
        pop = super().init_population(key, size)
        return DetPopulation(
            genes=pop.genes,
            fitness=pop.fitness,
            config=self,
            distance_variance=jnp.zeros(()),
        )


# Extend the history object so lax.scan tracks it over time
@struct.dataclass
class DetGenerationOutput(GeneticGenerationOutput):
    distance_variance: chex.Array


# Extend the Engine to inject custom JAX math into the step() loop
class DistanceMatrixEngine(GeneticEngine):
    def step(
        self, state: GeneticEvolutionState
    ) -> Tuple[GeneticEvolutionState, DetGenerationOutput]:
        # 1. Run the standard 5-phase loop (Selection, Crossover, Mutation, Merge, Eval)
        final_state, metrics = super().step(state)

        # 2. Compute pairwise distances and extract the variance as a diversity proxy
        genes = final_state.population.genes.values  # (pop_size, dim)
        diffs = genes[:, None, :] - genes[None, :, :]  # (pop_size, pop_size, dim)
        dist_matrix = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # (pop_size, pop_size)
        distance_variance = jnp.var(dist_matrix)

        # 3. Pack the metric into the custom population
        new_pop = DetPopulation(
            genes=final_state.population.genes,
            fitness=final_state.population.fitness,
            config=final_state.population.config,
            distance_variance=distance_variance,
        )
        final_state = final_state.replace(population=new_pop)

        # 4. Pack it into the custom history payload (lax.scan will track it)
        custom_metrics = DetGenerationOutput(
            best_fitness=metrics.best_fitness,
            mean_fitness=metrics.mean_fitness,
            std_fitness=metrics.std_fitness,
            generation=metrics.generation,
            random_key=metrics.random_key,
            distance_variance=distance_variance,
        )
        return final_state, custom_metrics


# ==============================================================================
# 2. Register with the Composer
# ==============================================================================

print("Registering custom genome and engine to the Composer Registry...")


@register_genome("distance_genome")
def build_dist_genome(**kwargs) -> DetGenomeConfig:
    return DetGenomeConfig(
        shape=kwargs.get("shape", (5,)),
        bounds=(-5.12, 5.12),
    )


@register_engine("distance_engine")
def build_dist_engine(evaluator, selection, crossover, mutation, **kwargs):
    from malthusjax.composer.engine_factory import GeneticEngineAdapter
    from malthusjax.composer.genome_catalog import GenomeCatalog

    genome_config = GenomeCatalog().get(kwargs.get("genome_type", "distance_genome"), **kwargs)

    engine = DistanceMatrixEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=GeneticEngineParams(
            pop_size=kwargs.get("pop_size", 32),
            num_generations=kwargs.get("generations", 50),
            elitism=kwargs.get("elitism", 2),
        ),
    )

    return GeneticEngineAdapter(
        genetic_engine=engine,
        genome_config=genome_config,
        prng_impl=None,
        maximize=kwargs.get("maximize", False),
    )


# ==============================================================================
# 3. Multi-seed run via the Composer (standard metrics)
# ==============================================================================

def main():
    composer = Composer.create_default()

    print("\nStarting Composer multi-seed run...")
    result = composer.quick_run(
        experiment_name="determinant_tracking",
        engine_type="distance_engine",   # Matches @register_engine
        genome_type="distance_genome",   # Matches @register_genome
        fitness="sphere:dim=5",
        selection="tournament:tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.2",
        pop_size=32,
        generations=50,
        seeds=[42, 100],                 # Two seeds — Composer handles the loop
    )

    print("\n--- Composer Run Complete (standard metrics across seeds) ---")
    summary = result.aggregated_summary()
    print(f"  Final best fitness:  mean={summary['best_fitness']['mean']:.4f}")
    print(f"  Total evaluations:   {summary['total_evaluations']['mean']:.0f}")

    # ==============================================================================
    # 4. Single direct engine run — to access the custom DetGenerationOutput history
    #
    # The Composer's quick_run() only surfaces standard metrics. For custom
    # lax.scan history (like distance_variance), run the engine directly.
    # ==============================================================================

    print("\n--- Direct engine run to read custom distance_variance history ---")
    from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
    from malthusjax.core.fitness.composable.environments import SphereEnv
    from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
    from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
    from malthusjax.operators.mutation.real import GaussianMutation
    from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
    from malthusjax.operators.selection.tournament import TournamentSelection

    genome_config = DetGenomeConfig(shape=(5,), bounds=(-5.12, 5.12))
    POP_SIZE = 32
    NUM_GENS = 50

    evaluator = OptimizationEvaluator(
        env=SphereEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )
    selection = TournamentSelection(num_selections=POP_SIZE, tournament_size=3)
    crossover = SimulatedBinaryCrossover(crossover_rate=0.9, eta=2.0)
    mutation = GaussianMutation(mutation_rate=0.2, mutation_strength=0.5)

    engine = DistanceMatrixEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=GeneticEngineParams(
            pop_size=POP_SIZE,
            num_generations=NUM_GENS,
            elitism=2,
        ),
    )

    key = jax.random.PRNGKey(42)
    state = engine.init_state(key)
    state = state.replace(population=genome_config.init_population(key, POP_SIZE))

    final_state, history, _ = engine.run(state)

    # history is a DetGenerationOutput with arrays of shape (NUM_GENS,)
    print(f"Final best fitness: {final_state.best_fitness:.4f}")
    print(f"Distance variance (last 5 gens): {history.distance_variance[-5:]}")
    print(f"Final population distance variance: {final_state.population.distance_variance:.4f}")


if __name__ == "__main__":
    main()
