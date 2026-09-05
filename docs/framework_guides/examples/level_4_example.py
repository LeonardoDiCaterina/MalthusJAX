import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Any
from flax import struct

# MalthusJAX core components
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEvolutionState, GeneticGenerationOutput

# Composer components
from malthusjax.composer.decorators import register_genome, register_engine
from malthusjax.composer.composer import Composer

print("=========================================================")
print("  Level 4 Example: Custom Orchestration via the Composer ")
print("=========================================================\n")

# ==============================================================================
# 1. Define the Custom Tracking Architecture
# ==============================================================================

# We subclass the Population to hold our custom metric
@struct.dataclass
class DetPopulation(RealPopulation):
    distance_det: chex.Array = struct.field(default_factory=lambda: jnp.zeros(()))

# We subclass the Config to instantiate our custom Population
@struct.dataclass
class DetGenomeConfig(RealGenomeConfig):
    def init_population(self, key: chex.PRNGKey, size: int) -> "DetPopulation":
        pop = super().init_population(key, size)
        return DetPopulation(
            genes=pop.genes,
            fitness=pop.fitness,
            config=self,
            distance_det=jnp.zeros(())
        )

# We subclass the History object so `jax.lax.scan` knows to track it over time
@struct.dataclass
class DetGenerationOutput(GeneticGenerationOutput):
    distance_det: chex.Array

# We subclass the Engine to inject the exact JAX math into the step() loop
class DistanceMatrixEngine(GeneticEngine):
    def step(self, state: GeneticEvolutionState) -> Tuple[GeneticEvolutionState, DetGenerationOutput]:
        # 1. Let MalthusJAX do the heavy lifting (Selection, Crossover, Mutation, Eval)
        final_state, metrics = super().step(state)

        # 2. Extract the new population's genomes.
        # For a ContinuousGenome, `values` is shape (POP_SIZE, DIM)
        genes = final_state.population.genes.values

        # 3. Calculate pairwise Euclidean distance matrix
        diffs = genes[:, None, :] - genes[None, :, :]  # shape (POP_SIZE, POP_SIZE, DIM)
        dist_matrix = jnp.sqrt(jnp.sum(diffs**2, axis=-1)) # shape (POP_SIZE, POP_SIZE)

        # 4. Calculate the determinant! (Note: a POP_SIZExPOP_SIZE matrix det might under/overflow easily if POP_SIZE is large,
        # but for this example we'll just track the log-determinant or pseudo-det, or just use a small population).
        # Let's just track the variance of the distances instead to ensure numerical stability in this example:
        variance_of_distances = jnp.var(dist_matrix)

        # 5. Pack it into the Custom Population
        new_pop = DetPopulation(
            genes=final_state.population.genes,
            fitness=final_state.population.fitness,
            config=final_state.population.config,
            distance_det=variance_of_distances
        )
        final_state = final_state.replace(population=new_pop)

        # 6. Pack it into the Custom History Payload
        custom_metrics = DetGenerationOutput(
            best_fitness=metrics.best_fitness,
            mean_fitness=metrics.mean_fitness,
            std_fitness=metrics.std_fitness,
            generation=metrics.generation,
            random_key=metrics.random_key,
            distance_det=variance_of_distances
        )

        return final_state, custom_metrics

# ==============================================================================
# 2. Register with the Composer
# ==============================================================================

print("Registering custom genome and engine to the Composer Registry...")

@register_genome("distance_genome")
def build_dist_genome(**kwargs) -> DetGenomeConfig:
    shape = kwargs.get("shape", (5,)) # Small 5-dimensional problem
    return DetGenomeConfig(shape=shape)

@register_engine("distance_engine")
def build_dist_engine(evaluator, selection, crossover, mutation, **kwargs):
    from malthusjax.engine.genetic_fastengine import GeneticEngineParams
    from malthusjax.composer.engine_factory import GeneticEngineAdapter
    from malthusjax.composer.genome_catalog import GenomeCatalog

    engine_params = GeneticEngineParams(
        pop_size=kwargs.get("pop_size", 32),
        num_generations=kwargs.get("generations", 50),
        elitism=kwargs.get("elitism", 2),
    )

    genome_type = kwargs.get("genome_type", "distance_genome")
    genome_config = GenomeCatalog().get(genome_type, **kwargs)

    engine = DistanceMatrixEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )

    return GeneticEngineAdapter(
        genetic_engine=engine,
        genome_config=genome_config,
        prng_impl=kwargs.get("prng_impl", None),
        maximize=kwargs.get("maximize", False),
        history_metrics=kwargs.get("history_metrics", None)
    )

# ==============================================================================
# 3. Execute declarative run via Composer API
# ==============================================================================

def main():
    composer = Composer.create_default()

    # We pass the configuration as a Python Dictionary (no TOML needed!)
    print("\nStarting Composer Quick Run...")
    result = composer.quick_run(
        experiment_name="determinant_tracking",
        engine_type="distance_engine",         # Matches @register_engine
        genome_type="distance_genome",         # Matches @register_genome
        fitness="sphere:dim=5",
        selection="tournament:tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.2",
        pop_size=32,
        generations=50,
        history_metrics=["best_fitness", "mean_fitness", "std_fitness", "distance_det"],
        seeds=[42, 100],  # Run on two seeds automatically!
    )

    print("\n--- Composer Run Complete! ---")
    print(f"Aggregated Summary across seeds: {result.aggregated_summary()}")

    # Look into the raw history payload from seed 0 (the first run)
    history_seed_0 = result.runs[0].history
    distance_det_history = [h["distance_det"] for h in history_seed_0]

    # Our custom metric was seamlessly tracked!
    print(f"\nLast 5 Generations of custom Distance Variance (Seed 0):")
    print(distance_det_history[-5:])

if __name__ == "__main__":
    main()
