"""LSP CGP Adapter.

Provides a factory for creating a MalthusJAX GeneticEngine pipeline
configured with Cartesian GP operators and the CartesianGPEvaluator.
"""

from typing import Any, Optional, Tuple

import chex
import jax.numpy as jnp
from flax import struct
from lsp.operators.cartesian import CartesianMicroMutation, NoOpCrossover

from malthusjax.core.genome.cartesian_genome import CartesianGenomeConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@struct.dataclass
class CGPSelection(ElitePoolSelection):
    """Custom selection for CGP to enable neutral drift.

    Standard ElitePoolSelection uses jnp.argsort(fitness). When a child has the
    same fitness as the parent (a neutral mutation), argsort often favors the parent
    (index 0). This prevents neutral drift.

    This subclass subtracts a tiny epsilon from higher indices to ensure that children
    win ties against the parent.
    """

    def __call__(
        self,
        keys: chex.Array,
        population: Any,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        # population argument could be a Population object OR just the fitness array
        fitness = getattr(population, "fitness", population)

        # Subtly favor higher indices (children) in ties to enable neutral drift
        pop_size = fitness.shape[0]
        tie_breaker = jnp.arange(pop_size) * 1e-9
        modified_fitness = fitness - tie_breaker

        # Pass the modified fitness array directly to super
        return super().__call__(keys, modified_fitness, config, **kwargs)


def build_cgp_engine(
    config: CartesianGenomeConfig,
    evaluator: Any,
    pop_size: int = 5,
    elitism: int = 1,
    mutation_rate: float = 0.05,
    num_generations: int = 10000,
) -> GeneticEngine:
    """Builds a GeneticEngine configured for Cartesian Genetic Programming.

    Constructs a 1 + λ Evolutionary Strategy engine (using ElitePoolSelection
    with elite_k=1, NoOpCrossover, and CartesianMicroMutation) ready to execute
    on `CartesianGenome`s.

    Args:
        config: Genome configuration.
        evaluator: The instantiated CartesianGPEvaluator.
        pop_size: Total population size (1 + λ). Default 5 implies λ=4.
        elitism: Number of elites to preserve (should always be 1 for CGP).
        mutation_rate: Probability of mutating each individual gene.
        num_generations: Number of generations to run. CGP typically needs many
                         (e.g., 10,000 to 100,000) due to point mutation.

    Returns:
        A compiled MalthusJAX GeneticEngine.
    """
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=elitism,
        num_generations=num_generations,
    )

    # In a 1 + λ ES, we preserve the 1 elite, and generate λ offspring.
    # The offspring are purely mutations of the 1 elite.
    # ElitePoolSelection(elite_k=1) ensures all parents selected for the
    # mating pool are copies of the best individual.
    lambda_size = pop_size - elitism

    engine = GeneticEngine(
        genome_config=config,
        evaluator=evaluator,
        selection=CGPSelection(
            num_selections=lambda_size * 2,  # *2 because crossover takes 2 parents
            elite_k=1,
            n_elites=elitism,
        ),
        crossover=NoOpCrossover(num_offspring=1),
        mutation=CartesianMicroMutation(mutation_rate=mutation_rate, num_offspring=1),
        engine_params=engine_params,
    )

    return engine
