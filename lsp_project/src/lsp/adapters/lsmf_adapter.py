"""LSMF dCGPANN Adapter.

Provides a factory for creating a MalthusJAX GeneticEngine pipeline
configured with the LSMF algorithm for evolving Neural Cartesian Genomes.
"""

from typing import Any

import optax
from lsp.adapters.cgp_adapter import CGPSelection
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.genome.neural_cartesian import NeuralCartesianGenomeConfig
from lsp.operators.cartesian import CartesianMicroMutation, NoOpCrossover

from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams


def build_lsmf_engine(
    config: NeuralCartesianGenomeConfig,
    base_evaluator: Any,
    pop_size: int = 100,
    elitism: int = 1,
    mutation_rate: float = 0.05,
    num_generations: int = 10,
    learning_rate: float = 0.01,
    cooldown_epochs: int = 1,
) -> GeneticEngine:
    """Builds a GeneticEngine configured for the LSMF algorithm.

    Constructs a 1 + λ Evolutionary Strategy engine that employs the LSMFEvaluator
    to optimize continuous weights via Optax (Adam) while preserving neutral
    drift using CGPSelection.

    Args:
        config: Genome configuration.
        base_evaluator: The NeuralCartesianEvaluator representing the forward pass.
        pop_size: Total population size (1 + λ). Default 100 implies λ=99.
        elitism: Number of elites to preserve (should always be 1 for CGP).
        mutation_rate: Probability of mutating topological genes (ops, args).
        num_generations: Number of evolutionary iterations.
        learning_rate: The learning rate for the inner optax SGD loop.
        cooldown_epochs: The number of SGD epochs to run per evolutionary cycle.

    Returns:
        A compiled MalthusJAX GeneticEngine.
    """
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=elitism,
        num_generations=num_generations,
    )

    lambda_size = pop_size - elitism

    # Wrap the evaluator in the LSMF Memetic layer
    optimizer = optax.adam(learning_rate)
    lsmf_evaluator = LSMFEvaluator(
        config=base_evaluator.config,
        data=base_evaluator.data,
        base_evaluator=base_evaluator,
        optimizer=optimizer,
        epochs=cooldown_epochs
    )

    engine = GeneticEngine(
        genome_config=config,
        evaluator=lsmf_evaluator,
        selection=CGPSelection(
            num_selections=lambda_size * 2,  # *2 because crossover takes 2 parents
            elite_k=1,
            n_elites=elitism,
        ),
        crossover=NoOpCrossover(num_offspring=1),
        mutation=CartesianMicroMutation(
            mutation_rate=mutation_rate,
            num_offspring=lambda_size,
        ),
        engine_params=engine_params,
    )

    return engine

__all__ = ["build_lsmf_engine"]
