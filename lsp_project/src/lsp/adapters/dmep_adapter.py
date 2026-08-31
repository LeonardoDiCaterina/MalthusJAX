"""LSP Differentiable MEP (dMEP) Adapter.

Provides a factory for creating a MalthusJAX GeneticEngine pipeline
configured with MEP operators for the discrete topology, and the LSMF
Evaluator for continuous weight/bias optimization via SGD/Adam.
"""

from typing import Any

import optax
from lsp.evaluator.lsmf_evaluator import LSMFEvaluator
from lsp.genome.neural_linear import NeuralPrefixGenomeConfig
from lsp.operators.crossover import MEPOnePointCrossover
from lsp.operators.mutation import MEPMicroMutation

from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.tournament import TournamentSelection


def build_dmep_engine(
    config: NeuralPrefixGenomeConfig,
    base_evaluator: Any,
    optimizer: optax.GradientTransformation,
    epochs: int = 5,
    pop_size: int = 30,
    elitism: int = 1,
    crossover_rate: float = 0.7,
) -> GeneticEngine:
    """Builds a GeneticEngine configured for Differentiable MEP (dMEP).

    Args:
        config: Genome configuration.
        base_evaluator: The instantiated NeuralPrefixEvaluator.
        optimizer: Optax optimizer for the continuous parameters.
        epochs: Number of SGD epochs per generation (Memetic Learn Step).
        pop_size: Number of individuals in the population.
        elitism: Number of best individuals preserved unconditionally.
        crossover_rate: Probability of applying crossover between pairs.

    Returns:
        A compiled MalthusJAX GeneticEngine.
    """
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=elitism,
        num_generations=50,
    )

    # Standard MEP Mutation: p_m = 2.0 / length
    mut_rate = 2.0 / config.length

    # Wrap the base evaluator in the Memetic LSMF Evaluator
    memetic_evaluator = LSMFEvaluator(
        config=base_evaluator.config,
        data=base_evaluator.data,
        base_evaluator=base_evaluator,
        optimizer=optimizer,
        epochs=epochs,
    )

    engine = GeneticEngine(
        genome_config=config,
        evaluator=memetic_evaluator,
        selection=TournamentSelection(
            num_selections=(pop_size - elitism) * 2, n_elites=elitism, tournament_size=2
        ),
        crossover=MEPOnePointCrossover(num_offspring=1, crossover_rate=crossover_rate),
        mutation=MEPMicroMutation(mutation_rate=mut_rate, num_offspring=1),
        engine_params=engine_params,
    )

    return engine
