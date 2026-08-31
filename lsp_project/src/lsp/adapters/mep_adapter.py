"""LSP MEP Adapter.

Provides a factory for creating a MalthusJAX GeneticEngine pipeline
configured with MEP operators and the LinearGPEvaluator.
"""

from typing import Any

from lsp.genome import PrefixGenomeConfig
from lsp.operators.crossover import MEPOnePointCrossover
from lsp.operators.mutation import MEPMicroMutation

from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.selection.tournament import TournamentSelection


def build_mep_engine(
    config: PrefixGenomeConfig,
    evaluator: Any,
    pop_size: int = 30,
    elitism: int = 1,
    crossover_rate: float = 0.7,
    mutation_rate: float = 1.0,  # Handled at the gene level in MEPMicroMutation
) -> GeneticEngine:
    """Builds a GeneticEngine configured for Multi Expression Programming.

    Constructs the engine with MEPOnePointCrossover, MEPMicroMutation, and
    LinearTournamentSelection, ready to execute on `LinearGenome`s.

    Args:
        config: Genome configuration.
        evaluator: The instantiated LinearGPEvaluator.
        pop_size: Number of individuals in the population.
        elitism: Number of best individuals preserved unconditionally.
        crossover_rate: Probability of applying crossover between pairs.
        mutation_rate: Probability of applying mutation (usually 1.0 since
            MEPMicroMutation has its own internal per-gene rate).

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

    engine = GeneticEngine(
        genome_config=config,
        evaluator=evaluator,
        selection=TournamentSelection(
            num_selections=(pop_size - elitism) * 2, n_elites=elitism, tournament_size=2
        ),
        crossover=MEPOnePointCrossover(num_offspring=1, crossover_rate=crossover_rate),
        mutation=MEPMicroMutation(mutation_rate=mut_rate, num_offspring=1),
        engine_params=engine_params,
    )

    return engine
