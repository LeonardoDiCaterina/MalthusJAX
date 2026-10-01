"""NEAT Evolutionary Engine Factory and Registry Integration.

Provides standard GeneticEngine construction tailored for Neuroevolution of
Augmenting Topologies, registering the 'neat' engine string with MalthusJAX EngineRegistry.
"""

from __future__ import annotations

from typing import Any

from malthus_neat.genomes.neat_genome import NeatGenomeConfig
from malthus_neat.operators.neat_crossover import NeatCrossover
from malthus_neat.operators.neat_mutation import NeatMutation
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.engine_factory import GeneticEngineAdapter
from malthusjax.composer.engine_registry import register as register_engine
from malthusjax.composer.genome_catalog import GenomeCatalog
from malthusjax.core.logger import get_logger
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams

logger = get_logger("malthus_neat.engine.neat_engine")


def build_neat_engine(
    genome_config: NeatGenomeConfig,
    evaluator: Any,
    pop_size: int = 50,
    elitism: int = 2,
    generations: int = 100,
    selection: Any = None,
    crossover: Any = None,
    mutation: Any = None,
    maximize: bool = False,
) -> GeneticEngine:
    """Builds a MalthusJAX GeneticEngine configured for NEAT.

    Args:
        genome_config: Instantiated NeatGenomeConfig.
        evaluator: Evaluator instance (e.g. NeatEvaluator or NeatQDEvaluator).
        pop_size: Number of network individuals per generation.
        elitism: Number of top-performing networks preserved untouched.
        generations: Total generation steps.
        selection: Selection operator (defaults to Tournament selection k=2).
        crossover: Crossover operator (defaults to NeatCrossover).
        mutation: Mutation operator (defaults to NeatMutation).
        maximize: Optimization direction (False for loss minimization).

    Returns:
        Compiled MalthusJAX GeneticEngine.
    """
    if selection is None:
        selection = OperatorCatalog().get("tournament:k=2", maximize=maximize)
    if crossover is None:
        crossover = NeatCrossover(genome_config=genome_config)
    if mutation is None:
        mutation = NeatMutation(genome_config=genome_config)

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=elitism,
        num_generations=generations,
    )

    logger.debug(
        "Building NEAT engine: pop_size=%d, generations=%d, elitism=%d",
        pop_size,
        generations,
        elitism,
    )

    return GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params,
    )


def _neat_factory(
    evaluator: Any,
    selection: Any = None,
    crossover: Any = None,
    mutation: Any = None,
    genome_type: str = "neat_genome",
    genome_config: Any = None,
    pop_size: int = 50,
    generations: int = 100,
    elitism: int = 2,
    **kwargs: Any,
) -> GeneticEngineAdapter:
    """EngineRegistry factory adapter for NEAT."""
    if genome_config is None:
        genome_config = GenomeCatalog().get(genome_type)

    maximize = (
        getattr(evaluator.config, "maximize", False) if hasattr(evaluator, "config") else False
    )

    engine = build_neat_engine(
        genome_config=genome_config,
        evaluator=evaluator,
        pop_size=pop_size,
        elitism=elitism,
        generations=generations,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        maximize=maximize,
    )

    return GeneticEngineAdapter(
        genetic_engine=engine,
        genome_config=genome_config,
        maximize=maximize,
    )


# Register "neat" engine with global EngineRegistry
register_engine(
    "neat",
    _neat_factory,
    defaults={
        "pop_size": 50,
        "generations": 100,
        "elitism": 2,
        "genome_type": "neat_genome",
    },
    override=True,
)

__all__ = ["build_neat_engine", "_neat_factory"]
