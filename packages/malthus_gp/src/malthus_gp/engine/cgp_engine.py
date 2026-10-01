"""Cartesian Genetic Programming Engine Factory.

Constructs a JIT-compiled GeneticEngine configured for Cartesian Genetic Programming,
implementing the (1 + λ)-ES strategy with neutral drift and point micro-mutations.
"""

from __future__ import annotations

from typing import Any, Optional

from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.cartesian_micro_mutation import CartesianMicroMutation
from malthus_gp.operators.neutral_selection import NeutralSelection
from malthus_gp.operators.no_op_crossover import NoOpCrossover
from malthusjax.composer.engine_registry import register as _register_engine
from malthusjax.core.logger import get_logger
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams

logger = get_logger("malthus_gp.engine.cgp_engine")


def build_cgp_engine(
    config: CartesianGenomeConfig,
    evaluator: Any,
    pop_size: int = 5,
    elitism: int = 1,
    mutation_rate: Optional[float] = None,
    num_generations: int = 2000,
    maximize: bool = False,
) -> GeneticEngine:
    """Builds a MalthusJAX GeneticEngine configured for Cartesian GP.

    Constructs a (1 + λ) Evolutionary Strategy engine with:
      - NeutralSelection (preserves 1 parent, prioritizes offspring on ties)
      - NoOpCrossover (identity pass-through for 1+λ ES)
      - CartesianMicroMutation (uniform point mutation across ops, args, and outputs)

    Args:
        config: Cartesian genome configuration.
        evaluator: Instantiated CartesianGPEvaluator.
        pop_size: Total population size (1 + λ). Default 5 implies λ=4.
        elitism: Number of elites to preserve (typically 1 for CGP).
        mutation_rate: Probability of mutating each individual gene.
                       If None, defaults to ~3 mutated genes per chromosome (3 / total_genes).
        num_generations: Number of generations to run.
        maximize: Optimization direction (False for loss minimization).

    Returns:
        A compiled MalthusJAX GeneticEngine ready for execution.
    """
    if mutation_rate is None:
        mutation_rate = 3.0 / float(config.total_genes)

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        elitism=elitism,
        num_generations=num_generations,
    )

    lambda_size = pop_size - elitism

    # In 1+λ ES, we select lambda_size * 2 parents for crossover slots
    selection_operator = NeutralSelection(
        num_selections=lambda_size * 2,
        elite_k=1,
        n_elites=elitism,
        maximize=maximize,
    )

    crossover_operator = NoOpCrossover(num_offspring=1)
    mutation_operator = CartesianMicroMutation(
        mutation_rate=mutation_rate,
        num_offspring=1,
    )

    logger.debug(
        "Building CGP engine: pop_size=%d (1+%d ES), generations=%d, mutation_rate=%.4f",
        pop_size,
        lambda_size,
        num_generations,
        mutation_rate,
    )

    return GeneticEngine(
        genome_config=config,
        evaluator=evaluator,
        selection=selection_operator,
        crossover=crossover_operator,
        mutation=mutation_operator,
        engine_params=engine_params,
    )


def _cgp_factory(
    evaluator: Any,
    selection: Any = None,
    crossover: Any = None,
    mutation: Any = None,
    genome_type: str = "cartesian_genome",
    genome_config: Any = None,
    pop_size: int = 5,
    generations: int = 2000,
    elitism: int = 1,
    mutation_rate: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """EngineRegistry factory adapter for Cartesian GP."""
    from malthusjax.composer.engine_factory import GeneticEngineAdapter
    from malthusjax.composer.genome_catalog import GenomeCatalog

    if genome_config is None:
        genome_config = GenomeCatalog().get(genome_type)

    maximize = (
        getattr(evaluator.config, "maximize", False) if hasattr(evaluator, "config") else False
    )

    engine = build_cgp_engine(
        config=genome_config,
        evaluator=evaluator,
        pop_size=pop_size,
        elitism=elitism,
        mutation_rate=mutation_rate,
        num_generations=generations,
        maximize=maximize,
    )

    return GeneticEngineAdapter(
        genetic_engine=engine,
        genome_config=genome_config,
        maximize=maximize,
    )


_register_engine(
    "cgp",
    _cgp_factory,
    defaults={
        "pop_size": 5,
        "generations": 2000,
        "elitism": 1,
        "genome_type": "cartesian_genome",
    },
    override=True,
)
