"""Genomes module for malthus_gp."""

from malthus_gp.genomes.cartesian_genome import (
    CartesianGenome,
    CartesianGenomeConfig,
    CartesianPopulation,
)
from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
    DifferentiableCartesianGenomeConfig,
    DifferentiableCartesianPopulation,
)

__all__ = [
    "CartesianGenome",
    "CartesianGenomeConfig",
    "CartesianPopulation",
    "DifferentiableCartesianGenome",
    "DifferentiableCartesianGenomeConfig",
    "DifferentiableCartesianPopulation",
]
