"""Prepare supplied genomes at the adapter boundary."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
    CategoricalPopulation,
)
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def prepare_initial_population(value: Any, config: Any, size: int) -> BasePopulation:
    """Return an owned population with the configured representation."""
    if isinstance(value, BasePopulation):
        if type(value.config) is not type(config) or value.config != config:
            raise ValueError("Initial population configuration does not match the engine")
        if value.fitness.shape[0] != size:
            raise ValueError(f"Initial population must contain {size} genomes")
        genes = getattr(value.genes, "values", None)
        if isinstance(config, (RealGenomeConfig, BinaryGenomeConfig, CategoricalGenomeConfig)):
            _validate_genes(jnp.asarray(genes), config, size)
        return value.copy()

    genes = jnp.asarray(value)
    _validate_genes(genes, config, size)
    values = jnp.array(genes, dtype=config.dtype, copy=True)
    fitness = jnp.zeros((size,))
    if isinstance(config, RealGenomeConfig):
        return RealPopulation(genes=RealGenome(values), fitness=fitness, config=config)
    if isinstance(config, BinaryGenomeConfig):
        return BinaryPopulation(genes=BinaryGenome(values), fitness=fitness, config=config)
    return CategoricalPopulation(genes=CategoricalGenome(values), fitness=fitness, config=config)


def _validate_genes(genes: jax.Array, config: Any, size: int) -> None:
    if isinstance(config, BinaryGenomeConfig):
        shape = config.resolved_shape()
    elif isinstance(config, (RealGenomeConfig, CategoricalGenomeConfig)):
        shape = config.shape
    else:
        raise TypeError("Custom genome configurations require a typed initial population")
    if genes.shape != (size, *shape):
        raise ValueError(f"Initial genes must have shape {(size, *shape)}, got {genes.shape}")
    if isinstance(config, RealGenomeConfig):
        return
    if not jnp.issubdtype(genes.dtype, jnp.integer) and genes.dtype != jnp.bool_:
        raise ValueError("Discrete initial genes must have an integer dtype")
    host = np.asarray(genes)
    categories = config.num_categories if isinstance(config, CategoricalGenomeConfig) else 2
    if np.any(host < 0) or np.any(host >= categories):
        raise ValueError("Initial genes contain out-of-range categories")
    if isinstance(config, CategoricalGenomeConfig):
        permutation = config.is_permutation or (
            config.is_permutation is None and shape == (categories,)
        )
        if permutation and (
            shape != (categories,) or not np.all(np.sort(host) == np.arange(categories))
        ):
            raise ValueError("Each initial genome must be a complete permutation")
