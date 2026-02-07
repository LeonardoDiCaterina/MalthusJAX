from __future__ import annotations

from typing import Any, Generic, Tuple, TypeVar, cast

import chex
import jax
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C", bound="BaseEvaluatorConfig")  # Config type
D = TypeVar("D")  # Data type (e.g., training data, environment params)


@struct.dataclass
class BaseEvaluatorConfig:
    """Base configuration for fitness evaluation.

    Attributes:
        maximize: Optimization direction. If True, higher fitness is better;
            otherwise, lower is better. Controls jax.lax.select branching in
            single and batch evaluators.
    """

    maximize: bool = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]


@struct.dataclass
class BaseEvaluator(Generic[G, C, D]):
    """JAX-native fitness evaluation interface with vmap composition.

    Defines single-genome evaluate(genome) -> scalar, which vmaps into
    evaluate_population(population) -> population for Struct-of-Arrays (SoA)
    batching. Config (C) and data (D) are static (pytree_node=False) to
    remain constant across vmap lifting; only genomes are batched.

    Type Parameters:
        G: Genome type (e.g., RealGenome, BinaryGenome).
        C: Config type; typically pytree_node=False (static across vmap).
        D: Data type (e.g., training data, problem parameters); static.
    """

    config: C
    data: D

    def evaluate(self, genome: G) -> chex.Numeric:
        """Compute fitness for a single genome.

        Args:
            genome: Individual genome instance with unbatched values shape.

        Returns:
            Scalar fitness value (JAX array, JIT-compatible).

        Note:
            Expected input shape: genome.values shape (d,) or scalar.
            Expected output shape: scalar (or (k,) for multi-objective).
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> BasePopulation[G]:
        """Vectorized population evaluation via jax.vmap.

        Applies evaluate() to each individual by vmapping over the SoA-lifted
        genes PyTree. Config and data remain static (in_axes=None); only
        individual genomes vary (in_axes matched to genes structure).

        Args:
            population: Population with batched genes (leading shape (N,)).

        Returns:
            New population with fitness values updated. Fitness shape: (N,).
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(BasePopulation[G], cast(Any, population).replace(fitness=fitness_scores))


# Type-safe alias for regression data (Features, Targets)
RegressionData = Tuple[chex.Array, chex.Array]
