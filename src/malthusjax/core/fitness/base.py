"""Core fitness evaluation interfaces and configurations.

Defines the abstract base classes that concrete evaluators inherit from.
Includes helper types such as regression data tuples and the static
configuration dataclass used across the evaluator hierarchy.
"""

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

        The provided genome should contain unbatched values. The return value is
        a scalar (or a small vector for multiobjective cases) suitable for
        JIT‑compiled pipelines. Implementations may use ``self.config`` and
        ``self.data`` to parameterize the evaluation.
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> BasePopulation[G]:
        """Vectorized population evaluation via :func:`jax.vmap`.

        Each member of *population.genes* is passed through ``evaluate``; the
        resulting fitness array replaces the old values in a new population
        object.
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(BasePopulation[G], cast(Any, population).replace(fitness=fitness_scores))


# Type-safe alias for regression data (Features, Targets)
RegressionData = Tuple[chex.Array, chex.Array]
