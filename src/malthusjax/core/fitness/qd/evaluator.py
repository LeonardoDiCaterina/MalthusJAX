"""Quality-Diversity evaluator interface."""

from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, C, D, G
from malthusjax.core.genome.qd.population import QDPopulation, has_descriptors


@struct.dataclass
class BaseQDEvaluator(BaseEvaluator[G, C, D]):
    """JAX-native Quality-Diversity fitness evaluation interface.

    Extends the standard evaluator to compute behavioral descriptors alongside
    fitness. Returns a specialized `QDPopulation` which safely stores the
    descriptors in its `.info` dictionary, making it a drop-in replacement
    for standard scalar evaluators.
    """

    def evaluate(self, genome: G) -> chex.Numeric:
        """Fallback standard evaluation that simply discards descriptors.

        This satisfies the Liskov Substitution Principle if this evaluator
        is used in a standard GeneticEngine.
        """
        fitness, _ = self.evaluate_qd(genome)
        return fitness

    def evaluate_qd(self, genome: G) -> Tuple[chex.Numeric, chex.Array]:
        """Compute fitness and behavioral descriptor for a single genome.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluate_population(
        self, population: BasePopulation[G], rng: chex.PRNGKey | None = None, **kwargs: Any
    ) -> QDPopulation[G]:
        """Vectorized population evaluation via :func:`jax.vmap`.

        Evaluates both fitness and descriptors, returning a QDPopulation.
        """
        fitness_scores, descriptors = jax.vmap(self.evaluate_qd)(population.genes)

        # Merge info safely, inserting descriptors
        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors

        return QDPopulation(
            genes=population.genes, fitness=fitness_scores, config=population.config, info=new_info
        )


@struct.dataclass
class SimpleQDEvaluator(BaseQDEvaluator[G, Any, Any]):
    """Standard wrapper for quality-diversity objective functions.

    Expects an objective function that takes a single array of genes
    and returns a tuple of (fitness, descriptors) or (fitness, descriptors, extra).
    """

    objective_function: Any = struct.field(pytree_node=False)

    def evaluate_qd(self, genome: G) -> Tuple[chex.Numeric, chex.Array]:
        # Unwrap genome to underlying values if applicable
        gene_values = getattr(genome, "values", genome)
        result = self.objective_function(gene_values)

        fitness = result[0]
        descriptors = result[1]

        # Follows canonical lower-is-better convention:
        # If maximize=True, negate raw objective so lower is better.
        if self.config is not None and getattr(self.config, "maximize", False):
            fitness = -1.0 * fitness

        return fitness, descriptors


@struct.dataclass
class ProjectedQDEvaluator(BaseQDEvaluator[G, Any, Any]):
    """Adapts any scalar BaseEvaluator or BaseComposableEvaluator for Quality-Diversity.

    Preserves underlying evaluator properties and handles:
    1. If the base evaluator returns a population with existing valid descriptors in `.info`,
       preserves them and returns a QDPopulation directly without overwriting.
    2. Otherwise, uses `descriptor_fn` (e.g. `GenotypeBoxDescriptorFn`) to extract
       and normalize descriptors into [0, 1]^k from genomes.
    3. Preserves all diagnostic information in `updated_pop.info`.
    """

    config: Any = struct.field(pytree_node=False, default=None)
    data: Any = struct.field(pytree_node=False, default=None)
    base_evaluator: Any = struct.field(pytree_node=False, default=None)
    descriptor_fn: Any = struct.field(pytree_node=False, default=None)

    @property
    def requires_rng(self) -> bool:
        return getattr(self.base_evaluator, "requires_rng", False)

    @property
    def interpreter(self) -> Any:
        return getattr(self.base_evaluator, "interpreter", None)

    @property
    def env(self) -> Any:
        return getattr(self.base_evaluator, "env", None)

    @property
    def output(self) -> Any:
        return getattr(self.base_evaluator, "output", None)

    @property
    def f_opt(self) -> chex.Numeric | None:
        return getattr(self.base_evaluator, "f_opt", None)

    @property
    def x_opt(self) -> chex.Array | None:
        return getattr(self.base_evaluator, "x_opt", None)

    def evaluate_qd(self, genome: G) -> Tuple[chex.Numeric, chex.Array]:
        from malthusjax.core.fitness.base import BaseEvaluator

        if isinstance(self.base_evaluator, BaseEvaluator):
            fitness = self.base_evaluator.evaluate(genome)
        else:
            fitness = 0.0
        desc = self.descriptor_fn.compute(genome, None, getattr(genome, "values", genome))
        return fitness, desc

    def evaluate_population(
        self, population: BasePopulation[G], rng: chex.PRNGKey | None = None, **kwargs: Any
    ) -> QDPopulation[G]:
        from malthusjax.core.fitness.base import dispatch_evaluate_population

        updated_pop = dispatch_evaluate_population(
            self.base_evaluator, population, key=rng, **kwargs
        )

        if has_descriptors(updated_pop):
            if isinstance(updated_pop, QDPopulation):
                return updated_pop
            return QDPopulation.from_population(updated_pop)

        # Fallback projection using descriptor_fn
        descriptors = jax.vmap(
            lambda g: self.descriptor_fn.compute(g, None, getattr(g, "values", g))
        )(population.genes)
        return QDPopulation.from_population(updated_pop, descriptors=descriptors)


def to_repertoire_fitness(
    evaluator: Any, eval_fitness: chex.Array, engine_maximize: bool
) -> chex.Array:
    """Convert an evaluator's population fitness into QDax/MAP-Elites repertoire fitness (higher-is-better).

    If the evaluator obeys MalthusJAX's Single Source of Truth contract (has output.maximize
    or config.maximize), eval_fitness is already in lower-is-better form, so repertoire
    fitness is simply -eval_fitness.
    If the evaluator is an un-normalized raw function, we negate when engine_maximize is False.
    """
    target = getattr(evaluator, "base_evaluator", evaluator)
    has_output_maximize = hasattr(target, "output") and hasattr(target.output, "maximize")
    has_config_maximize = (
        hasattr(target, "config")
        and target.config is not None
        and getattr(target.config, "maximize", None) is not None
    )

    if has_output_maximize or has_config_maximize:
        # Standard MalthusJAX evaluator: eval_fitness is already lower-is-better.
        # To make it higher-is-better for the repertoire: -eval_fitness
        return -eval_fitness

    # Un-normalized raw evaluator or mock:
    return eval_fitness if engine_maximize else -eval_fitness


def ensure_qd_evaluator(
    evaluator: Any,
    *,
    num_descriptors: int = 2,
    bounds: tuple[float, float] = (-5.0, 5.0),
    descriptor_fn: Any | None = None,
) -> Any:
    """Ensure an evaluator produces a QDPopulation with valid descriptors.

    If the evaluator is already a BaseQDEvaluator or ProjectedQDEvaluator, or
    a BaseComposableEvaluator configured with QDOutput, it is returned directly.
    Otherwise, wraps it in ProjectedQDEvaluator using the provided or default
    GenotypeBoxDescriptorFn.
    """
    if isinstance(evaluator, ProjectedQDEvaluator):
        return evaluator

    if hasattr(evaluator, "output"):
        from malthusjax.core.fitness.composable.base import QDOutput

        if isinstance(evaluator.output, QDOutput):
            return evaluator

    if isinstance(evaluator, BaseQDEvaluator):
        return evaluator

    if descriptor_fn is None:
        from malthusjax.core.fitness.composable.base import GenotypeBoxDescriptorFn

        descriptor_fn = GenotypeBoxDescriptorFn(num_descriptors=num_descriptors, bounds=bounds)

    cfg = getattr(evaluator, "config", None)
    data = getattr(evaluator, "data", None)
    return ProjectedQDEvaluator(
        base_evaluator=evaluator,
        descriptor_fn=descriptor_fn,
        config=cfg,
        data=data,
    )
