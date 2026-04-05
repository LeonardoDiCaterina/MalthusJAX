"""Wrapper around evosax BBOB benchmark problems.

Exposes a MalthusJAX-friendly evaluator that leverages the evosax
BBOBProblem API for deterministic black‑box function evaluation. Plays
nicely with our population and maximization conventions.
"""

from __future__ import annotations

from typing import Any, cast

import chex
import flax
import jax
from evosax.problems import BBOBProblem
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome

BBOB_NAME_ALIASES = {
    "sphere": "sphere",
    "rastrigin": "rastrigin",
    "rastrigin_original": "rastrigin",
    "rastrigin_rotated": "rastrigin_rotated",
    "ellipsoidal": "ellipsoidal",
    "ellipsoidal_original": "ellipsoidal",
    "ellipsoidal_rotated": "ellipsoidal_rotated",
    "rosenbrock": "rosenbrock",
    "rosenbrock_original": "rosenbrock",
    "rosenbrock_rotated": "rosenbrock_rotated",
    "schwefel": "schwefel",
    "griewank_rosenbrock": "griewank_rosenbrock",
}


@struct.dataclass
class BBOBConfig(BaseEvaluatorConfig):
    """Configuration for BBOB black-box optimization benchmarks.

    Attributes:
        fn_name: BBOB function identifier (lowercase: 'sphere', 'rastrigin', etc).
        num_dims: Problem dimensionality.
        seed: Seed ID for problem instance (rotation matrices, shifts).
    """

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore[no-untyped-call]
    num_dims: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    seed: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]


@struct.dataclass
class BBOBEvaluator(BaseEvaluator[RealGenome, BBOBConfig, Any]):
    """BBOB benchmark wrapper for MalthusJAX RealGenome optimization.

    Wraps evosax BBOBProblem for standard black-box test functions. The problem
    instance and its state are stored as non-PyTree fields (static).
    """

    # Evosax problem is static (doesn't change)
    evosax_problem: Any = flax.struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    problem_state: Any = flax.struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    @classmethod
    def create(cls, config: BBOBConfig) -> BBOBEvaluator:
        """Factory method to initialize the evosax BBOBProblem."""
        fn_name_lower = config.fn_name.lower()
        fn_name_normalized = BBOB_NAME_ALIASES.get(fn_name_lower, config.fn_name)

        problem = BBOBProblem(fn_name=fn_name_normalized, num_dims=config.num_dims, seed=config.seed)

        rng = jax.random.PRNGKey(config.seed)
        problem_state = problem.init(rng)

        return cls(config=config, data=None, evosax_problem=problem, problem_state=problem_state)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Single genome evaluation via the evosax fitness function.

        The input vector is reshaped to match evosax expectations and evaluated
        deterministically. Output sign is flipped when minimizing.
        """
        x = genome.values[None, :]
        rng = jax.random.PRNGKey(0)

        fitness_scores, _, _ = self.evosax_problem.eval(rng, x, self.problem_state)
        result = fitness_scores[0]

        # Evosax BBOB problems are minimization objectives by default.
        # For maximize=True we keep the raw score as-is (higher is better).
        # For maximize=False we negate the objective so the engine can
        # maximize fitness internally.
        return result if self.config.maximize else -result

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """Vectorized batch evaluation using evosax native evaluation.

        The population genes array is fed directly into the fitness function with
        deterministic evaluation. The resulting fitness vector respects the maximize flag.
        """
        X = population.genes.values

        rng = jax.random.PRNGKey(0)

        fitness_scores, _, _ = self.evosax_problem.eval(rng, X, self.problem_state)

        # Evosax BBOB problems are minimization objectives by default.
        # When maximize=False we negate the raw scores so the genetic engine can
        # always operate with maximization semantics internally.
        final_fitness = jax.lax.select(self.config.maximize, fitness_scores, -fitness_scores)

        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=final_fitness)
        )
