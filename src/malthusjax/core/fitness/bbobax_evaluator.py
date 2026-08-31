from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.random as jr

# bbobax imports
from bbobax.bbob import BBOB_PROBLEMS
from bbobax.problem import BBOBParams, BBOBProblem
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@struct.dataclass
class BBOBAXConfig(BaseEvaluatorConfig):
    """Configuration for the bbobax-based evaluator.

    Attributes:
        fn_name: Name of the BBOB function (e.g., 'sphere', 'rastrigin').
        num_dims: Problem dimensionality.
        seed: Seed for sampling instance parameters (shifts/rotations).
        max_dims: The fixed-size dimension for JIT (defaults to num_dims).
    """

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore
    num_dims: int = struct.field(pytree_node=False, default=2)  # type: ignore
    seed: int = 0
    max_dims: int = struct.field(pytree_node=False, default=None)  # type: ignore


@struct.dataclass
class BBOBAXEvaluator(BaseEvaluator[RealGenome, BBOBAXConfig, Any]):
    """Evaluator using the pure-JAX bbobax implementation."""

    # task is static as it contains function references
    task: BBOBProblem = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    params: BBOBParams

    @classmethod
    def create(cls, config: BBOBAXConfig) -> BBOBAXEvaluator:
        """Factory method to initialize the bbobax task and instance parameters."""
        max_dims = config.max_dims or config.num_dims

        if config.fn_name not in BBOB_PROBLEMS:
            raise ValueError(
                f"Unknown function '{config.fn_name}'. Available: {list(BBOB_PROBLEMS.keys())}"
            )

        # Initialize the specific BBOB problem
        task = BBOB_PROBLEMS[config.fn_name](num_dims=max_dims)

        rng = jr.PRNGKey(config.seed)
        params = task.sample(rng)

        return cls(config=config, data=None, task=task, params=params)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single solution vector."""
        x = genome.values
        # Note: BBOB uses a key for its internal noise model
        # We use a dummy key here to keep evaluation deterministic relative to task seed
        rng = jr.PRNGKey(0)

        eval_result = self.task.evaluate(rng, x, self.params)
        # Respect MalthusJAX minimization convention
        # bbobax returns minimization objective by default.
        # If config says maximize=True, we negate so that the engine's argmin maximizes it.
        return -eval_result.fitness if self.config.maximize else eval_result.fitness

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """Vectorized evaluation of a whole population
        Uses jax.vmap to lift the single evaluate call
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=fitness_scores)
        )
