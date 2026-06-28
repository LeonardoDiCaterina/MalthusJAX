from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.random as jr

# bbobax imports
from bbobax.bbob import BBOB
from bbobax.fitness_fns import bbob_fns
from bbobax.types import BBOBParams, BBOBState
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

    fn_name: str = struct.field(pytree_node=False, default="sphere")
    num_dims: int = struct.field(pytree_node=False, default=2)
    seed: int = 0
    max_dims: int = struct.field(pytree_node=False, default=None)


@struct.dataclass
class BBOBAXEvaluator(BaseEvaluator[RealGenome, BBOBAXConfig, Any]):
    """Evaluator using the pure-JAX bbobax implementation."""

    # task is static as it contains function references
    task: BBOB = struct.field(pytree_node=False)
    params: BBOBParams
    problem_state: BBOBState

    @classmethod
    def create(cls, config: BBOBAXConfig) -> BBOBAXEvaluator:
        """Factory method to initialize the bbobax task and instance parameters."""
        max_dims = config.max_dims or config.num_dims

        # Initialize the Task suite
        task = BBOB.create_default(
            min_num_dims=config.num_dims,
            max_num_dims=max_dims,
        )
        available_fns = list(bbob_fns.keys())
        if config.fn_name not in available_fns:
            raise ValueError(f"Unknown function '{config.fn_name}'. Available: {available_fns}")
        fn_id = available_fns.index(config.fn_name)
        rng = jr.PRNGKey(config.seed)
        rng_params, rng_state = jr.split(rng)
        params = task.sample(rng_params)
        # Override sampled fn_id with our specific choice
        params = params.replace(fn_id=fn_id, num_dims=config.num_dims)
        state = task.init(rng_state, params)

        return cls(config=config, data=None, task=task, params=params, problem_state=state)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single solution vector."""
        x = genome.values
        # Note: BBOB uses a key for its internal noise model
        # We use a dummy key here to keep evaluation deterministic relative to task seed
        rng = jr.PRNGKey(0)

        _, eval_result = self.task.evaluate(rng, x, self.problem_state, self.params)
        # Respect MalthusJAX maximization convention
        # bbobax returns minimization objective by default.
        return eval_result.fitness if self.config.maximize else -eval_result.fitness

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
