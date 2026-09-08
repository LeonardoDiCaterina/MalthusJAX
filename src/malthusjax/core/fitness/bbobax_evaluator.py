from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.random as jr

# bbobax imports — compatible with bbobax >= 0.2 (BBOB + bbob_fns API)
from bbobax import BBOB, BBOBParams
from bbobax.bbob import bbob_fns
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome

# Public registry of supported BBOB function names
BBOB_PROBLEMS = bbob_fns


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

    # task and state are static as they contain function references and initial RNG state
    task: BBOB = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    params: BBOBParams
    state: Any = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    @classmethod
    def create(cls, config: BBOBAXConfig) -> BBOBAXEvaluator:
        """Factory method to initialize the bbobax task and instance parameters."""
        max_dims = config.max_dims or config.num_dims

        if config.fn_name not in BBOB_PROBLEMS:
            raise ValueError(
                f"Unknown function '{config.fn_name}'. Available: {list(BBOB_PROBLEMS.keys())}"
            )

        # Initialize the specific BBOB problem with a fixed dimensionality
        fn = BBOB_PROBLEMS[config.fn_name]
        task = BBOB(
            fitness_fns=[fn],
            min_num_dims=max_dims,
            max_num_dims=max_dims,
        )

        rng = jr.PRNGKey(config.seed)
        sample_key, init_key = jr.split(rng)
        params = task.sample(sample_key)
        state = task.init(init_key, params)

        return cls(config=config, data=None, task=task, params=params, state=state)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single solution vector."""
        x = genome.values
        # Use a fixed dummy key for deterministic evaluation given the task seed
        rng = jr.PRNGKey(0)
        _, eval_result = self.task.evaluate(rng, x, self.state, self.params)
        # bbobax returns a minimization objective.
        # If config says maximize=True, negate so the engine's argmin maximizes it.
        return -eval_result.fitness if self.config.maximize else eval_result.fitness

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """Vectorized evaluation of a whole population.
        Uses jax.vmap to lift the single evaluate call.
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=fitness_scores)
        )
