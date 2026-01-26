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


@struct.dataclass
class BBOBConfig(BaseEvaluatorConfig):
    """Configuration for BBOB benchmark tasks."""

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore[no-untyped-call]
    num_dims: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    seed: int = struct.field(pytree_node=False, default=42)  # type: ignore[no-untyped-call]


@struct.dataclass
class BBOBEvaluator(BaseEvaluator[RealGenome, BBOBConfig, Any]):
    """
    Wraps evosax BBOB problems for MalthusJAX.

    This evaluator leverages evosax's native batch processing, providing
    an optimized interface for standard black-box optimization benchmarks.
    """

    # Evosax problem and state are stored directly.
    # Problem is static (static_arg), state is a PyTree node.
    evosax_problem: Any = flax.struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    evosax_state: Any = flax.struct.field(pytree_node=True)  # type: ignore[no-untyped-call]

    @classmethod
    def create(cls, config: BBOBConfig) -> BBOBEvaluator:
        """Factory method to initialize the evosax problem and its internal state."""
        problem = BBOBProblem(fn_name=config.fn_name, num_dims=config.num_dims, seed=config.seed)

        # Initialize the problem state (rotation matrices, optimum shifts, etc.)
        rng = jax.random.PRNGKey(config.seed)
        state = problem.init(rng)

        return cls(config=config, data=None, evosax_problem=problem, evosax_state=state)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """
        Single genome evaluation.

        Note: While MalthusJAX engines prioritize evaluate_population,
        this remains for compatibility and single-step debugging.
        """
        # Evosax expects (batch, dims)
        x = genome.values[None, :]
        rng = jax.random.PRNGKey(0)

        # evosax.eval returns (fitness, state, info)
        fitness, _, _ = self.evosax_problem.eval(rng, x, self.evosax_state)
        result = fitness[0]

        # standard BBOB is minimization; we flip if the engine expects maximization
        return jax.lax.select(self.config.maximize, -result, result)

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """
        Vectorized evaluation using evosax's native high-performance batching.
        """
        # Extract the batched values: (pop_size, num_dims)
        X = population.genes.values

        # Standard BBOB is deterministic, but we split keys to satisfy the API
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, X.shape[0])

        fitness_scores, _, _ = self.evosax_problem.eval(keys, X, self.evosax_state)

        # Handle optimization direction
        final_fitness = jax.lax.select(self.config.maximize, -fitness_scores, fitness_scores)

        # Cast for MyPy strictness on dynamically added .replace
        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=final_fitness)
        )
