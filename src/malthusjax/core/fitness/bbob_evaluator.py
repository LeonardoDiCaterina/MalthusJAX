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
    """Configuration for BBOB black-box optimization benchmarks.

    Attributes:
        fn_name: BBOB function identifier (e.g., 'sphere', 'rastrigin').
        num_dims: Problem dimensionality.
        seed: Random seed for problem initialization (rotation matrices, shifts).
    """

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore[no-untyped-call]
    num_dims: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    seed: int = struct.field(pytree_node=False, default=42)  # type: ignore[no-untyped-call]


@struct.dataclass
class BBOBEvaluator(BaseEvaluator[RealGenome, BBOBConfig, Any]):
    """BBOB benchmark wrapper for MalthusJAX RealGenome optimization.

    Wraps evosax BBOBProblem for standard black-box test functions. Problem
    and state are stored as non-PyTree fields (pytree_node=False for problem,
    True for state) since problem.eval mutates internal state during
    evaluation and must be threaded through the computation.
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
        """Single genome evaluation via evosax wrapper.

        Args:
            genome: RealGenome with values shape (d,).

        Returns:
            Scalar fitness (JAX array). Sign flips according to config.maximize.

        Note:
            Uses deterministic key (key=0) since BBOB evaluation is deterministic
            and evosax API requires key argument. Key does not affect output.
        """
        x = genome.values[None, :]
        rng = jax.random.PRNGKey(0)

        fitness, _, _ = self.evosax_problem.eval(rng, x, self.evosax_state)
        result = fitness[0]

        return jax.lax.select(self.config.maximize, -result, result)

    def evaluate_population(
        self, population: BasePopulation[RealGenome]
    ) -> BasePopulation[RealGenome]:
        """Vectorized batch evaluation using evosax native batching.

        Extracts (N, d) values array and evaluates all individuals in parallel
        via evosax.problem.eval. More efficient than vmapping evaluate() due
        to evosax's internal optimizations.

        Args:
            population: Population with genes.values shape (N, d).

        Returns:
            Population with fitness shape (N,). Sign flipped per maximize flag.
        """
        X = population.genes.values

        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, X.shape[0])

        fitness_scores, _, _ = self.evosax_problem.eval(keys, X, self.evosax_state)

        final_fitness = jax.lax.select(self.config.maximize, -fitness_scores, fitness_scores)

        return cast(
            BasePopulation[RealGenome], cast(Any, population).replace(fitness=final_fitness)
        )
