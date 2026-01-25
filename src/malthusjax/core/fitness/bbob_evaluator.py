"""
BBOB Adapter for MalthusJAX.
Wraps evosax's BBOB suite for use in MalthusJAX engines.
"""
from typing import Any

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
    fn_name: str = struct.field(pytree_node=False, default="sphere")
    num_dims: int = struct.field(pytree_node=False, default=2)
    seed: int = struct.field(pytree_node=False, default=42)

@struct.dataclass
class BBOBEvaluator(BaseEvaluator[RealGenome, BBOBConfig, Any]):
    """
    Wraps evosax BBOB problems for MalthusJAX.
    
    This evaluator overrides `evaluate_population` to use evosax's 
    native batch processing for maximum performance.
    """
    # We store the evosax problem instance and its state as 'data'
    # evosax problems are Flax dataclasses, so they are valid PyTree nodes.
    evosax_problem: Any = flax.struct.field(pytree_node=False)
    evosax_state: Any = flax.struct.field(pytree_node=True)

    @classmethod
    def create(cls, config: BBOBConfig) -> "BBOBEvaluator":
        """Factory method to initialize the evosax problem."""
        # 1. Initialize the BBOB problem from evosax
        problem = BBOBProblem(
            fn_name=config.fn_name,
            num_dims=config.num_dims,
            seed=config.seed
        )

        # 2. Initialize the problem state (contains rotation matrices etc.)
        # We use a fixed key for problem init to ensure reproducibility of the function landscape
        rng = jax.random.PRNGKey(config.seed)
        state = problem.init(rng)

        return cls(
            config=config,
            data=None, # Not used, we use specific fields below
            evosax_problem=problem,
            evosax_state=state
        )

    def evaluate(self, genome: RealGenome) -> chex.Array:
        """
        Single genome evaluation.
        Note: MalthusJAX engines typically call evaluate_population, not this.
        """
        # Expand dims to make it a batch of 1 for evosax
        x = genome.values[None, :]

        # We use a dummy key here because we assume noiseless BBOB for standard optimization
        rng = jax.random.PRNGKey(0)
        fitness, _, _ = self.evosax_problem.eval(rng, x, self.evosax_state)

        # BBOB returns minimization cost. If MalthusJAX is set to maximize,
        # we flip the sign.
        # However, BaseEvaluator logic usually handles direction in the Engine.
        # Standard BBOB is minimization.
        result = fitness[0]

        if self.config.maximize:
            return -result
        return result

    def evaluate_population(self, population: BasePopulation[RealGenome]) -> BasePopulation[RealGenome]:
        """
        Vectorized evaluation using evosax's native batching.
        """
        # 1. Extract raw JAX array from MalthusJAX population
        # Shape: (pop_size, num_dims)
        X = population.genes.values

        # 2. Call evosax
        # We create a batch of keys for stochastic functions (though standard BBOB is often deterministic)
        rng = jax.random.PRNGKey(0) # In a real loop, you might want to pass this in, but Evaluators are pure
        keys = jax.random.split(rng, X.shape[0])

        fitness_scores, _, _ = self.evosax_problem.eval(keys, X, self.evosax_state)

        # 3. Handle Optimization Direction (Minimization -> Maximization)
        # MalthusJAX engines generally assume HIGHER fitness is better.
        # BBOB problems return COST (lower is better).
        if self.config.maximize:
            fitness_scores = -fitness_scores

        return population.replace(fitness=fitness_scores)
