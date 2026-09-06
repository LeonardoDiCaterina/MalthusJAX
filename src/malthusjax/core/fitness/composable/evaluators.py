"""Composable Evaluator implementations.

Provides:
- ``SupervisedEvaluator``: runs SL evaluation loop (predict → loss).
- ``OptimizationEvaluator``: runs direct objective evaluation loop.
- ``RLEvaluator``: runs RL evaluation loop.

These evaluators follow the four-axis decomposition:
  Evaluator = Problem(TaskType × Environment × Transform × Interpreter) × Output

See docs/evaluator_design.md for the full design.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.composable.base import (
    BaseInterpreter,
    BaseOptimizationEnvironment,
    BaseOutputMode,
    BaseRLEnvironment,
    BaseSupervisedEnvironment,
    BaseTransform,
    IdentityTransform,
)

E = TypeVar("E")
T = TypeVar("T", bound=BaseTransform)
I = TypeVar("I", bound=BaseInterpreter)
O = TypeVar("O", bound=BaseOutputMode)


# =============================================================================
# BaseComposableEvaluator
# =============================================================================

@struct.dataclass
class BaseComposableEvaluator(Generic[E, T, I, O]):
    """Base class for the 4-axis evaluator paradigm."""

    env: E = struct.field(pytree_node=False)
    interpreter: I = struct.field(pytree_node=False)
    output: O = struct.field(pytree_node=False)
    transform: T = struct.field(pytree_node=False, default_factory=IdentityTransform)

    def get_gap_to_optimum(self, fitness: jax.Array) -> float:
        """Returns the gap to optimum, delegating to the environment if possible."""
        if hasattr(self.env, "get_gap_to_optimum"):
            return self.env.get_gap_to_optimum(fitness)
        # Default fallback if optimum is unknown
        import jax.numpy as jnp
        return float(jnp.nan)

    def evaluate(self, raw_genome: Any, compiled_genome: Any, *args, **kwargs) -> tuple[chex.Array, dict]:
        """Evaluate a single genome and return (fitness, info)."""
        raise NotImplementedError

    def evaluate_population(
        self, population: BasePopulation, rng: chex.PRNGKey | None = None, state: Any = None, **kwargs
    ) -> BasePopulation:
        """Vectorized stochastic or deterministic population evaluation.

        Args:
            population: Population with ``.genes`` containing a batch of genomes.
            rng: PRNGKey required for stochastic evaluations (like RL).
            state: Optional algorithm state, used by some Transforms (like TensorNEAT).
            kwargs: Additional arguments to pass to ``evaluate``.

        Returns:
            Population with updated ``.fitness`` array and ``.info`` dictionary.
        """
        # 1. Transform population
        compiled_pop = self.transform.transform_population(population.genes, state=state)

        # 2. Evaluate
        if rng is not None:
            rngs = jax.random.split(rng, population.fitness.shape[0])
            fitness_scores, infos = jax.vmap(self.evaluate)(population.genes, compiled_pop, rngs, **kwargs)
        else:
            fitness_scores, infos = jax.vmap(self.evaluate)(population.genes, compiled_pop, **kwargs)

        return cast(Any, population).replace(fitness=fitness_scores, info=infos)


# =============================================================================
# SupervisedEvaluator
# =============================================================================

@struct.dataclass
class SupervisedEvaluator(
    BaseComposableEvaluator[BaseSupervisedEnvironment, BaseTransform, BaseInterpreter, BaseOutputMode]
):
    """Composable evaluator for SupervisedTask problems."""

    def evaluate(self, raw_genome: Any, compiled_genome: Any, *args, **kwargs) -> tuple[chex.Array, dict]:
        X = self.env.X
        y = self.env.y

        predictions = jax.vmap(
            lambda x_i: self.interpreter.apply(compiled_genome, x_i)
        )(X)

        return self.output.process_sl(raw_genome, X, y, predictions)


# =============================================================================
# OptimizationEvaluator
# =============================================================================

@struct.dataclass
class OptimizationEvaluator(
    BaseComposableEvaluator[BaseOptimizationEnvironment, BaseTransform, BaseInterpreter, BaseOutputMode]
):
    """Composable evaluator for OptimizationTask problems."""

    def evaluate(self, raw_genome: Any, compiled_genome: Any, *args, **kwargs) -> tuple[chex.Array, dict]:
        solution = self.interpreter.apply(compiled_genome, inputs=None)
        raw_score = self.env.evaluate(solution)
        return self.output.process_opt(raw_genome, solution, raw_score)


# =============================================================================
# RLEvaluator
# =============================================================================

@struct.dataclass
class RLEvaluator(
    BaseComposableEvaluator[BaseRLEnvironment, BaseTransform, BaseInterpreter, BaseOutputMode]
):
    """Composable evaluator for ReinforcementTask problems."""

    num_eval_envs: int = struct.field(pytree_node=False, default=1)
    max_steps: int = struct.field(pytree_node=False, default=500)

    def evaluate(self, raw_genome: Any, compiled_genome: Any, rng: chex.PRNGKey, **kwargs) -> tuple[chex.Array, dict]:
        max_steps = getattr(self.env, "max_steps", self.max_steps)

        def rollout_episode(rng_input: chex.PRNGKey) -> tuple[chex.Numeric, Any, Any]:
            rng_reset, rng_episode = jax.random.split(rng_input)
            obs, env_state = self.env.reset(rng_reset)

            def step_fn(
                carry: tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], _: Any
            ) -> tuple[tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], Any]:
                env_state, obs, step_rng, cum_reward, done = carry
                step_rng, rng_action = jax.random.split(step_rng, 2)

                flat_obs = self.env.preprocess_obs(obs)
                action_logits = self.interpreter.apply(compiled_genome, flat_obs)
                action = self.env.postprocess_action(action_logits, obs)

                next_obs, next_state, reward, next_done, info = self.env.step(
                    env_state, action, rng_action
                )

                reward = reward * (1.0 - done)
                new_cum_reward = cum_reward + reward
                new_done = jnp.logical_or(done, next_done)

                step_info = self.output.observe_step(env_state, obs, action, reward)

                return (next_state, next_obs, step_rng, new_cum_reward, new_done), step_info

            carry_init = (env_state, obs, rng_episode, jnp.array(0.0), jnp.array(False, dtype=bool))
            final_carry, step_infos = jax.lax.scan(step_fn, carry_init, None, length=max_steps)
            final_state, _, _, total_reward, _ = final_carry

            return total_reward, final_state, step_infos

        rngs = jax.random.split(rng, self.num_eval_envs)
        total_rewards, final_states, stacked_step_infos = jax.vmap(rollout_episode)(rngs)

        mean_reward = jnp.mean(total_rewards)

        return self.output.process_rl(raw_genome, mean_reward, final_states, stacked_step_infos)


# =============================================================================
# TensorNeatEvaluator
# =============================================================================

try:
    from tensorneat.common import State as TNState
except ImportError:
    TNState = Any

@struct.dataclass
class TensorNeatEvaluator(
    BaseComposableEvaluator[BaseOptimizationEnvironment, BaseTransform, BaseInterpreter, BaseOutputMode]
):
    """Composable evaluator specifically designed to wrap TensorNEAT problems.
    
    This overrides `evaluate_population` to seamlessly integrate TensorNEAT's internal
    state management, batch evaluation, and descriptor extraction.
    """

    forward_fn: Any = struct.field(pytree_node=False, default=None)
    seed: int = struct.field(pytree_node=False, default=42)
    maximize: bool = struct.field(pytree_node=False, default=True)

    def evaluate(self, raw_genome: Any, compiled_genome: Any, *args, **kwargs) -> tuple[chex.Array, dict]:
        raise NotImplementedError("TensorNeatEvaluator relies on vectorized batch evaluation via evaluate_population.")

    def evaluate_population(
        self, population: BasePopulation, rng: chex.PRNGKey | None = None, state: Any = None, **kwargs
    ) -> BasePopulation:

        # 1. Initialize a localized TensorNEAT State to manage PRNG for this generation
        key = jax.random.PRNGKey(self.seed) if rng is None else rng
        tn_state = TNState(randkey=key, generation=jnp.float32(0))

        # 2. Extract raw topology matrices
        pop_values = getattr(population.genes, "values", population.genes)
        pop_size = pop_values[0].shape[0]
        keys = jax.random.split(key, pop_size)

        # 3. Transform Population
        # We use the transform axis which internally calls algorithm.transform
        transformed_pop = self.transform.transform_population(population.genes, state=tn_state)

        # 4. Evaluate using the TensorNEATProblemWrapper
        problem = getattr(self.env, "problem", self.env)
        raw_results = jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(
            tn_state, keys, self.forward_fn, transformed_pop
        )

        if isinstance(raw_results, tuple) and len(raw_results) == 2:
            fitnesses, descriptors = raw_results
        else:
            fitnesses = raw_results
            descriptors = jnp.zeros((pop_size, 0))

        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        if not self.maximize:
            fitnesses = -1.0 * fitnesses

        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors

        return cast(Any, population).replace(fitness=fitnesses, info=new_info)

