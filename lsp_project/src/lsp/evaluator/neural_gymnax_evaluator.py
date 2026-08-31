"""NeuralGymnaxEvaluator: RL environment rollout for NeuralGenome."""

from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import chex
import gymnax
import jax
import jax.numpy as jnp
from flax import struct
from lsp.evaluator.neural_evaluator import neural_forward_one
from lsp.genome.neural import NeuralGenome

from malthusjax.composer.decorators import register_fitness
from malthusjax.core.fitness.base import BaseEvaluatorConfig, StochasticEvaluator


@struct.dataclass
class NeuralGymnaxEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for NeuralGymnaxEvaluator."""

    env_name: str = struct.field(pytree_node=False, default="CartPole-v1")
    activation: str = struct.field(pytree_node=False, default="tanh")
    obs_dim: int = struct.field(pytree_node=False, default=4)
    action_dim: int = struct.field(pytree_node=False, default=2)
    max_steps: int = struct.field(pytree_node=False, default=500)
    num_eval_envs: int = struct.field(pytree_node=False, default=4)
    # Gradient refinement settings (hybrid mode)
    refine_steps: int = struct.field(pytree_node=False, default=0)
    refine_lr: float = struct.field(pytree_node=False, default=1e-2)


@struct.dataclass
class NeuralGymnaxEvaluator(StochasticEvaluator[NeuralGenome, NeuralGymnaxEvaluatorConfig, Any]):
    """Gymnax fitness evaluation for NeuralGenome using lax.scan."""

    env: Any = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, config: NeuralGymnaxEvaluatorConfig) -> NeuralGymnaxEvaluator:
        env, env_params = gymnax.make(config.env_name)
        return cls(
            config=config,
            data=None,
            env=env,
            env_params=env_params,
        )

    def _get_activation(self) -> Callable[[chex.Array], chex.Array]:
        from lsp.genome.neural import ACTIVATIONS

        return ACTIVATIONS[self.config.activation]

    def refine_weights(self, genome: NeuralGenome, rng: chex.PRNGKey) -> NeuralGenome:
        """Run gradient ascent on the RL reward (Policy Gradient / Backprop)."""
        if self.config.refine_steps == 0:
            return genome

        lr = self.config.refine_lr

        def loss_fn(weights: chex.Array) -> chex.Array:
            """Negative reward as a function of weights (to minimize)."""
            g_w = genome.replace(weights=weights)  # type: ignore[attr-defined]
            # Average reward over a few random rollouts
            rngs = jax.random.split(rng, self.config.num_eval_envs)
            rewards = jax.vmap(self._rollout_episode, in_axes=(None, 0))(g_w, rngs)
            return -jnp.mean(rewards)

        grad_fn = jax.grad(loss_fn)

        weights = genome.weights
        for _ in range(self.config.refine_steps):
            g = grad_fn(weights)
            weights = weights - lr * g

        return genome.replace(weights=weights)  # type: ignore[attr-defined]

    def _rollout_episode(self, genome: NeuralGenome, rng_input: chex.PRNGKey) -> chex.Numeric:
        """Roll out one episode using NeuralGenome for the policy."""
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, env_state = self.env.reset(rng_reset, self.env_params)

        act_fn = self._get_activation()
        forward = functools.partial(
            neural_forward_one,
            num_inputs=self.config.obs_dim,
            output_dim=self.config.action_dim,
            activation_fn=act_fn,
        )

        def step_fn(
            carry: Tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], _: Any
        ) -> Tuple[Tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], Any]:
            env_state, obs, rng, cum_reward, done = carry
            rng, rng_step = jax.random.split(rng, 2)

            # Forward pass: obs is the input. all_outputs: (L, action_dim)
            all_outputs = forward(genome, obs)

            # We use the deepest row (last row) as the action output
            action_logits = all_outputs[-1]  # (action_dim,)

            # Determine action depending on continuous vs discrete
            if hasattr(self.env.action_space(self.env_params), "n"):
                action = jnp.argmax(action_logits)
            else:
                action = action_logits

            next_obs, next_state, reward, next_done, info = self.env.step(
                rng_step, env_state, action, self.env_params
            )

            reward = reward * (1.0 - done)
            new_cum_reward = cum_reward + reward
            new_done = jnp.logical_or(done, next_done)

            return (next_state, next_obs, rng, new_cum_reward, new_done), None

        carry_init = (env_state, obs, rng_episode, jnp.array(0.0), jnp.array(False, dtype=bool))
        final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=self.config.max_steps)
        _, _, _, total_reward, _ = final_carry
        return total_reward

    def evaluate(self, genome: NeuralGenome, rng: chex.PRNGKey | None = None) -> chex.Numeric:
        if rng is None:
            raise ValueError(f"{self.__class__.__name__} requires an `rng` key for evaluation.")

        rng_refine, rng_eval = jax.random.split(rng)

        # 1. Optional gradient refinement
        genome = self.refine_weights(genome, rng_refine)

        # 2. Vmap rollouts over multiple environments for stable fitness
        rngs = jax.random.split(rng_eval, self.config.num_eval_envs)
        rewards = jax.vmap(self._rollout_episode, in_axes=(None, 0))(genome, rngs)

        mean_reward = jnp.mean(rewards)

        # Engine expects to minimize, so we return negative reward
        return -mean_reward


@register_fitness("lsp_neural_gymnax")
def _create_neural_gymnax_evaluator(**kwargs) -> NeuralGymnaxEvaluator:
    # Pop any kwargs not in the config (e.g., seed passed by Composer)
    kwargs.pop("seed", None)
    config = NeuralGymnaxEvaluatorConfig(**kwargs)
    return NeuralGymnaxEvaluator.create(config)
