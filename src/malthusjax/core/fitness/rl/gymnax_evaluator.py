"""Gymnax evaluator integration for MalthusJAX."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig, StochasticEvaluator
from malthusjax.core.genome.real_genome import RealGenome


@struct.dataclass
class GymnaxEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Gymnax RL environments."""

    env_name: str = struct.field(pytree_node=False, default="CartPole-v1")
    network_layers: Tuple[int, ...] = struct.field(pytree_node=False, default=(64, 64))
    max_steps: int = struct.field(pytree_node=False, default=500)
    num_eval_envs: int = struct.field(pytree_node=False, default=1)
    seed: int = struct.field(pytree_node=False, default=42)
    maximize: bool = struct.field(pytree_node=False, default=True)


class MLP(nn.Module):
    """Simple MLP policy."""

    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        return x


@struct.dataclass
class GymnaxEvaluator(StochasticEvaluator[RealGenome, GymnaxEvaluatorConfig, Any]):
    """Gymnax fitness evaluation interface."""

    env: Any = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=False)
    policy: nn.Module = struct.field(pytree_node=False)
    unflatten_fn: Callable[[chex.Array], Any] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, config: GymnaxEvaluatorConfig) -> GymnaxEvaluator:
        env, env_params = gymnax.make(config.env_name)

        # Policy
        if hasattr(env.action_space(env_params), "n"):
            action_dim = env.action_space(env_params).n
        else:
            action_dim = env.action_space(env_params).shape[0]

        policy = MLP(features=config.network_layers + (action_dim,))

        # Dummy init to get structure
        rng = jax.random.PRNGKey(0)
        obs_shape = env.observation_space(env_params).shape
        dummy_obs = jnp.zeros((1,) + obs_shape)
        dummy_params = policy.init(rng, dummy_obs)

        # Flatten and unflatten fn
        flat_params, tree_def = jax.tree_util.tree_flatten(dummy_params)
        shapes = [p.shape for p in flat_params]
        sizes = [p.size for p in flat_params]

        # Precompute static split indices for jnp.split
        split_idx_list = []
        curr = 0
        for s in sizes[:-1]:
            curr += s
            split_idx_list.append(curr)
        split_indices = tuple(split_idx_list)

        def unflatten_fn(genome_values: chex.Array) -> Any:
            split_arrays = jnp.split(genome_values, split_indices)
            reshaped_arrays = [x.reshape(s) for x, s in zip(split_arrays, shapes)]
            return jax.tree_util.tree_unflatten(tree_def, reshaped_arrays)

        return cls(
            config=config,
            data=None,
            env=env,
            env_params=env_params,
            policy=policy,
            unflatten_fn=unflatten_fn,
        )

    def evaluate(self, genome: RealGenome, rng: chex.PRNGKey | None = None) -> chex.Numeric:
        if rng is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an `rng` key for evaluation, "
                "but None was provided."
            )

        # 1. Decode genome weights
        weights = self.unflatten_fn(genome.values)

        def rollout_episode(rng_input: chex.PRNGKey) -> chex.Numeric:
            rng_reset, rng_episode = jax.random.split(rng_input)
            obs, env_state = self.env.reset(rng_reset, self.env_params)

            def step_fn(
                carry: Tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], _: Any
            ) -> Tuple[Tuple[Any, chex.Array, chex.PRNGKey, chex.Numeric, chex.Array], Any]:
                env_state, obs, rng, cum_reward, done = carry
                rng, rng_step = jax.random.split(rng, 2)

                # Forward pass
                action_logits = self.policy.apply(weights, obs[None, ...])[0]

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

        # Vmap over eval envs
        rngs = jax.random.split(rng, self.config.num_eval_envs)
        rewards = jax.vmap(rollout_episode)(rngs)
        mean_reward = jnp.mean(rewards)

        return -mean_reward if self.config.maximize else mean_reward
