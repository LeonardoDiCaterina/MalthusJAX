"""Brax evaluator integration for MalthusJAX."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from brax import envs
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig, StochasticEvaluator
from malthusjax.core.genome.real_genome import RealGenome


@struct.dataclass
class BraxEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Brax RL environments."""

    env_name: str = struct.field(pytree_node=False, default="ant")
    network_layers: Tuple[int, ...] = struct.field(pytree_node=False, default=(64, 64))
    max_steps: int = struct.field(pytree_node=False, default=1000)
    num_eval_envs: int = struct.field(pytree_node=False, default=1)
    seed: int = struct.field(pytree_node=False, default=42)
    maximize: bool = struct.field(pytree_node=False, default=True)


class ContinuousMLP(nn.Module):
    """MLP policy with tanh output for continuous control."""

    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        return nn.tanh(x)  # Brax actions are typically bounded [-1, 1]


@struct.dataclass
class BraxEvaluator(StochasticEvaluator[RealGenome, BraxEvaluatorConfig, Any]):
    """Brax fitness evaluation interface."""

    env: Any = struct.field(pytree_node=False)
    policy: nn.Module = struct.field(pytree_node=False)
    unflatten_fn: Callable[[chex.Array], Any] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, config: BraxEvaluatorConfig) -> BraxEvaluator:
        env = envs.create(env_name=config.env_name)

        action_dim = env.action_size

        policy = ContinuousMLP(features=config.network_layers + (action_dim,))

        # Dummy init to get structure
        rng = jax.random.PRNGKey(0)
        obs_shape = (env.observation_size,)
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
            env_state = self.env.reset(rng_input)

            def step_fn(
                carry: Tuple[Any, chex.Numeric, chex.Array], _: Any
            ) -> Tuple[Tuple[Any, chex.Numeric, chex.Array], Any]:
                env_state, cum_reward, done = carry

                # Forward pass
                action = self.policy.apply(weights, env_state.obs[None, ...])[0]

                next_state = self.env.step(env_state, action)

                reward = next_state.reward * (1.0 - done)
                new_cum_reward = cum_reward + reward
                new_done = jnp.logical_or(done, next_state.done)

                return (next_state, new_cum_reward, new_done), None

            carry_init = (env_state, jnp.array(0.0), jnp.array(False, dtype=bool))
            final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=self.config.max_steps)
            _, total_reward, _ = final_carry
            return total_reward

        # Vmap over eval envs
        rngs = jax.random.split(rng, self.config.num_eval_envs)
        rewards = jax.vmap(rollout_episode)(rngs)
        mean_reward = jnp.mean(rewards)

        return -mean_reward if self.config.maximize else mean_reward
