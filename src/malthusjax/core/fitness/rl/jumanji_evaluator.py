"""Jumanji evaluator integration for MalthusJAX."""

from __future__ import annotations

from typing import Any, Callable, Tuple, cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@struct.dataclass
class JumanjiEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Jumanji combinatorial RL environments."""

    env_name: str = struct.field(pytree_node=False, default="Snake-v1")
    network_layers: Tuple[int, ...] = struct.field(pytree_node=False, default=(64, 64))
    max_steps: int = struct.field(pytree_node=False, default=1000)
    num_eval_envs: int = struct.field(pytree_node=False, default=1)
    seed: int = struct.field(pytree_node=False, default=42)
    maximize: bool = struct.field(pytree_node=False, default=True)


class GenericMaskedPolicy(nn.Module):
    """Basic MLP policy that attempts to handle Jumanji's masked observations."""

    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, obs: Any) -> chex.Array:
        # Extract the action mask if it exists
        mask = getattr(obs, "action_mask", None)

        # Heuristic to extract the main feature array from the observation struct
        if hasattr(obs, "observation"):
            flat_obs = jnp.reshape(obs.observation, -1)
        elif hasattr(obs, "grid"):
            flat_obs = jnp.reshape(obs.grid, -1)
        elif hasattr(obs, "feature"):
            flat_obs = jnp.reshape(obs.feature, -1)
        else:
            # If it's a simple array, just flatten it
            flat_obs = jnp.reshape(obs, -1)

        x = flat_obs
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.features[-1])(x)

        # Apply action masking
        if mask is not None:
            # Set logits of invalid actions to a very small number
            logits = jnp.where(mask, logits, -1e9)

        return logits


@struct.dataclass
class JumanjiEvaluator(BaseEvaluator[RealGenome, JumanjiEvaluatorConfig, Any]):
    """Jumanji fitness evaluation interface."""

    env: Any = struct.field(pytree_node=False)
    policy: nn.Module = struct.field(pytree_node=False)
    unflatten_fn: Callable[[chex.Array], Any] = struct.field(pytree_node=False)

    @classmethod
    def create(cls, config: JumanjiEvaluatorConfig) -> JumanjiEvaluator:
        env = jumanji.make(config.env_name)  # type: ignore[attr-defined]

        # Determine action dimension
        action_spec = env.action_spec
        if hasattr(action_spec, "num_values"):
            # typically discrete or multi-discrete
            # if multi-discrete, num_values is an array. We will take prod
            num_vals = jnp.array(action_spec.num_values)
            action_dim = int(jnp.prod(num_vals))
        else:
            action_dim = action_spec.shape[-1] if len(action_spec.shape) > 0 else 1

        policy = GenericMaskedPolicy(features=config.network_layers + (action_dim,))

        # Dummy init to get structure
        rng = jax.random.PRNGKey(0)
        _, dummy_timestep = env.reset(rng)

        # Policy is evaluated on the observation, not the full timestep
        dummy_params = policy.init(rng, dummy_timestep.observation)

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

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        rng = jax.random.PRNGKey(self.config.seed)
        params = self.unflatten_fn(genome.values)

        def rollout_episode(rng_input: chex.PRNGKey) -> chex.Numeric:
            rng_reset, rng_episode = jax.random.split(rng_input)
            env_state, timestep = self.env.reset(rng_reset)

            def step_fn(
                carry: Tuple[Any, Any, chex.Numeric, chex.Array], _: Any
            ) -> Tuple[Tuple[Any, Any, chex.Numeric, chex.Array], Any]:
                env_state, timestep, cum_reward, done = carry

                # Forward pass - directly pass observation (no batch dim needed since we flatten inside)
                action_logits = cast(chex.Array, self.policy.apply(params, timestep.observation))
                action = jnp.argmax(action_logits)

                next_state, next_timestep = self.env.step(env_state, action)

                # Jumanji timestep has reward
                reward = next_timestep.reward * (1.0 - done)
                new_cum_reward = cum_reward + reward

                # Check for termination
                new_done = jnp.logical_or(done, next_timestep.last())

                return (next_state, next_timestep, new_cum_reward, new_done), None

            carry_init = (env_state, timestep, jnp.array(0.0), jnp.array(False, dtype=bool))
            final_carry, _ = jax.lax.scan(step_fn, carry_init, None, length=self.config.max_steps)
            _, _, total_reward, _ = final_carry
            return total_reward

        # Vmap over eval envs
        rngs = jax.random.split(rng, self.config.num_eval_envs)
        rewards = jax.vmap(rollout_episode)(rngs)
        mean_reward = jnp.mean(rewards)

        return mean_reward if self.config.maximize else -mean_reward
