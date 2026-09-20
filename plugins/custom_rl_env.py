from typing import Any
import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseRLEnvironment

@struct.dataclass
class CustomRLEnv(BaseRLEnvironment):
    """A custom environment."""

    def reset(self, key: chex.PRNGKey):
        # TODO: Return initial (obs, state)
        return jnp.zeros(self.obs_dim), None

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey):
        # TODO: Return (next_obs, next_state, reward, done, info)
        return jnp.zeros(self.obs_dim), state, 0.0, False, {}

    def preprocess_obs(self, obs: Any) -> chex.Array:
        return obs

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        return logits

    @property
    def obs_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

