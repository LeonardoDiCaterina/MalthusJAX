"""TensorNEAT State Management & PRNG Lifecycle for MalthusJAX."""

from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp

try:
    from tensorneat.common import State
except ImportError:
    State = Any

from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.core.state")


def create_neat_state(key: chex.PRNGKey, generation: int = 0) -> State:
    """Initializes a new TensorNEAT State container with a PRNG key and generation counter."""
    state = State()
    state = state.register(randkey=key, generation=jnp.float32(generation))
    return state


def update_neat_key(state: State, key: chex.PRNGKey) -> State:
    """Updates the PRNG key in the TensorNEAT state."""
    return state.update(randkey=key)


__all__ = ["State", "create_neat_state", "update_neat_key"]
