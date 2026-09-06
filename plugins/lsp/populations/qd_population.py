from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation

@struct.dataclass
class QDPopulation(BasePopulation):
    """A custom population container."""
    # Add your custom fields here
    # Example: custom_metadata: chex.Array = struct.field(default_factory=lambda: jnp.array([]))
    pass
