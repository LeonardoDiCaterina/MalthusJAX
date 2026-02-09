"""
PRNG key factory and helpers for MalthusJAX.

Centralizes creation of typed JAX PRNG keys and helper utilities for
legacy/compat detection and warnings.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union
import warnings

import jax
import jax.numpy as jnp


class PRNGImpl(str, Enum):
    """Available JAX PRNG implementations.

    Mirrors the pattern used by other enums in the codebase.
    """

    THREEFRY = "threefry2x32"
    PHILOX = "philox4x32_10"
    RBG = "rbg"
    UNSAFE_RBG = "unsafe_rbg"


DEFAULT_IMPL = PRNGImpl.THREEFRY


def create_key(seed: int, impl: Optional[PRNGImpl] = None) -> jax.Array:
    """Create a PRNG key using the requested implementation.

    Falls back to ``jax.random.PRNGKey`` if the new ``jax.random.key`` API
    is not available in the installed JAX version.

    Args:
        seed: Integer seed.
        impl: Optional ``PRNGImpl`` specifying the backend (philox, rbg, ...).

    Returns:
        A JAX PRNG key (array-like). Downstream ``split`` / ``fold_in`` inherit
        the implementation encoded in this key.
    """
    impl = impl or DEFAULT_IMPL

    # Prefer new jax.random.key API when available
    key_ctor = getattr(jax.random, "key", None)
    if callable(key_ctor):
        try:
            return key_ctor(seed, impl=impl.value)
        except TypeError:
            # Older jax.random.key may not accept impl argument
            warnings.warn(
                "jax.random.key exists but does not accept 'impl='; falling back to PRNGKey",
                RuntimeWarning,
            )

    # Fallback: legacy PRNGKey (uint32[2]) — preserves deterministic behavior
    warnings.warn(
        "Using legacy jax.random.PRNGKey fallback; consider upgrading JAX to use 'jax.random.key'",
        DeprecationWarning,
    )
    return jax.random.PRNGKey(seed)


def _is_legacy_prngkey(key: jax.Array) -> bool:
    """Heuristic to detect legacy PRNGKey (uint32[2]) used by older JAX.

    Legacy keys produced by ``jax.random.PRNGKey`` are uint32 arrays of length 2.
    New-style typed keys typically have dtype != uint32 or length != 2 (e.g., Philox uses a larger key).
    """
    try:
        arr = jnp.asarray(key)
        return arr.dtype == jnp.uint32 and arr.size == 2
    except Exception:
        return False


def is_new_style_key(key: jax.Array) -> bool:
    """Return True if `key` looks like a new-style typed key, False for legacy PRNGKey."""
    return not _is_legacy_prngkey(key)


def validate_key(key: jax.Array, context: str = "") -> None:
    """Validate the provided key and warn on legacy-style keys.

    This should be called at engine boundaries (e.g., `init_state`) to guide users
    toward using `create_key()` or `jax.random.key()`.
    """
    if _is_legacy_prngkey(key):
        warnings.warn(
            f"Legacy PRNGKey detected{(' in ' + context) if context else ''}. "
            "For explicit PRNG backend control use malthusjax.core.random.create_key() or jax.random.key()",
            DeprecationWarning,
            stacklevel=3,
        )
