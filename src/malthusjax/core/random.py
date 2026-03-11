"""
PRNG key factory and helpers for MalthusJAX.

Centralizes creation of typed JAX PRNG keys and helper utilities for
legacy/compat detection and warnings.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional, cast

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

# Mapping of short user-facing names to enum values
_PRNG_ALIASES: dict[str, PRNGImpl] = {
    "threefry": PRNGImpl.THREEFRY,
    "threefry2x32": PRNGImpl.THREEFRY,
    "philox": PRNGImpl.PHILOX,
    "philox4x32_10": PRNGImpl.PHILOX,
    "rbg": PRNGImpl.RBG,
    "unsafe_rbg": PRNGImpl.UNSAFE_RBG,
}


def resolve_prng_impl(name: str | PRNGImpl | None) -> PRNGImpl:
    """Resolve a user-facing PRNG name to a :class:`PRNGImpl` enum value.

    Accepts short names (``"threefry"``, ``"philox"``), full JAX backend
    strings (``"threefry2x32"``), ``PRNGImpl`` members, or ``None``
    (falls back to :data:`DEFAULT_IMPL`).

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name is None:
        return DEFAULT_IMPL
    if isinstance(name, PRNGImpl):
        return name
    key = name.lower().strip()
    if key in _PRNG_ALIASES:
        return _PRNG_ALIASES[key]
    raise ValueError(
        f"Unknown PRNG implementation {name!r}. Choose from: {sorted(_PRNG_ALIASES.keys())}"
    )


def create_key(seed: int, impl: Optional[PRNGImpl] = None) -> jax.Array:
    """Create a PRNG key using the requested implementation.

    Falls back to ``jax.random.PRNGKey`` if the new ``jax.random.key`` API
    is not available in the installed JAX version.

    The returned key encodes the chosen backend and can be split or folded
    to propagate the implementation information.
    """
    impl = impl or DEFAULT_IMPL

    key_ctor = getattr(jax.random, "key", None)
    if callable(key_ctor):
        try:
            return cast(jax.Array, key_ctor(seed, impl=impl.value))
        except TypeError:
            warnings.warn(
                "jax.random.key exists but does not accept 'impl='; falling back to PRNGKey",
                RuntimeWarning,
            )

    warnings.warn(
        "Using legacy jax.random.PRNGKey fallback; consider upgrading JAX to use 'jax.random.key'",
        DeprecationWarning,
    )
    return jax.random.PRNGKey(seed)


def _is_legacy_prngkey(key: jax.Array) -> bool:
    """Heuristic to detect legacy PRNGKey (uint32[2]) used by older JAX.

    Legacy keys produced by ``jax.random.PRNGKey`` are uint32 arrays of length 2.
    New-style typed keys typically have dtype != uint32 or length != 2
    (e.g., Philox uses a larger key).
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
            (
                f"Legacy PRNGKey detected{' in ' + context if context else ''}. "
                "For explicit PRNG backend control use "
                "malthusjax.core.random.create_key() or jax.random.key()"
            ),
            DeprecationWarning,
            stacklevel=3,
        )
