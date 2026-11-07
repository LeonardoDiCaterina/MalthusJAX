"""
PRNG key factory and helpers for MalthusJAX.

Centralizes creation of typed JAX PRNG keys and helper utilities for
legacy/compat detection and warnings.
"""

from __future__ import annotations

import math
import warnings
from enum import Enum
from typing import Any, Optional, cast

import jax
import jax.lax as lax
import jax.numpy as jnp


_PHILOX_REGISTERED = False


def _register_philox_prng() -> None:
    """Register the native XLA Philox-4x32-10 PRNG implementation with JAX if not present."""
    global _PHILOX_REGISTERED
    if _PHILOX_REGISTERED:
        return
    try:
        try:
            import jax._src.prng as _prng
        except (ImportError, ModuleNotFoundError):
            try:
                import jax._src.random as _prng
            except (ImportError, ModuleNotFoundError):
                _prng = None

        if _prng is None or not hasattr(_prng, "prngs"):
            return

        import jax.extend.random as jext_rnd

        if "philox4x32" in _prng.prngs:
            return

        def _xla_philox_seed(seed: jax.Array) -> jax.Array:
            halfkey = _prng.threefry_seed(seed)
            return jnp.concatenate(
                [halfkey, halfkey ^ jnp.array([0x9E3779B9, 0xBB67AE85], dtype=jnp.uint32)]
            )

        def _xla_philox_split(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            num = math.prod(shape)
            _, keys = lax.rng_bit_generator(
                key,
                (num, 4),
                dtype=jnp.uint32,
                algorithm=lax.RandomAlgorithm.RNG_PHILOX,
            )
            return cast(jax.Array, keys.reshape(*shape, 4))

        def _xla_philox_fold_in(key: jax.Array, data: Any) -> jax.Array:
            data_u32 = jnp.asarray(data, dtype=jnp.uint32)
            mixed = key ^ jnp.stack(
                [data_u32, jnp.uint32(0x9E3779B9), jnp.uint32(0), jnp.uint32(0)]
            )
            _, bits = lax.rng_bit_generator(
                mixed,
                (4,),
                dtype=jnp.uint32,
                algorithm=lax.RandomAlgorithm.RNG_PHILOX,
            )
            return cast(jax.Array, bits)

        def _xla_philox_random_bits(
            key: jax.Array, bit_width: int, shape: tuple[int, ...]
        ) -> jax.Array:
            uint_dtypes = {8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}
            if not (key.shape == (4,) and key.dtype == jnp.dtype("uint32")):
                raise TypeError("_xla_philox_random_bits got invalid prng key.")
            if bit_width not in uint_dtypes:
                raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
            _, bits = lax.rng_bit_generator(
                key,
                shape,
                dtype=uint_dtypes[bit_width],
                algorithm=lax.RandomAlgorithm.RNG_PHILOX,
            )
            return cast(jax.Array, bits)

        spec = jext_rnd.define_prng_impl(
            key_shape=(4,),
            seed=_xla_philox_seed,
            split=_xla_philox_split,
            random_bits=_xla_philox_random_bits,
            fold_in=cast(Any, _xla_philox_fold_in),
            name="philox4x32",
            tag="phx",
        )
        _prng.register_prng(getattr(spec, "_impl", spec))
        _PHILOX_REGISTERED = True
    except Exception as e:
        warnings.warn(f"Failed to register native XLA Philox PRNG: {e}", RuntimeWarning)


class PRNGImpl(str, Enum):
    """Available JAX PRNG implementations.

    Mirrors the pattern used by other enums in the codebase.
    """

    THREEFRY = "threefry2x32"
    PHILOX = "philox4x32"
    RBG = "rbg"
    UNSAFE_RBG = "unsafe_rbg"


DEFAULT_IMPL = PRNGImpl.THREEFRY

# Mapping of short user-facing names to enum values
_PRNG_ALIASES: dict[str, PRNGImpl] = {
    "threefry": PRNGImpl.THREEFRY,
    "threefry2x32": PRNGImpl.THREEFRY,
    "philox": PRNGImpl.PHILOX,
    "philox4x32": PRNGImpl.PHILOX,
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
    if impl == PRNGImpl.PHILOX:
        _register_philox_prng()

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

    Legacy keys produced by ``jax.random.PRNGKey`` are uint32 arrays with trailing dimension 2
    (or a single key of shape (2,) / size 2).
    New-style typed keys typically have dtype != uint32 (e.g. key<fry> or key<phx>).
    """
    try:
        arr = jnp.asarray(key)
        if arr.dtype != jnp.uint32:
            return False
        if arr.ndim == 0:
            return False
        if arr.ndim == 1:
            return arr.size == 2
        return arr.shape[-1] == 2
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
