"""Unit tests for malthusjax.core.random"""

import jax
import jax.random as jr
import jax.numpy as jnp
import pytest

from malthusjax.core.random import PRNGImpl, create_key, is_new_style_key, validate_key


def test_create_key_default_impl():
    k1 = create_key(42)
    k2 = create_key(42)
    assert isinstance(k1, jax.Array)
    assert isinstance(k2, jax.Array)
    # Deterministic
    assert jnp.allclose(jnp.asarray(k1), jnp.asarray(k2))


def test_create_key_each_impl(prng_impl):
    # prng_impl fixture will skip unsupported implementations
    k = create_key(0, impl=prng_impl)
    assert isinstance(k, jax.Array)


def test_create_key_deterministic():
    try:
        k1 = create_key(123, PRNGImpl.PHILOX)
        k2 = create_key(123, PRNGImpl.PHILOX)
    except ValueError:
        pytest.skip("PHILOX not supported by this JAX build")

    assert jnp.allclose(jnp.asarray(k1), jnp.asarray(k2))


def test_different_seeds_different_keys():
    try:
        k1 = create_key(1, PRNGImpl.PHILOX)
        k2 = create_key(2, PRNGImpl.PHILOX)
    except ValueError:
        pytest.skip("PHILOX not supported by this JAX build")

    assert not jnp.allclose(jnp.asarray(k1), jnp.asarray(k2))


def test_different_impls_different_keys():
    try:
        k1 = create_key(42, PRNGImpl.PHILOX)
        k2 = create_key(42, PRNGImpl.THREEFRY)
    except ValueError:
        pytest.skip("One of the impls not supported by this JAX build")

    # It's extremely likely they differ; assert non-equality
    assert not jnp.allclose(jnp.asarray(k1), jnp.asarray(k2))


def test_impl_propagates_through_split():
    try:
        master = create_key(7, PRNGImpl.PHILOX)
    except ValueError:
        pytest.skip("PHILOX not supported by this JAX build")

    subkeys = jr.split(master, 3)
    for k in subkeys:
        assert is_new_style_key(k)


def test_is_new_style_key_detection():
    legacy = jr.PRNGKey(42)
    assert not is_new_style_key(legacy)
    newk = create_key(42)
    assert is_new_style_key(newk)


def test_validate_key_warns_on_legacy():
    legacy = jr.PRNGKey(42)
    with pytest.warns(DeprecationWarning):
        validate_key(legacy, context="test")
