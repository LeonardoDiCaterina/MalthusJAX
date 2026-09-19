"""Targeted coverage tests for core/logger.py and core/random.py."""

import logging
from unittest.mock import MagicMock, patch
import pytest

from malthusjax.core.logger import (
    MalthusColorFormatter,
    MalthusJSONFormatter,
    _host_log_nan_anomaly,
    _host_log_step,
    _init_from_env,
    _resolve_level,
)
from malthusjax.core.random import (
    DEFAULT_IMPL,
    PRNGImpl,
    _is_legacy_prngkey,
    create_key,
    resolve_prng_impl,
    validate_key,
)


def test_malthus_color_formatter():
    formatter = MalthusColorFormatter(use_color=True, show_timestamps=True)

    # Record with exact root name "malthusjax"
    rec_root = logging.LogRecord(
        name="malthusjax",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Root test",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(rec_root)
    assert "[core]" in formatted

    # Record with exception info
    try:
        raise ValueError("test exception")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    rec_exc = logging.LogRecord(
        name="malthusjax.sub",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted_exc = formatter.format(rec_exc)
    assert "ValueError: test exception" in formatted_exc


def test_malthus_json_formatter():
    formatter = MalthusJSONFormatter()

    class Unserializable:
        def __str__(self):
            return "unserializable_obj"

    rec = logging.LogRecord(
        name="malthusjax.runner",
        level=logging.WARNING,
        pathname="test.py",
        lineno=20,
        msg="JSON test",
        args=(),
        exc_info=None,
    )
    rec.custom_data = Unserializable()
    res = formatter.format(rec)
    assert "unserializable_obj" in res

    # With exception
    try:
        raise RuntimeError("json exc")
    except RuntimeError:
        import sys

        rec.exc_info = sys.exc_info()

    res_exc = formatter.format(rec)
    assert "json exc" in res_exc


def test_resolve_level():
    assert _resolve_level(logging.DEBUG) == logging.DEBUG
    assert _resolve_level("warning") == logging.WARNING
    with pytest.raises(ValueError, match="Invalid log level"):
        _resolve_level("UNKNOWN_XYZ")


def test_host_callbacks():
    class BadInt:
        def __int__(self):
            raise TypeError("bad int")

        def __str__(self):
            return "bad_int"

    class BadFloat:
        def __float__(self):
            raise ValueError("bad float")

        def __str__(self):
            return "bad_float"

    # Step callback with conversion failures
    _host_log_step(
        gen=BadInt(),
        best_fitness=BadFloat(),
        mean_fitness=BadFloat(),
        logger_name="malthusjax.test",
    )

    # Anomaly callback with conversion failures
    _host_log_nan_anomaly(
        gen=BadInt(),
        val=BadFloat(),
        metric_name="test_nan",
        logger_name="malthusjax.anomaly.test",
    )


def test_init_from_env():
    with patch.dict("os.environ", {"MALTHUSJAX_LOG_LEVEL": "INVALID_NAME"}):
        _init_from_env()  # Should silently pass without crashing


def test_resolve_prng_impl():
    assert resolve_prng_impl(None) == DEFAULT_IMPL
    assert resolve_prng_impl(PRNGImpl.THREEFRY) == PRNGImpl.THREEFRY
    assert resolve_prng_impl("threefry") == PRNGImpl.THREEFRY
    with pytest.raises(ValueError, match="Unknown PRNG implementation"):
        resolve_prng_impl("quantum_prng_unknown")


def test_create_key_fallback():
    # Test jax.random.key TypeError fallback
    mock_key = MagicMock(side_effect=TypeError("No impl arg"))
    with patch("jax.random.key", mock_key):
        with pytest.warns(RuntimeWarning, match="does not accept 'impl='"):
            key = create_key(123, PRNGImpl.THREEFRY)
            assert key is not None


def test_is_legacy_prngkey_and_validate():
    import jax

    # Exception in asarray
    class BadArray:
        pass

    assert _is_legacy_prngkey(BadArray()) is False

    # Legacy key warning in validate_key
    legacy_k = jax.random.PRNGKey(42)
    with pytest.warns(DeprecationWarning, match="Legacy PRNGKey detected in test_context"):
        validate_key(legacy_k, context="test_context")
