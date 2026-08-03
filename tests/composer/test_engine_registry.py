"""Tests for the engine registry module."""

import pytest

from malthusjax.composer.engine_registry import (
    _ENGINE_REGISTRY,
    get_registry,
    list_available,
    register,
    register_table,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_ENGINE_REGISTRY)
    yield
    _ENGINE_REGISTRY.clear()
    _ENGINE_REGISTRY.update(saved)


# -- register / get_registry -------------------------------------------


def test_register_and_get():
    """Single engine can be registered and retrieved."""

    def dummy_factory(**kwargs):
        return {"type": "dummy", **kwargs}

    register("test_engine", dummy_factory, {"pop_size": 42}, override=True)

    reg = get_registry()
    assert "test_engine" in reg
    factory, defaults = reg["test_engine"]
    assert defaults == {"pop_size": 42}

    result = factory(pop_size=100)
    assert result == {"type": "dummy", "pop_size": 100}


def test_register_duplicate_raises():
    """Duplicate registration without override raises KeyError."""

    def factory_a(**kw):
        return "a"

    def factory_b(**kw):
        return "b"

    register("dup_engine", factory_a, override=True)

    with pytest.raises(KeyError, match="already registered"):
        register("dup_engine", factory_b)


def test_register_duplicate_with_override():
    """Duplicate registration with override=True replaces entry."""

    def factory_a(**kw):
        return "a"

    def factory_b(**kw):
        return "b"

    register("dup_engine", factory_a, override=True)
    register("dup_engine", factory_b, override=True)

    factory, _ = get_registry()["dup_engine"]
    assert factory() == "b"


# -- register_table ------------------------------------------------------


def test_register_table():
    """Bulk registration populates the registry."""

    def f1(**kw):
        return "f1"

    def f2(**kw):
        return "f2"

    register_table(
        [
            ("engine_a", f1, {"x": 1}),
            ("engine_b", f2, {"y": 2}),
        ],
        override=True,
    )

    reg = get_registry()
    assert "engine_a" in reg
    assert "engine_b" in reg
    assert reg["engine_a"][1] == {"x": 1}
    assert reg["engine_b"][1] == {"y": 2}


# -- get_registry returns copy -------------------------------------------


def test_get_registry_is_copy():
    """get_registry() returns a copy, not the internal dict."""
    reg = get_registry()
    reg["phantom"] = (lambda: None, {})
    assert "phantom" not in get_registry()


# -- list_available -------------------------------------------------------


def test_list_available_sorted():
    """list_available() returns sorted engine names."""
    register("zz_engine", lambda **kw: None, override=True)
    register("aa_engine", lambda **kw: None, override=True)

    names = list_available()
    assert names == sorted(names)
    assert "aa_engine" in names
    assert "zz_engine" in names


# -- defaults are optional -----------------------------------------------


def test_register_no_defaults():
    """Registration without defaults stores empty dict."""
    register("bare_engine", lambda **kw: "bare", override=True)
    _, defaults = get_registry()["bare_engine"]
    assert defaults == {}


# -- GA is pre-registered via engine/__init__.py --------------------------


def test_ga_registered_at_import():
    """Importing malthusjax.engine registers 'ga'."""
    import malthusjax.engine  # noqa: F401

    reg = get_registry()
    assert "ga" in reg
    factory, defaults = reg["ga"]
    assert "pop_size" in defaults
    assert callable(factory)
