from typing import Any

import pytest

from malthusjax.composer._shared_registry import make_catalog_registry


def dummy_factory(**kwargs: Any) -> Any:
    return "dummy"


def test_make_catalog_registry_register():
    register, _, get_registry, list_available, _ = make_catalog_registry("TestEntity")

    # Test register
    register("test1", dummy_factory)
    reg = get_registry()
    assert "test1" in reg
    assert reg["test1"][0] is dummy_factory
    assert reg["test1"][1] == {}

    # Test defaults
    register("test2", dummy_factory, {"a": 1})
    reg = get_registry()
    assert reg["test2"][1] == {"a": 1}


def test_make_catalog_registry_duplicate_register():
    register, _, _, _, _ = make_catalog_registry("CustomEntity")

    register("test1", dummy_factory)

    # Should raise specific error message
    with pytest.raises(KeyError, match="CustomEntity 'test1' is already registered"):
        register("test1", dummy_factory)

    # Should succeed with override
    register("test1", dummy_factory, {"new": True}, override=True)


def test_make_catalog_registry_register_table():
    _, register_table, get_registry, list_available, _ = make_catalog_registry("TestEntity")

    entries = [
        ("t1", dummy_factory, {"a": 1}),
        ("t2", dummy_factory, {"b": 2}),
    ]
    register_table(entries)

    reg = get_registry()
    assert "t1" in reg
    assert "t2" in reg
    assert len(list_available()) == 2


def test_make_catalog_registry_get_registry_returns_copy():
    register, _, get_registry, _, _ = make_catalog_registry("TestEntity")

    register("test1", dummy_factory)
    reg = get_registry()

    # Mutate the returned copy
    reg["test1"] = ("hacked", {})

    # Ensure original is unchanged
    clean_reg = get_registry()
    assert clean_reg["test1"][0] is dummy_factory


def test_make_catalog_registry_list_available_sorted():
    register, _, _, list_available, _ = make_catalog_registry("TestEntity")

    register("z", dummy_factory)
    register("a", dummy_factory)
    register("m", dummy_factory)

    available = list_available()
    assert available == ["a", "m", "z"]
