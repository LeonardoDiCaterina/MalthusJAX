import pytest

from malthusjax.composer import _genome_registry as genome_registry


def test_register_and_get_registry_copy():
    backup = genome_registry._GENOME_REGISTRY.copy()
    genome_registry._GENOME_REGISTRY.clear()
    try:

        def factory(**kwargs):
            return kwargs

        genome_registry.register("test", factory, {"foo": 1})
        registry_copy = genome_registry.get_registry()

        assert "test" in registry_copy
        assert registry_copy["test"][0] is factory
        assert registry_copy["test"][1] == {"foo": 1}

        # ensure get_registry returns a copy, not the live registry
        registry_copy["test"] = (lambda **kwargs: None, {})
        assert genome_registry._GENOME_REGISTRY["test"][0] is factory
    finally:
        genome_registry._GENOME_REGISTRY.clear()
        genome_registry._GENOME_REGISTRY.update(backup)


def test_register_duplicate_raises_key_error():
    backup = genome_registry._GENOME_REGISTRY.copy()
    genome_registry._GENOME_REGISTRY.clear()
    try:
        genome_registry.register("dup", lambda **kwargs: None)
        with pytest.raises(KeyError, match="already registered"):
            genome_registry.register("dup", lambda **kwargs: None)
    finally:
        genome_registry._GENOME_REGISTRY.clear()
        genome_registry._GENOME_REGISTRY.update(backup)


def test_register_table_with_override():
    backup = genome_registry._GENOME_REGISTRY.copy()
    genome_registry._GENOME_REGISTRY.clear()
    try:
        genome_registry.register("one", lambda **kwargs: 1)
        genome_registry.register_table(
            [
                ("one", lambda **kwargs: 2, {}),
                ("two", lambda **kwargs: 3, {"x": 1}),
            ],
            override=True,
        )

        registry_copy = genome_registry.get_registry()
        assert registry_copy["one"][0]() == 2
        assert registry_copy["two"][1] == {"x": 1}
    finally:
        genome_registry._GENOME_REGISTRY.clear()
        genome_registry._GENOME_REGISTRY.update(backup)
