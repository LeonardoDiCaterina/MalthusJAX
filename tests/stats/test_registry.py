import pytest

from malthusjax.stats.registry import (
    get_correction,
    get_diagnostic,
    get_effect,
    get_test,
    register_correction,
    register_diagnostic,
    register_effect,
    register_test,
)


def test_register_and_get():
    def dummy_test():
        pass

    register_test("dummy", dummy_test)
    assert get_test("dummy") is dummy_test


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="Unknown test"):
        get_test("nonexistent")


def test_overwrite_registration():
    def test1():
        pass

    def test2():
        pass

    register_effect("eff", test1)
    register_effect("eff", test2)
    assert get_effect("eff") is test2


def test_diagnostic_registry():
    def d():
        pass

    register_diagnostic("d", d)
    assert get_diagnostic("d") is d


def test_correction_registry():
    def c():
        pass

    register_correction("c", c)
    assert get_correction("c") is c
