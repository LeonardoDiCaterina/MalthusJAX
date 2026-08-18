"""Tests for the decorator-based registration API."""

from malthusjax.composer import (
    register_crossover,
    register_engine,
    register_fitness,
    register_genome,
    register_mutation,
    register_selection,
)
from malthusjax.composer._genome_registry import get_registry as get_genome_registry
from malthusjax.composer._registry import get_registry as get_operator_registry
from malthusjax.composer.engine_registry import get_registry as get_engine_registry


def test_register_selection():
    @register_selection("test_custom_selection", defaults={"k": 5})
    class CustomSelection:
        pass

    reg = get_operator_registry()
    assert "test_custom_selection" in reg
    factory, defaults = reg["test_custom_selection"]
    assert factory is CustomSelection
    assert defaults == {"k": 5}


def test_register_mutation():
    @register_mutation("test_custom_mutation")
    class CustomMutation:
        pass

    reg = get_operator_registry()
    assert "test_custom_mutation" in reg


def test_register_crossover():
    @register_crossover("test_custom_crossover")
    class CustomCrossover:
        pass

    reg = get_operator_registry()
    assert "test_custom_crossover" in reg


def test_register_fitness():
    @register_fitness("test_custom_fitness")
    class CustomFitness:
        pass

    reg = get_operator_registry()
    assert "test_custom_fitness" in reg


def test_register_genome():
    @register_genome("test_custom_genome")
    class CustomGenome:
        pass

    reg = get_genome_registry()
    assert "test_custom_genome" in reg


def test_register_engine():
    @register_engine("test_custom_engine")
    class CustomEngine:
        pass

    reg = get_engine_registry()
    assert "test_custom_engine" in reg


def test_decorators_allow_override():
    """Ensure that running the decorator twice on the same key doesn't crash.
    This is essential for Jupyter Notebook iterative development.
    """

    @register_selection("test_override_op", override=True)
    class FirstVersion:
        pass

    @register_selection("test_override_op", override=True)
    class SecondVersion:
        pass

    reg = get_operator_registry()
    factory, _ = reg["test_override_op"]
    assert factory is SecondVersion
