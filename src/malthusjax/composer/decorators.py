"""Decorators for extending MalthusJAX components.

This module provides an elegant, user-friendly API for registering custom
operators, engines, fitness functions, and genomes into the global MalthusJAX
catalog.

By default, these decorators use `override=False` to prevent accidental
collisions. To re-evaluate in interactive environments like Jupyter notebooks
without raising `KeyError` exceptions, pass `override=True`.
"""

from typing import Any, Callable, Dict, Optional

from malthusjax.composer._genome_registry import register as _register_genome
from malthusjax.composer._registry import register as _register_operator
from malthusjax.composer.engine_registry import register as _register_engine


def _operator_decorator(
    name: str, defaults: Optional[Dict[str, Any]] = None, override: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Internal decorator factory for operator registration."""

    def wrapper(cls_or_func: Callable[..., Any]) -> Callable[..., Any]:
        _register_operator(name, cls_or_func, defaults, override=override)
        return cls_or_func

    return wrapper


def _engine_decorator(
    name: str, defaults: Optional[Dict[str, Any]] = None, override: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Internal decorator factory for engine registration."""

    def wrapper(cls_or_func: Callable[..., Any]) -> Callable[..., Any]:
        _register_engine(name, cls_or_func, defaults, override=override)
        return cls_or_func

    return wrapper


def _genome_decorator(
    name: str, defaults: Optional[Dict[str, Any]] = None, override: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Internal decorator factory for genome registration."""

    def wrapper(cls_or_func: Callable[..., Any]) -> Callable[..., Any]:
        _register_genome(name, cls_or_func, defaults, override=override)
        return cls_or_func

    return wrapper


# Semantic Aliases for public use
register_selection = _operator_decorator
register_mutation = _operator_decorator
register_crossover = _operator_decorator
register_fitness = _operator_decorator

register_engine = _engine_decorator
register_genome = _genome_decorator
