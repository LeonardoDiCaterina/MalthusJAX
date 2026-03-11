"""Lightweight utility used by the Composer pipeline system.

The registry simply associates string keys with factory callables that
produce pipeline nodes.  Designed to be minimal so it has no external
dependencies or lifecycle management.
"""

from typing import Any, Callable, Dict, List

NodeFactory = Callable[[Any, Dict[str, Any], Dict[str, Any]], Any]


class Registry:
    """Map short names to ``NodeFactory`` callables.

    The stored factories are expected to accept a PRNG *key*, a parameters
    dictionary and a dictionary of previously-built inputs.  This is the
    primitive used by :mod:`composer.pipeline` to construct a computation
    graph from declarative definitions.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, NodeFactory] = {}

    def register(self, name: str, factory: NodeFactory, override: bool = True) -> None:
        if not override and name in self._factories:
            raise KeyError(f"'{name}' already registered")
        self._factories[name] = factory

    def get(self, name: str) -> NodeFactory:
        try:
            return self._factories[name]
        except KeyError as e:
            raise KeyError(f"No factory registered under '{name}'") from e

    def list(self) -> List[str]:
        return list(self._factories.keys())
