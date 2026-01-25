from typing import Any, Callable, Dict, List

NodeFactory = Callable[[Any, Dict[str, Any], Dict[str, Any]], Any]


class Registry:
    """Tiny registry for mapping a short name -> factory callable."""

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
