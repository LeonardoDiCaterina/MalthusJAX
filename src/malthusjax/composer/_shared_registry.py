from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, Tuple


class RegisterFunc(Protocol):
    def __call__(
        self,
        name: str,
        factory: Callable[..., Any],
        defaults: Dict[str, Any] | None = None,
        *,
        override: bool = False,
    ) -> None: ...


class RegisterTableFunc(Protocol):
    def __call__(
        self,
        entries: list[Tuple[str, Callable[..., Any], Dict[str, Any]]],
        *,
        override: bool = False,
    ) -> None: ...


def make_catalog_registry(
    entity_name: str,
) -> Tuple[
    RegisterFunc,
    RegisterTableFunc,
    Callable[[], Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]]],
    Callable[[], List[str]],
    Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]],
]:
    """Factory to create a standard registry module API.

    Returns the four standard functions used by composer registries:
    register, register_table, get_registry, list_available, and the underlying dict.
    """
    _registry: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {}

    def register(
        name: str,
        factory: Callable[..., Any],
        defaults: Dict[str, Any] | None = None,
        *,
        override: bool = False,
    ) -> None:
        if not override and name in _registry:
            raise KeyError(f"{entity_name} '{name}' is already registered")
        _registry[name] = (factory, defaults or {})

    def register_table(
        entries: list[Tuple[str, Callable[..., Any], Dict[str, Any]]],
        *,
        override: bool = False,
    ) -> None:
        for name, factory, defaults in entries:
            register(name, factory, defaults, override=override)

    def get_registry() -> Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]]:
        """Return a **copy** of the current registry."""
        return dict(_registry)

    def list_available() -> List[str]:
        """Return sorted list of registered names."""
        return sorted(_registry.keys())

    return register, register_table, get_registry, list_available, _registry
