"""Engine Catalog — string-spec → engine instance resolution.

The catalog is a thin wrapper around the global :mod:`engine_registry`.
Engine modules register themselves at import time (see
``engine/__init__.py``).  ``EngineRegistry`` triggers those imports on
first construction and then delegates ``get()`` to the registry.

Public API
~~~~~~~~~~

* ``catalog.get("ga", evaluator=ev, selection=sel, crossover=cx, mutation=mut)``
* ``catalog.get("ga:pop_size=200", ...)``
* ``catalog.parse_spec("nsga2:pop_size=100,generations=200")``
* ``catalog.list_available()``
* ``catalog.get_help("ga")``
* ``catalog.register("custom", factory)``

Examples::

    registry = EngineRegistry()
    engine = registry.get(
        "ga:pop_size=100",
        evaluator=sphere_eval,
        selection=tournament,
        crossover=blend,
        mutation=gaussian,
        generations=200,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from .engine_registry import get_registry
from .engine_registry import register as _engine_register


def _ensure_engines_registered() -> None:
    """Force-import the engine package so that its
    ``_register_engines()`` call runs and populates the global registry.

    Idempotent — repeated calls are cheap (Python caches imports).
    """
    import malthusjax.engine  # noqa: F401


class EngineRegistry:
    """Catalog for resolving engine specifications.

    Supports format: ``"engine_name:param1=value1,param2=value2"``

    Available Engines:
        - ga: Standard genetic algorithm (GeneticEngine)

    Examples::

        registry = EngineRegistry()
        engine = registry.get("ga", evaluator=ev, selection=sel,
                              crossover=cx, mutation=mut)
        engine = registry.get("ga:pop_size=200,elitism=4", ...)
    """

    def __init__(self) -> None:
        _ensure_engines_registered()
        self._registry = get_registry()

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        """Parse an engine specification such as ``"ga"`` or
        ``"nsga2:pop_size=100,generations=200"`` and return the
        ``(engine_name, params_dict)`` tuple.  A ``ValueError`` is raised when
        the string cannot be interpreted.
        """
        spec = spec.strip()
        if not spec:
            raise ValueError("Empty engine specification")

        if ":" not in spec:
            return spec, {}

        engine_name, params_str = spec.split(":", 1)
        engine_name = engine_name.strip()
        params: Dict[str, Any] = {}

        if params_str.strip():
            for param_pair in params_str.split(","):
                param_pair = param_pair.strip()
                if "=" not in param_pair:
                    raise ValueError(
                        f"Invalid parameter format: '{param_pair}'. Expected 'key=value'"
                    )
                key, value = param_pair.split("=", 1)
                params[key.strip()] = self._convert_value(value.strip())

        return engine_name, params

    @staticmethod
    def _convert_value(value_str: str) -> Union[int, float, str, bool]:
        """Convert string value to appropriate Python type."""
        value_str = value_str.strip()

        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        try:
            return int(value_str)
        except ValueError:
            pass

        try:
            return float(value_str)
        except ValueError:
            pass

        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        return value_str

    def get(
        self,
        spec: str,
        evaluator: Any,
        selection: Any,
        crossover: Any,
        mutation: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve *spec* to a concrete engine instance.

        The specification string and any ``**kwargs`` are merged with registry
        defaults before invoking the underlying factory.  The caller must
        supply already-resolved operator instances for evaluator,
        selection, crossover and mutation; all other keyword parameters are
        forwarded transparently.  Errors during construction are re‑raised as
        ``KeyError`` (unknown engine name) or ``ValueError`` (invalid
        parameters).
        """
        engine_name, spec_params = self.parse_spec(spec)

        if engine_name not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Unknown engine '{engine_name}'. Available: [{available}]"
            )

        factory, defaults = self._registry[engine_name]
        merged_params = {**defaults, **spec_params, **kwargs}

        try:
            return factory(
                evaluator=evaluator,
                selection=selection,
                crossover=crossover,
                mutation=mutation,
                **merged_params,
            )
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for engine '{engine_name}': {e}"
            ) from e

    def register(
        self,
        engine_name: str,
        factory: Callable[..., Any],
        defaults: Dict[str, Any] | None = None,
        override: bool = False,
    ) -> None:
        """Register a new engine type in the catalog.

        *engine_name* gives the string key (e.g. ``"nsga2"``) and *factory*
        is a callable that constructs instances when invoked.  Optional
        *defaults* provide baseline parameters; set *override* if you wish to
        replace an existing registration.
        """
        if not override and engine_name in self._registry:
            raise KeyError(f"Engine '{engine_name}' is already registered")

        _engine_register(engine_name, factory, defaults, override=True)
        self._registry[engine_name] = (factory, defaults or {})

    def list_available(self) -> List[str]:
        """Return sorted list of all registered engine names."""
        return sorted(self._registry.keys())

    def get_help(self, engine_name: str) -> str:
        """Return a formatted help string for a registered engine.

        The output includes the engine's docstring and its default parameters.
        A ``KeyError`` is raised if *engine_name* isn't present in the catalog.
        """
        if engine_name not in self._registry:
            raise KeyError(f"Unknown engine: '{engine_name}'")

        factory, defaults = self._registry[engine_name]
        doc = factory.__doc__ or "No documentation available."
        defaults_str = ", ".join(f"{k}={v}" for k, v in defaults.items())

        return (
            f"{engine_name}\n"
            f"{'-' * len(engine_name)}\n\n"
            f"{doc}\n\n"
            f"Defaults: {defaults_str if defaults_str else '(none)'}"
        )
