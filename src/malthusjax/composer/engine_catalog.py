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

    # ------------------------------------------------------------------
    # Spec parsing
    # ------------------------------------------------------------------

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        """Parse engine specification string.

        Args:
            spec: String like ``"ga"`` or ``"nsga2:pop_size=100,generations=200"``

        Returns:
            Tuple of ``(engine_name, params_dict)``.

        Raises:
            ValueError: If *spec* format is invalid.
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

        # Strip quotes
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        return value_str

    # ------------------------------------------------------------------
    # Instance creation
    # ------------------------------------------------------------------

    def get(
        self,
        spec: str,
        evaluator: Any,
        selection: Any,
        crossover: Any,
        mutation: Any,
        **kwargs: Any,
    ) -> Any:
        """Get configured engine from specification string.

        The factory receives the resolved operator *instances* plus any
        additional keyword arguments (merged from registry defaults,
        spec-level overrides, and explicit ``**kwargs``).

        Args:
            spec: e.g. ``"ga"`` or ``"ga:pop_size=100"``
            evaluator: Fitness evaluator instance (e.g. ``SphereEvaluator``)
            selection: Selection operator instance
            crossover: Crossover operator instance
            mutation: Mutation operator instance
            **kwargs: Additional engine params (``pop_size``, ``generations``,
                ``genome_type``, ``bounds``, ``elitism``, ``prng_impl``, etc.)

        Returns:
            Engine instance satisfying the
            :class:`~malthusjax.benchmarking.runner.Engine` protocol.

        Raises:
            KeyError: If engine name is not registered.
            ValueError: If parameters are invalid for the engine.
        """
        engine_name, spec_params = self.parse_spec(spec)

        if engine_name not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Unknown engine '{engine_name}'. Available: [{available}]"
            )

        factory, defaults = self._registry[engine_name]

        # Merge precedence: defaults < spec_params < explicit kwargs
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

    # ------------------------------------------------------------------
    # Introspection & extension
    # ------------------------------------------------------------------

    def register(
        self,
        engine_name: str,
        factory: Callable[..., Any],
        defaults: Dict[str, Any] | None = None,
        override: bool = False,
    ) -> None:
        """Register a new engine type at runtime.

        Args:
            engine_name: String name for the engine (e.g. ``"nsga2"``).
            factory: Callable that creates engine instances.
            defaults: Default kwargs for the factory.
            override: Whether to override existing registrations.
        """
        if not override and engine_name in self._registry:
            raise KeyError(f"Engine '{engine_name}' is already registered")

        # Also push into global registry so subsequent EngineRegistry()
        # instances see the new entry.
        _engine_register(engine_name, factory, defaults, override=True)
        self._registry[engine_name] = (factory, defaults or {})

    def list_available(self) -> List[str]:
        """Return sorted list of all registered engine names."""
        return sorted(self._registry.keys())

    def get_help(self, engine_name: str) -> str:
        """Get help string for an engine type.

        Args:
            engine_name: Registered engine name.

        Returns:
            Formatted help string with docstring and defaults.

        Raises:
            KeyError: If engine name is not registered.
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
