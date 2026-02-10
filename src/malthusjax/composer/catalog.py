"""Operator catalog — string-spec → operator instance resolution.

The catalog is a thin wrapper around the global :mod:`_registry`.  Operator
sub-packages register themselves at import time (see each package's
``__init__.py``).  ``OperatorCatalog`` triggers those imports on first
construction and then delegates ``get()`` to the registry.

Public API (fully backward-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``catalog.get("tournament:num_selections=50,tournament_size=3")``
* ``catalog.parse_spec("gaussian:mutation_rate=0.1")``
* ``catalog.register("custom", factory)``
* ``catalog.list_available()``
* ``catalog.get_help("tournament")``

Examples::

    catalog = OperatorCatalog()
    catalog.get("tournament")                       # default params
    catalog.get("tournament:num_selections=50")      # override defaults
    catalog.get("gaussian:mutation_rate=0.1")
    catalog.get("blend:alpha=0.5")
    catalog.get("sphere:dim=10")
    catalog.get("bbob:fn_name=rastrigin,dim=5")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from ._registry import get_registry, register as _registry_register


def _ensure_registered() -> None:
    """Force-import every operator sub-package so that their
    ``_register_*()`` calls run and populate the global registry.

    Idempotent — repeated calls are cheap (Python caches imports).
    """
    import malthusjax.operators.selection  # noqa: F401
    import malthusjax.operators.crossover  # noqa: F401
    import malthusjax.operators.mutation  # noqa: F401
    import malthusjax.core.fitness  # noqa: F401


class OperatorCatalog:
    """Catalog for creating operators from string specifications.

    Supports format: ``"operator_type:param1=value1,param2=value2"``

    Available Selection Operators:
        - tournament: Tournament selection
        - roulette: Roulette wheel selection
        - elite_pool: Elite pool selection (deterministic)

    Available Real-Valued Crossover Operators:
        - blend: Blend crossover (BLX)
        - blend_injection: Blend crossover (injection variant)
        - simulated_binary: Simulated Binary Crossover (SBX)
        - simulated_binary_injection: SBX (injection variant)
        - binomial: Binomial crossover
        - binomial_injection: Binomial crossover (injection variant)
        - uniform_real: Uniform crossover for real genomes
        - uniform_real_injection: Uniform crossover (injection variant)
        - evosax_uniform_crossover: Evosax uniform crossover wrapper

    Available Binary Crossover Operators:
        - uniform_binary: Uniform crossover for binary genomes
        - single_point: Single-point crossover

    Available Real-Valued Mutation Operators:
        - gaussian: Gaussian (normal) mutation
        - gaussian_injection: Gaussian mutation (injection variant)
        - ball: Ball mutation
        - ball_injection: Ball mutation (injection variant)
        - polynomial: Polynomial mutation
        - polynomial_injection: Polynomial mutation (injection variant)
        - evosax_gaussian: Evosax Gaussian mutation wrapper

    Available Binary Mutation Operators:
        - bitflip: Bit-flip mutation
        - scramble: Scramble mutation
        - swap: Swap mutation

    Available Fitness Evaluators:
        - sphere: Sphere optimization (maximization)
        - rastrigin: Rastrigin optimization (maximization)
        - knapsack: Knapsack problem
        - bbob: General BBOB function family
        - sphere_minimize: Sphere minimization
        - sphere_maximize: Sphere maximization
        - griewank: Griewank function
        - binary_sum: Binary sum (OneMax)

    Available Evosax Strategies (use via Composer backend="evosax"):
        - evosax_simplega: Simple Genetic Algorithm
        - evosax_mr15: MR15 Genetic Algorithm
        - evosax_de: Differential Evolution

    Examples::

        catalog.get("tournament")  # Default parameters
        catalog.get("tournament:num_selections=50,tournament_size=3")
        catalog.get("gaussian:mutation_rate=0.1")
        catalog.get("blend:alpha=0.5")
        catalog.get("sphere:dim=10")
    """

    def __init__(self) -> None:
        _ensure_registered()
        self._registry = get_registry()

        # Evosax strategy name helpers (return plain strings, not operators)
        self._evosax_strategies: Dict[str, str] = {
            "evosax_simplega": "SimpleGA",
            "evosax_mr15": "MR15_GA",
            "evosax_de": "DifferentialEvolution",
        }

    # ------------------------------------------------------------------
    # Spec parsing
    # ------------------------------------------------------------------

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        """Parse operator specification string.

        Args:
            spec: String like ``"operator_type:param1=value1,param2=value2"``

        Returns:
            Tuple of ``(operator_type, params_dict)``.

        Raises:
            ValueError: If *spec* format is invalid.
        """
        spec = spec.strip()
        if not spec:
            raise ValueError("Empty operator specification")

        if ":" not in spec:
            return spec, {}

        operator_type, params_str = spec.split(":", 1)
        operator_type = operator_type.strip()
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

        return operator_type, params

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

    def get(self, spec: str) -> Any:
        """Get configured operator instance from specification string.

        Args:
            spec: e.g. ``"tournament:num_selections=5,tournament_size=3"``

        Returns:
            Configured operator instance.

        Raises:
            KeyError: If operator type is not registered.
            ValueError: If parameters are invalid for the operator.
        """
        operator_type, user_params = self.parse_spec(spec)

        # Evosax strategy helpers (return plain name strings)
        if operator_type in self._evosax_strategies:
            return self._evosax_strategies[operator_type]

        if operator_type not in self._registry:
            available = sorted(
                list(self._registry.keys()) + list(self._evosax_strategies.keys())
            )
            raise KeyError(
                f"Unknown operator type: '{operator_type}'. Available: {available}"
            )

        factory, defaults = self._registry[operator_type]
        merged = {**defaults, **user_params}

        try:
            return factory(**merged)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for '{operator_type}': {e}") from e

    # ------------------------------------------------------------------
    # Introspection & extension
    # ------------------------------------------------------------------

    def register(self, operator_type: str, factory: Callable, override: bool = False) -> None:
        """Register a new operator type at runtime.

        Args:
            operator_type: String name for the operator.
            factory: Callable that creates operator instances from ``**kwargs``.
            override: Whether to override existing registrations.
        """
        if not override and (
            operator_type in self._registry or operator_type in self._evosax_strategies
        ):
            raise KeyError(f"Operator type '{operator_type}' already registered")

        # Also push into global registry so subsequent OperatorCatalog()
        # instances see the new entry.
        _registry_register(operator_type, factory, override=True)
        self._registry[operator_type] = (factory, {})

    def list_available(self) -> List[str]:
        """Return sorted list of all registered operator keys."""
        return sorted(
            list(self._registry.keys()) + list(self._evosax_strategies.keys())
        )

    def get_help(self, operator_type: str) -> str:
        """Get help string for operator type."""
        if operator_type not in self._registry and operator_type not in self._evosax_strategies:
            return f"Unknown operator: {operator_type}"
        return f"{operator_type}:param1=value1,param2=value2,..."


DEFAULT_CATALOG = OperatorCatalog()
