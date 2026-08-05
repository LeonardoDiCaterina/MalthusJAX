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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ._registry import get_registry
from ._registry import register as _registry_register


def _ensure_registered() -> None:
    """Force-import every operator sub-package so that their
    ``_register_*()`` calls run and populate the global registry.

    Idempotent — repeated calls are cheap (Python caches imports).
    """
    import malthusjax.core.fitness  # noqa: F401
    import malthusjax.operators.crossover  # noqa: F401
    import malthusjax.operators.emitters  # noqa: F401
    import malthusjax.operators.mutation  # noqa: F401
    import malthusjax.operators.selection  # noqa: F401


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

        self._evosax_strategies: Dict[str, str] = {
            "evosax_simplega": "SimpleGA",
            "evosax_mr15": "MR15_GA",
            "evosax_de": "DifferentialEvolution",
        }

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        """Parse an operator specification such as
        ``"operator_type:param1=value1,param2=value2"`` and return a
        tuple ``(operator_type, params_dict)``.  A ``ValueError`` is raised if
        the string is malformed.
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

    def get(self, spec: str, data_registry: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Resolve *spec* to a configured operator instance.  The
        spec string may include comma-separated parameter overrides.  A
        ``KeyError`` is raised for unknown operator types and a
        ``ValueError`` for invalid parameter combinations.
        """
        operator_type, user_params = self.parse_spec(spec)

        merged_params = {**user_params, **kwargs}

        if data_registry is not None and "data_id" in merged_params:
            data_id = merged_params.pop("data_id")
            if data_id not in data_registry:
                raise KeyError(f"Data ID '{data_id}' not in registry")
            merged_params["_resolved_data"] = data_registry[data_id]

        if operator_type in self._evosax_strategies:
            return self._evosax_strategies[operator_type]

        if operator_type not in self._registry:
            available = sorted(list(self._registry.keys()) + list(self._evosax_strategies.keys()))
            raise KeyError(f"Unknown operator type: '{operator_type}'. Available: {available}")

        factory, default_params = self._registry[operator_type]

        merged = default_params.copy()
        merged.update(merged_params)

        import inspect

        factory_sig = inspect.signature(factory)
        valid_keys = factory_sig.parameters.keys()
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in factory_sig.parameters.values()
        )

        invalid_keys = [k for k in merged if k not in valid_keys]
        if invalid_keys and not has_kwargs:
            raise ValueError(
                f"Invalid parameters for '{operator_type}': unexpected keyword argument(s) {invalid_keys}"
            )

        filtered_merged = {k: v for k, v in merged.items() if k in valid_keys or has_kwargs}

        try:
            return factory(**filtered_merged)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for '{operator_type}': {e}") from e

    def register(
        self,
        operator_type: str,
        factory: Callable[..., Any],
        override: bool = False,
    ) -> None:
        """Register a new operator type in the catalog.  Supply a string
        key and a factory callable which accepts ``**kwargs``.  Set *override*
        to ``True`` to replace an existing entry.
        """
        if not override and (
            operator_type in self._registry or operator_type in self._evosax_strategies
        ):
            raise KeyError(f"Operator type '{operator_type}' already registered")

        _registry_register(operator_type, factory, override=True)
        self._registry[operator_type] = (factory, {})

    def list_available(self) -> List[str]:
        """Return sorted list of all registered operator keys."""
        return sorted(list(self._registry.keys()) + list(self._evosax_strategies.keys()))

    def get_help(self, operator_type: str) -> str:
        """Get help string for operator type."""
        if operator_type not in self._registry and operator_type not in self._evosax_strategies:
            return f"Unknown operator: {operator_type}"
        return f"{operator_type}:param1=value1,param2=value2,..."


DEFAULT_CATALOG = OperatorCatalog()
