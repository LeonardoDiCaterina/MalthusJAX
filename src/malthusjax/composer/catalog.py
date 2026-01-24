from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.fitness.binary_evaluators import KnapsackConfig, KnapsackEvaluator

from ..operators.crossover.binary import UniformCrossover
from ..operators.crossover.real import BlendCrossover
from ..operators.mutation.binary import BitFlipMutation
from ..operators.mutation.real import GaussianMutation
from ..operators.selection.roulette import RouletteSelection

# Import major MalthusJAX operators for catalog
from ..operators.selection.tournament import TournamentSelection


class OperatorCatalog:
    """Catalog for creating operators from string specifications.
    Supports format: "operator_type:param1=value1,param2=value2"
    Examples:
        catalog.get("tournament")  # Default parameters
        catalog.get("tournament:selections=5,size=3")  # With parameters
        catalog.get("gaussian:rate=0.1")  # Single parameter
    """
    def __init__(self) -> None:
        """Initialize the operator catalog with default operators."""
        # Map operator_type -> factory_class or factory_function
        self._factories: Dict[str, Callable] = {
            # Selection operators
            "tournament": self._create_tournament_selection,
            "roulette": self._create_roulette_selection,
            # Crossover operators
            "blend": BlendCrossover,
            "uniform": UniformCrossover,

            # Mutation operators
            "gaussian": GaussianMutation,
            "bitflip": BitFlipMutation,

            # Fitness evaluators (need special handling for config)
            "sphere": self._create_sphere_evaluator,
            "griewank": self._create_griewank_evaluator,
            "rastrigin": self._create_rastrigin_evaluator,
            "knapsack": self._create_knapsack_evaluator,
        }

    def parse_spec(self, spec: str) -> Tuple[str, Dict[str, Any]]:
        """Parse operator specification string.
        Args:
            spec: String like "operator_type:param1=value1,param2=value2"
        Returns:
            Tuple of (operator_type, params_dict)
        Raises:
            ValueError: If spec format is invalid
        """
        spec = spec.strip()
        if not spec:
            raise ValueError("Empty operator specification")

        if ":" not in spec:
            return spec, {}  # Just operator name, no params

        operator_type, params_str = spec.split(":", 1)
        operator_type = operator_type.strip()
        params = {}

        if params_str.strip():
            # Parse param1=value1,param2=value2
            for param_pair in params_str.split(","):
                param_pair = param_pair.strip()
                if "=" not in param_pair:
                    raise ValueError(
                        "Invalid parameter format: "
                        f"'{param_pair}'. Expected 'key=value'"
                    )

                key, value = param_pair.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Try to convert to appropriate type
                params[key] = self._convert_value(value)

        return operator_type, params

    def _convert_value(self, value_str: str) -> Union[int, float, str]:
        """Convert string value to appropriate Python type."""
        value_str = value_str.strip()

        try:
            return int(value_str)
        except ValueError:
            pass

        try:
            return float(value_str)
        except ValueError:
            pass

        # Keep as string (remove quotes if present)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        return value_str

    def get(self, spec: str) -> Any:
        """Get configured operator instance from specification string.
        Args:
            spec: Operator specification like "tournament:selections=5,size=3"
        Returns:
            Configured operator instance
        Raises:
            KeyError: If operator type is not registered
            ValueError: If spec format is invalid or parameters are invalid
        """
        operator_type, params = self.parse_spec(spec)

        if operator_type not in self._factories:
            available = sorted(self._factories.keys())
            raise KeyError(f"Unknown operator type: '{operator_type}'. Available: {available}")

        factory = self._factories[operator_type]

        try:
            return factory(**params)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for '{operator_type}': {e}")

    def register(self, operator_type: str, factory: Callable, override: bool = False) -> None:
        """Register a new operator type.
        Args:
            operator_type: String name for the operator
            factory: Callable that creates operator instances
            override: Whether to override existing registrations
        """
        if not override and operator_type in self._factories:
            raise KeyError(f"Operator type '{operator_type}' already registered")

        self._factories[operator_type] = factory

    def list_available(self) -> List[str]:
        """Return list of available operator types."""
        return sorted(self._factories.keys())

    def get_help(self, operator_type: str) -> str:
        """Get help string for operator type."""
        if operator_type not in self._factories:
            return f"Unknown operator: {operator_type}"

        return f"{operator_type}:param1=value1,param2=value2,..."

    def _create_sphere_evaluator(self, **kwargs: Any) -> BBOBEvaluator:
        """Create BBOBEvaluator configured for sphere function."""
        config = BBOBConfig(
            fn_name="sphere",
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            maximize=kwargs.get("maximize", False),
            seed=kwargs.get("seed", 42)
        )
        return BBOBEvaluator.create(config)

    def _create_griewank_evaluator(self, **kwargs: Any) -> BBOBEvaluator:
        """Create BBOBEvaluator configured for griewank function."""
        config = BBOBConfig(
            fn_name="griewank",
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            maximize=kwargs.get("maximize", False),
            seed=kwargs.get("seed", 42)
        )
        return BBOBEvaluator.create(config)

    def _create_rastrigin_evaluator(self, **kwargs: Any) -> BBOBEvaluator:
        """Create BBOBEvaluator configured for rastrigin function."""
        config = BBOBConfig(
            fn_name="rastrigin",
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            maximize=kwargs.get("maximize", False),
            seed=kwargs.get("seed", 42)
        )
        return BBOBEvaluator.create(config)

    def _create_knapsack_evaluator(self, **kwargs: Any) -> KnapsackEvaluator:
        """Create KnapsackEvaluator with KnapsackConfig."""
        config = KnapsackConfig(**kwargs)
        return KnapsackEvaluator(config)

    def _create_tournament_selection(self, **kwargs: Any) -> TournamentSelection:
        return TournamentSelection(
            num_selections=kwargs.get("num_selections", 4),
            tournament_size=kwargs.get("tournament_size", 3),
            **{k: v for k, v in kwargs.items() if k not in ["num_selections", "tournament_size"]}
        )

    def _create_roulette_selection(self, **kwargs: Any) -> RouletteSelection:
        return RouletteSelection(
            num_selections=kwargs.get("num_selections", 4),
            **{k: v for k, v in kwargs.items() if k not in ["num_selections"]}
        )


# Global default catalog instance
DEFAULT_CATALOG = OperatorCatalog()
