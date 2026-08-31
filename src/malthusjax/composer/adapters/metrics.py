from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass
class MetricSpec:
    """Specification for a metric extracted from an evolutionary adapter.

    Attributes:
        name: The standard MalthusJAX name (e.g., 'best_fitness')
        source: The framework's native key or a callable to extract it from the metrics dict
        is_objective_value: True if the metric represents an objective value and requires
            sign-flipping when the framework's native optimization direction does not match
            the user's specified optimization direction.
        description: Optional description of the metric for tooling and dashboards.
    """

    name: str
    source: Union[str, Callable[[Any], Any]]
    is_objective_value: bool = False
    description: str = ""
