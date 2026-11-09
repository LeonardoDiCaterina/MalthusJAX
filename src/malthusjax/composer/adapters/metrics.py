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

    units: str = "objective"
    orientation: str = "as_requested"
    definition: str = ""


def archive_metric_specs():
    return [
        MetricSpec(
            "best_fitness",
            "max_fitness",
            True,
            units="objective",
            definition="Best occupied-cell objective",
        ),
        MetricSpec(
            "mean_fitness",
            "mean_fitness",
            True,
            units="objective",
            definition="Mean over occupied archive cells",
        ),
        MetricSpec(
            "std_fitness",
            "std_fitness",
            False,
            units="objective",
            orientation="dispersion",
            definition="Occupied-cell population standard deviation; ddof=0",
        ),
        MetricSpec(
            "qd_score",
            "qd_score",
            False,
            units="objective_sum",
            orientation="higher_is_better",
            definition="Sum of higher-is-better repertoire fitness over occupied cells; offset=0",
        ),
        MetricSpec(
            "coverage",
            "coverage",
            False,
            units="percent",
            orientation="higher_is_better",
            definition="100 * occupied cells / total cells",
        ),
        MetricSpec("occupied_cells", "occupied_cells", False, units="cells", orientation="count"),
    ]


def archive_metric_metadata(maximize):
    return {
        "schema_version": 1,
        "archive_empty": "coverage=0; objective statistics unavailable (NaN)",
        "invalid_archive": "NaN/+inf invalidate archive statistics",
        "quality_offset": 0.0,
        "objective_orientation": "higher_is_better" if maximize else "lower_is_better",
        "metrics": {
            m.name: {
                "units": m.units,
                "orientation": ("higher_is_better" if maximize else "lower_is_better")
                if m.is_objective_value
                else m.orientation,
                "definition": m.definition,
            }
            for m in archive_metric_specs()
        },
    }
