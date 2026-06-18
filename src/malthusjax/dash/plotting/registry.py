from malthusjax.dash.plotting.base import BasePlotGenerator
from malthusjax.dash.plotting.boxplot import BoxPlot
from malthusjax.dash.plotting.scaling import ScalingPlot


_PLOT_REGISTRY: dict[str, BasePlotGenerator] = {
    "boxplot": BoxPlot(),
    "scaling": ScalingPlot(),
}

def register_plot_generator(name: str, generator: BasePlotGenerator) -> None:
    """Register a new plot generator."""
    _PLOT_REGISTRY[name] = generator

def get_plot_generator(name: str) -> BasePlotGenerator:
    """Retrieve a plot generator by name."""
    if name not in _PLOT_REGISTRY:
        raise KeyError(f"Unknown plot type: '{name}'. Available: {list(_PLOT_REGISTRY)}")
    return _PLOT_REGISTRY[name]
