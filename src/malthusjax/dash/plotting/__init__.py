from malthusjax.dash.plotting.base import BasePlotGenerator
from malthusjax.dash.plotting.registry import get_plot_generator, register_plot_generator
from malthusjax.dash.plotting.style import PlotStyle
from malthusjax.dash.plotting.templates import get_dark_style, get_default_style

__all__ = [
    "PlotStyle",
    "BasePlotGenerator",
    "get_plot_generator",
    "register_plot_generator",
    "get_default_style",
    "get_dark_style",
]
