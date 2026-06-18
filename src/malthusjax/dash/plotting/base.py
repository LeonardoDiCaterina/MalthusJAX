from typing import Protocol, Any
import pandas as pd

from malthusjax.dash.plotting.style import PlotStyle


class BasePlotGenerator(Protocol):
    """Protocol for all plot generators in MalthusDash."""

    def render(self, df: pd.DataFrame, spec: dict[str, Any], style: PlotStyle) -> Any:
        """Render the plot.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to plot (already filtered and scoped).
        spec : dict[str, Any]
            The plot-specific configuration from the TOML file.
        style : PlotStyle
            The merged PlotStyle (global + plot-specific).

        Returns
        -------
        Any
            The resulting Figure object (e.g., matplotlib.figure.Figure),
            so the user can further modify it or save it.
        """
        ...
