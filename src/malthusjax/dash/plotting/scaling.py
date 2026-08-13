from typing import Any, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from malthusjax.dash.plotting.base import BasePlotGenerator
from malthusjax.dash.plotting.style import PlotStyle


class ScalingPlot(BasePlotGenerator):
    """Generates a log-log scaling scatter plot with regression lines."""

    def render(self, df: pd.DataFrame, spec: dict[str, Any], style: PlotStyle) -> Figure:
        # We use lmplot which creates a FacetGrid (and its own Figure) rather than an Axes
        df_clean = df.copy()

        y_col = spec.get("y", "execution_time")
        x_col = spec.get("x", "D")
        hue_col = spec.get("hue", "pipeline")

        # Log scaling logic
        if y_col == "execution_time":
            df_clean["log_Y"] = np.log(df_clean[y_col] + 1e-9)
        else:
            min_y = df_clean[y_col].min()
            shift = abs(min_y) + 1 if min_y <= 0 else 0
            df_clean["log_Y"] = np.log(df_clean[y_col] + shift)

        df_clean["log_X"] = np.log(df_clean[x_col])

        # lmplot manages its own figure, so we cannot easily pass an existing axis.
        # It takes height and aspect instead of figsize.
        height = style.height or 6
        width = style.width or 8
        aspect = width / height

        # Extract standard kwargs
        scatter_kws = spec.get("scatter_kws", {"alpha": 0.5})

        grid = sns.lmplot(
            data=df_clean,
            x="log_X",
            y="log_Y",
            hue=hue_col,
            height=height,
            aspect=aspect,
            palette=style.palette,
            scatter_kws=scatter_kws,
            **style.kwargs,
        )

        fig = grid.fig
        ax = grid.ax

        if "title" in spec:
            ax.set_title(spec["title"], fontsize=style.title_fontsize)

        if style.grid is not None:
            ax.grid(style.grid, linestyle="--", alpha=0.7)

        if style.tick_fontsize:
            ax.tick_params(axis="both", which="major", labelsize=style.tick_fontsize)

        if style.label_fontsize:
            ax.set_xlabel(ax.get_xlabel(), fontsize=style.label_fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=style.label_fontsize)

        if hue_col and style.legend_loc and grid.legend:
            sns.move_legend(ax, style.legend_loc)

        fig.tight_layout()
        return cast(Figure, fig)
