from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from malthusjax.dash.plotting.base import BasePlotGenerator
from malthusjax.dash.plotting.style import PlotStyle


class BoxPlot(BasePlotGenerator):
    """Generates a boxplot for distributions (e.g. fitness, time) across pipelines."""

    def render(self, df: pd.DataFrame, spec: dict[str, Any], style: PlotStyle) -> Figure:
        fig, ax = plt.subplots(
            figsize=(style.width or 8, style.height or 6)
        )
        
        x_col = spec.get("x", "pipeline")
        y_col = spec.get("y", "best_fitness")
        hue_col = spec.get("hue", None)
        
        sns.boxplot(
            data=df, 
            x=x_col, 
            y=y_col,
            hue=hue_col,
            palette=style.palette,
            ax=ax,
            **style.kwargs
        )
        
        if spec.get("log_y", False):
            ax.set_yscale("log")
            
        if "title" in spec:
            ax.set_title(spec["title"], fontsize=style.title_fontsize)
            
        if style.grid is not None:
            ax.grid(style.grid, linestyle="--", alpha=0.7)
            
        if style.tick_fontsize:
            ax.tick_params(axis="both", which="major", labelsize=style.tick_fontsize)
            
        if style.label_fontsize:
            ax.set_xlabel(ax.get_xlabel(), fontsize=style.label_fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=style.label_fontsize)
            
        if hue_col and style.legend_loc and ax.get_legend():
            sns.move_legend(ax, style.legend_loc)
            
        fig.tight_layout()
        return fig
