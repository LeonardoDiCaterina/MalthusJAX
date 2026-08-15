import pandas as pd
from matplotlib.figure import Figure

from malthusjax.dash.plotting.boxplot import BoxPlot
from malthusjax.dash.plotting.style import PlotStyle


def test_boxplot_render():
    df = pd.DataFrame({"pipeline": ["A", "A", "B", "B"], "best_fitness": [1.0, 2.0, 1.5, 2.5]})

    spec = {"title": "Test Boxplot"}
    style = PlotStyle(width=8, height=6, title_fontsize=14)

    plotter = BoxPlot()
    fig = plotter.render(df, spec, style)

    assert isinstance(fig, Figure)

    ax = fig.axes[0]
    assert ax.get_title() == "Test Boxplot"
