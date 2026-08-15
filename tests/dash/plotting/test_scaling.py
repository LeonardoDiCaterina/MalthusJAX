import pandas as pd
from matplotlib.figure import Figure

from malthusjax.dash.plotting.scaling import ScalingPlot
from malthusjax.dash.plotting.style import PlotStyle


def test_scaling_render():
    df = pd.DataFrame(
        {
            "pipeline": ["A", "A", "B", "B"],
            "D": [10, 20, 10, 20],
            "execution_time": [1.0, 4.0, 2.0, 8.0],
        }
    )

    spec = {"title": "Test Scaling"}
    style = PlotStyle(width=8, height=6)

    plotter = ScalingPlot()
    fig = plotter.render(df, spec, style)

    assert isinstance(fig, Figure)
