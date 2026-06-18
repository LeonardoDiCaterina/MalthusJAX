from malthusjax.dash.plotting.style import PlotStyle

def test_plot_style_merge_basic():
    base = PlotStyle(width=10, grid=True)
    override = PlotStyle(width=15, title_fontsize=12)
    
    merged = base.merge(override)
    
    assert merged.width == 15
    assert merged.grid is True
    assert merged.title_fontsize == 12

def test_plot_style_merge_kwargs():
    base = PlotStyle(kwargs={"alpha": 0.5, "color": "red"})
    override = PlotStyle(kwargs={"alpha": 0.8, "marker": "o"})
    
    merged = base.merge(override)
    
    assert merged.kwargs["alpha"] == 0.8
    assert merged.kwargs["color"] == "red"
    assert merged.kwargs["marker"] == "o"

def test_plot_style_from_dict():
    data = {
        "width": 12,
        "grid": False,
        "alpha": 0.5,
        "color": "blue"
    }
    
    style = PlotStyle.from_dict(data)
    
    assert style.width == 12
    assert style.grid is False
    assert style.kwargs["alpha"] == 0.5
    assert style.kwargs["color"] == "blue"
