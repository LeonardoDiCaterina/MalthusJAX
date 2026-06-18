from malthusjax.dash.plotting.style import PlotStyle


def get_default_style() -> PlotStyle:
    """Provides the baseline styling template for MalthusDash."""
    return PlotStyle(
        width=10,
        height=6,
        palette=["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"],
        title_fontsize=16,
        label_fontsize=12,
        tick_fontsize=10,
        grid=True,
        legend_loc="upper right",
        kwargs={}
    )

def get_dark_style() -> PlotStyle:
    """Provides a premium dark mode styling template."""
    return PlotStyle(
        width=10,
        height=6,
        palette=["#5dade2", "#ec7063", "#58d68d", "#f4d03f", "#af7ac5", "#aeb6bf"],
        title_fontsize=16,
        label_fontsize=12,
        tick_fontsize=10,
        grid=True,
        legend_loc="upper right",
        kwargs={"facecolor": "#2c3e50"}
    )
