from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class PlotStyle:
    """CSS-like style configuration for plots that supports cascading."""

    width: float | None = None
    height: float | None = None
    palette: list[str] | None = None
    title_fontsize: int | None = None
    label_fontsize: int | None = None
    tick_fontsize: int | None = None
    grid: bool | None = None
    legend_loc: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "PlotStyle | None") -> "PlotStyle":
        """Merge another PlotStyle into this one.

        Values in `other` will override values in `self` if they are not None.
        Dictionaries (like kwargs) will be shallow merged.
        """
        if not other:
            return self

        updates = {}
        for k, v in other.__dict__.items():
            if k == "kwargs":
                updates["kwargs"] = {**self.kwargs, **other.kwargs}
            elif v is not None:
                updates[k] = v

        return replace(self, **updates)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotStyle":
        """Create a PlotStyle from a raw dictionary (e.g. from TOML)."""
        if not data:
            return cls()

        known_fields = {
            "width",
            "height",
            "palette",
            "title_fontsize",
            "label_fontsize",
            "tick_fontsize",
            "grid",
            "legend_loc",
        }

        kwargs = {}
        fields_data = {}

        for k, v in data.items():
            if k in known_fields:
                fields_data[k] = v
            else:
                kwargs[k] = v

        return cls(**fields_data, kwargs=kwargs)
