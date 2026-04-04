"""Data registry for experiment configurations."""

from typing import Any, Dict

from .io import DataLoader


class DataRegistry:
    """Manage data sources for evaluators."""

    def __init__(self) -> None:
        self._registry: Dict[str, Any] = {}

    def register(self, data_id: str, config: Dict[str, Any]) -> None:
        """Register a data source configuration."""
        self._registry[data_id] = config

    def resolve(self, data_id: str) -> Any:
        """Load and return data by ID depending on its source."""
        config = self._registry.get(data_id)
        if config is None:
            raise KeyError(f"Data ID '{data_id}' not found in registry")

        source = config.get("source", "synthetic")

        if source == "file":
            path = config.get("path")
            if not path:
                raise ValueError(f"File source requires 'path' for data_id {data_id}")
            return DataLoader.load_any(path)
        elif source == "synthetic":
            # For synthetic, we might just return the config dictates
            # and let the evaluator factory generate it.
            return config
        else:
            raise ValueError(f"Unknown data source type '{source}' for data_id {data_id}")

