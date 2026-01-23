
from .pipeline import Pipeline


class Composer:
    """Minimal composer stub for PR1: load_config and compose() are side-effect free."""
    @classmethod
    def load_config(cls, path: str, pipeline_name: str) -> "Composer":
        return cls()

    def compose(self) -> Pipeline:
        # Return an empty Pipeline by default (fill in later)
        return Pipeline(name="empty", nodes=[])
