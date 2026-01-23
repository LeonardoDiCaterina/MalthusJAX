"""malthusjax.composer — minimal composition primitives (defensive imports)."""
__all__ = []

try:
    from .registry import Registry  # type: ignore
    __all__.append("Registry")
except Exception:
    Registry = None  # type: ignore

try:
    from .node import Node  # type: ignore
    __all__.append("Node")
except Exception:
    Node = None  # type: ignore

try:
    from .pipeline import Pipeline  # type: ignore
    __all__.append("Pipeline")
except Exception:
    Pipeline = None  # type: ignore

try:
    from .config import load_config  # type: ignore
    __all__.append("load_config")
except Exception:
    load_config = None  # type: ignore

try:
    from .composer import Composer  # type: ignore
    __all__.append("Composer")
except Exception:
    Composer = None  # type: ignore
