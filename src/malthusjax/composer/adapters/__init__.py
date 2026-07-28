"""Universal Adapters for MalthusJAX."""

import enum

class EvalMode(str, enum.Enum):
    NATIVE = "native"
    MALTHUSJAX = "malthusjax"

from .base import UniversalAdapterEngine
from .decorators import adapter

__all__ = ["UniversalAdapterEngine", "adapter", "EvalMode"]
