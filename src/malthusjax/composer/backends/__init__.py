"""Backend providers for MalthusJAX — triggers self-registration on import."""
from malthusjax.composer.backends import (  # noqa: F401
    evosax,
    malthusjax as _malthusjax,
    map_elites,
    qdax,
    stub,
    tensorneat,
)

__all__ = [
    "evosax",
    "map_elites",
    "qdax",
    "stub",
    "tensorneat",
]
