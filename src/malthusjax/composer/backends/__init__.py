"""Backend providers for MalthusJAX — triggers self-registration on import."""

from malthusjax.composer.backends import (  # noqa: F401
    evosax,
    map_elites,
    qdax,
    stub,
    tensorneat,
)
from malthusjax.composer.backends import (
    malthusjax as _malthusjax,  # noqa: F401
)

__all__ = [
    "evosax",
    "map_elites",
    "qdax",
    "stub",
    "tensorneat",
]
