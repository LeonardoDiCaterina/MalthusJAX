"""
Selection Operators Module.
"""

from .elite_pool import ElitePoolSelection
from .roulette import RouletteSelection
from .tournament import TournamentSelection

__all__ = [
    "TournamentSelection",
    "RouletteSelection",
    "ElitePoolSelection",
]


def _register_selection() -> None:
    """Register selection operators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            ("tournament", TournamentSelection, {"num_selections": 4, "tournament_size": 3}),
            ("roulette", RouletteSelection, {"num_selections": 4}),
            ("elite_pool", ElitePoolSelection, {"num_selections": 4, "elite_k": 2}),
        ],
        override=True,
    )


_register_selection()
