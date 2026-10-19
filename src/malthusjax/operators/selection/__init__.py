"""
Selection Operators Module.
"""

from typing import Any

from .elite_pool import ElitePoolSelection
from .evosax_mimic import EvoSaxMimicSelection
from .roulette import RouletteSelection
from .tournament import TournamentSelection

__all__ = [
    "TournamentSelection",
    "RouletteSelection",
    "ElitePoolSelection",
    "SimplifiedElitePoolSelection",
    "EvoSaxMimicSelection",
]


def register_selection_catalog(register_table_fn: Any) -> None:
    """Register selection operators with the global catalog registry."""
    register_table_fn(
        [
            ("tournament", TournamentSelection, {"num_selections": 4, "tournament_size": 3}),
            ("roulette", RouletteSelection, {"num_selections": 4}),
            (
                "elite_pool",
                ElitePoolSelection,
                {"num_selections": 4, "elite_k": 2, "sampling_method": "choice"},
            ),
            ("evosax_mimic_selection", EvoSaxMimicSelection, {"num_selections": 4, "elite_k": 2}),
        ],
        override=True,
    )


def _register_selection(register_table_fn: Any = None) -> None:
    """Backward-compatible registration helper."""
    if register_table_fn is None:
        from malthusjax.composer._registry import register_table

        register_table_fn = register_table
    register_selection_catalog(register_table_fn)
