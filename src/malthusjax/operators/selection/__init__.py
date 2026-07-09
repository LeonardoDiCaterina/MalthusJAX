"""
Selection Operators Module.
"""

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


def _register_selection() -> None:
    """Register selection operators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            ("tournament", TournamentSelection, {"num_selections": 4, "tournament_size": 3}),
            ("roulette", RouletteSelection, {"num_selections": 4}),
            (
                "elite_pool",
                ElitePoolSelection,
                {"num_selections": 4, "elite_k": 2, "sampling_method": "choice"},
            ),
            ("evosax_mimic_selection", EvoSaxMimicSelection, {"num_selections": 4, "elite_k": 2}),
            ("prefix_tournament", "malthusjax.operators.selection.prefix.tournament:PrefixTournamentSelection", {"num_selections": 4, "tournament_size": 3}),
        ],
        override=True,
    )


_register_selection()
