"""LSP Selection — thin alias over MalthusJAX TournamentSelection."""

from malthusjax.operators.selection.tournament import TournamentSelection

# Binary tournament (q=2) is the recommended setting for MEP (§3.6).
# LinearTournamentSelection is a direct alias — override here for LSP-specific
# selection logic (e.g., linear rank selection) in the future.
LinearTournamentSelection = TournamentSelection

__all__ = ["LinearTournamentSelection"]
