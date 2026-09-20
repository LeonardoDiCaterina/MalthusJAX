"""Tests for Level 2 operator diagnostic logging integration."""

import logging

from malthusjax.core.logger import configure_logging
from malthusjax.operators.crossover.real import BlendCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def test_operator_diagnostic_logging(caplog):
    """Verify that operator setters emit diagnostic traces under malthusjax.operators."""
    configure_logging(level="DEBUG")

    with caplog.at_level(logging.DEBUG, logger="malthusjax.operators"):
        # 1. Mutation
        mut = GaussianMutation(mutation_rate=0.1)
        mut = mut.set_input_length(50)
        mut = mut.set_typed_keys(True)
        mut = mut.set_max_generations(100)

        assert any("GaussianMutation: set input_length=50" in r.message for r in caplog.records)
        assert any("GaussianMutation: set typed_keys=True" in r.message for r in caplog.records)
        assert any("GaussianMutation: set max_generations=100" in r.message for r in caplog.records)

        # 2. Crossover
        cross = BlendCrossover(alpha=0.5)
        cross = cross.set_input_length(25)
        cross = cross.set_typed_keys(False)

        assert any("BlendCrossover: set input_length=25" in r.message for r in caplog.records)
        assert any("BlendCrossover: set typed_keys=False" in r.message for r in caplog.records)

        # 3. Selection
        sel = TournamentSelection(num_selections=20, tournament_size=3)
        sel = sel.set_input_length(50)
        sel = sel.set_n_elites(4)

        assert any("TournamentSelection: set input_length=50" in r.message for r in caplog.records)
        assert any("TournamentSelection: set n_elites=4" in r.message for r in caplog.records)
