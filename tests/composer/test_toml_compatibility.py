"""Tests verifying TOML configuration compatibility with the backend registry."""

from __future__ import annotations

import pytest

from malthusjax.composer.composer import Composer

TOML_FIXTURES = [
    (
        "malthusjax_ga",
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 4
        generations = 1
        seeds = [1]
        [pipelines.ga]
        crossover = "blend:alpha=0.5"
        """,
        ["ga"],
    ),
    (
        "evosax_backend",
        """
        [experiment.shared]
        backend = "evosax"
        evosax_strategy = "SimpleGA"
        fitness = "sphere:dim=2"
        pop_size = 4
        generations = 1
        seeds = [1]
        [pipelines.es]
        """,
        ["es"],
    ),
    (
        "mixed_backends",
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 4
        generations = 1
        seeds = [1]
        [pipelines.native]
        crossover = "blend:alpha=0.5"
        [pipelines.evosax]
        backend = "evosax"
        evosax_strategy = "SimpleGA"
        """,
        ["native", "evosax"],
    ),
]


@pytest.mark.parametrize(
    "desc,toml_content,expected", TOML_FIXTURES, ids=lambda x: x if isinstance(x, str) else ""
)
def test_from_toml_backward_compat(desc, toml_content, expected, tmp_path):
    """Existing TOML configs must continue to produce results with expected pipeline names."""
    path = tmp_path / f"{desc}.toml"
    path.write_text(toml_content)
    result = Composer.from_toml(str(path), shared_initial_population=False)
    for name in expected:
        assert name in result.pipelines
