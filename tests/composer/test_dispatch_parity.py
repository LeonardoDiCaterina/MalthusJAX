"""Tests verifying that the BackendRegistry dispatch routes to the expected builder."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.core import (
    EvoSAXStrategy,
    GeneticStrategy,
    MapElitesStrategy,
    QDAXStrategy,
    TensorNEATStrategy,
)

DISPATCH_MATRIX = [
    ("malthusjax", GeneticStrategy, "build_real_engine"),
    ("evosax", EvoSAXStrategy, "build_evosax_engine"),
    ("composable_evosax", EvoSAXStrategy, "build_composable_evosax_engine"),
    ("qdax", QDAXStrategy, "build_qdax_engine"),
    ("tensorneat", TensorNEATStrategy, "build_tensorneat_engine"),
    ("composable_tensorneat", TensorNEATStrategy, "build_composable_tensorneat_engine"),
    ("map_elites", MapElitesStrategy, "build_map_elites_engine"),
]


BACKEND_MODULE_MAP = {
    "build_real_engine": "malthusjax.composer.factory",
    "build_evosax_engine": "malthusjax.composer.backends.evosax",
    "build_composable_evosax_engine": "malthusjax.composer.backends.evosax",
    "build_qdax_engine": "malthusjax.composer.backends.qdax",
    "build_tensorneat_engine": "malthusjax.composer.backends.tensorneat",
    "build_composable_tensorneat_engine": "malthusjax.composer.backends.tensorneat",
    "map_elites": "malthusjax.composer.backends.map_elites",
    "build_map_elites_engine": "malthusjax.composer.backends.map_elites",
}


@pytest.mark.parametrize("backend,strategy_cls,builder_name", DISPATCH_MATRIX)
def test_registry_routes_to_correct_builder(backend, strategy_cls, builder_name, monkeypatch):
    """The registry must route each backend to the corresponding builder function."""
    called_with = {}

    def spy(*args, **kwargs):
        called_with["args"] = args
        called_with["kwargs"] = kwargs
        mock_eng = MagicMock()
        mock_eng.run_once.return_value = {
            "best_fitness": 0.0,
            "final_generation": 1,
            "history": [{"generation": 0, "best_fitness": 0.0}],
        }
        return mock_eng

    mod_name = BACKEND_MODULE_MAP[builder_name]
    monkeypatch.setattr(f"{mod_name}.{builder_name}", spy)

    composer = Composer()
    kwargs = {
        "backend": backend,
        "fitness": "sphere:dim=2",
        "pop_size": 4,
        "generations": 1,
        "seeds": (1,),
    }
    if backend in ("tensorneat", "composable_tensorneat"):
        kwargs["fitness"] = "xor"

    composer.quick_run(**kwargs)

    assert "args" in called_with, f"Expected {builder_name} to be called for backend={backend}"
