"""Tests for the self-registering catalog architecture.

Covers:
  - _registry module: register / register_table / get_registry
  - Completeness: every expected catalog key is present
  - Round-trip: every key resolves to the correct operator class
  - Newly-added operators that were missing from the old monolithic catalog
  - Evosax strategy helpers
"""

from __future__ import annotations

import pytest

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator
from malthusjax.operators.crossover import (
    BinaryUniformCrossover,
    BinomialCrossover,
    BinomialCrossover_injection,
    BlendCrossover,
    BlendCrossover_injection,
    EvosaxUniformCrossoverWrapper,
    RealUniformCrossover,
    RealUniformCrossover_injection,
    SimulatedBinaryCrossover,
    SimulatedBinaryCrossover_injection,
    SinglePointCrossover,
)
from malthusjax.operators.mutation import (
    BallMutation,
    BallMutation_injection,
    BitFlipMutation,
    EvosaxGaussianWrapper,
    GaussianMutation,
    GaussianMutation_injection,
    PolynomialMutation,
    PolynomialMutation_injection,
    ScrambleMutation,
    SwapMutation,
)
from malthusjax.operators.selection import (
    ElitePoolSelection,
    RouletteSelection,
    TournamentSelection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> OperatorCatalog:
    return OperatorCatalog()


# ---------------------------------------------------------------------------
# 1. Completeness — every expected key is registered
# ---------------------------------------------------------------------------

# 3 selection + 11 crossover + 10 mutation + 8 fitness + 3 evosax strategies = 35
EXPECTED_SELECTION = {"tournament", "roulette", "elite_pool"}
EXPECTED_CROSSOVER = {
    "uniform_real",
    "uniform_real_injection",
    "blend",
    "blend_injection",
    "simulated_binary",
    "simulated_binary_injection",
    "binomial",
    "binomial_injection",
    "evosax_uniform_crossover",
    "uniform_binary",
    "single_point",
}
EXPECTED_MUTATION = {
    "gaussian",
    "gaussian_injection",
    "ball",
    "ball_injection",
    "polynomial",
    "polynomial_injection",
    "evosax_gaussian",
    "bitflip",
    "scramble",
    "swap",
}
EXPECTED_FITNESS = {
    "sphere",
    "rastrigin",
    "sphere_minimize",
    "sphere_maximize",
    "bbob",
    "griewank",
    "binary_sum",
    "knapsack",
    "tsp",
}
EXPECTED_EVOSAX = {"evosax_simplega", "evosax_mr15", "evosax_de"}

ALL_EXPECTED = (
    EXPECTED_SELECTION | EXPECTED_CROSSOVER | EXPECTED_MUTATION | EXPECTED_FITNESS | EXPECTED_EVOSAX
)


def test_all_expected_keys_present(catalog: OperatorCatalog) -> None:
    available = set(catalog.list_available())
    missing = ALL_EXPECTED - available
    assert not missing, f"Missing catalog keys: {sorted(missing)}"


def test_no_unexpected_keys(catalog: OperatorCatalog) -> None:
    """Guard against accidental rogue registrations.

    We tolerate keys injected by other test files in the same session
    (e.g. 'custom' from test_catalog.py::test_register_custom_operator).
    """
    # Keys that may be left behind by other test modules
    KNOWN_TEST_ARTIFACTS = {"custom", "__runtime_test__", "__test_unique_op__"}
    available = set(catalog.list_available())
    extra = available - ALL_EXPECTED - KNOWN_TEST_ARTIFACTS
    assert not extra, f"Unexpected catalog keys: {sorted(extra)}"


def test_expected_count(catalog: OperatorCatalog) -> None:
    """At least ALL_EXPECTED keys must be present (may be more from test artifacts)."""
    assert len(catalog.list_available()) >= len(ALL_EXPECTED)


# ---------------------------------------------------------------------------
# 2. Round-trip: each selection key → correct class
# ---------------------------------------------------------------------------

# imports moved to module top


@pytest.mark.parametrize(
    "key, cls",
    [
        ("tournament", TournamentSelection),
        ("roulette", RouletteSelection),
        ("elite_pool", ElitePoolSelection),
    ],
)
def test_selection_roundtrip(catalog: OperatorCatalog, key: str, cls: type) -> None:
    op = catalog.get(key)
    assert isinstance(op, cls)


# ---------------------------------------------------------------------------
# 3. Round-trip: each crossover key → correct class
# ---------------------------------------------------------------------------

# imports moved to module top


@pytest.mark.parametrize(
    "key, cls",
    [
        ("uniform_real", RealUniformCrossover),
        ("uniform_real_injection", RealUniformCrossover_injection),
        ("blend", BlendCrossover),
        ("blend_injection", BlendCrossover_injection),
        ("simulated_binary", SimulatedBinaryCrossover),
        ("simulated_binary_injection", SimulatedBinaryCrossover_injection),
        ("binomial", BinomialCrossover),
        ("binomial_injection", BinomialCrossover_injection),
        ("evosax_uniform_crossover", EvosaxUniformCrossoverWrapper),
        ("uniform_binary", BinaryUniformCrossover),
        ("single_point", SinglePointCrossover),
    ],
)
def test_crossover_roundtrip(catalog: OperatorCatalog, key: str, cls: type) -> None:
    op = catalog.get(key)
    assert isinstance(op, cls)


# ---------------------------------------------------------------------------
# 4. Round-trip: each mutation key → correct class
# ---------------------------------------------------------------------------

# imports moved to module top


@pytest.mark.parametrize(
    "key, cls",
    [
        ("gaussian", GaussianMutation),
        ("gaussian_injection", GaussianMutation_injection),
        ("ball", BallMutation),
        ("ball_injection", BallMutation_injection),
        ("polynomial", PolynomialMutation),
        ("polynomial_injection", PolynomialMutation_injection),
        ("evosax_gaussian", EvosaxGaussianWrapper),
        ("bitflip", BitFlipMutation),
        ("scramble", ScrambleMutation),
        ("swap", SwapMutation),
    ],
)
def test_mutation_roundtrip(catalog: OperatorCatalog, key: str, cls: type) -> None:
    op = catalog.get(key)
    assert isinstance(op, cls)


# ---------------------------------------------------------------------------
# 5. Round-trip: each fitness key → correct evaluator
# ---------------------------------------------------------------------------

# imports moved to module top


@pytest.mark.parametrize(
    "spec, expected_cls",
    [
        ("sphere", BBOBEvaluator),
        ("sphere:dim=5", BBOBEvaluator),
        ("rastrigin", BBOBEvaluator),
        ("sphere_minimize", BBOBEvaluator),
        ("sphere_maximize", BBOBEvaluator),
        ("bbob", BBOBEvaluator),
        ("bbob:fn_name=rastrigin,dim=5", BBOBEvaluator),
    ],
)
def test_fitness_bbob_roundtrip(catalog: OperatorCatalog, spec: str, expected_cls: type) -> None:
    evaluator = catalog.get(spec)
    assert isinstance(evaluator, expected_cls)


def test_fitness_griewank(catalog: OperatorCatalog) -> None:
    from malthusjax.core.fitness import GriewankEvaluator

    evaluator = catalog.get("griewank")
    assert isinstance(evaluator, GriewankEvaluator)


def test_fitness_binary_sum(catalog: OperatorCatalog) -> None:
    from malthusjax.core.fitness import BinarySumEvaluator

    evaluator = catalog.get("binary_sum")
    assert isinstance(evaluator, BinarySumEvaluator)


def test_fitness_knapsack(catalog: OperatorCatalog) -> None:
    import jax.numpy as jnp

    from malthusjax.core.fitness import KnapsackEvaluator

    # Knapsack requires weights, values, capacity — register with defaults
    catalog.register(
        "knapsack",
        lambda **kw: KnapsackEvaluator(
            __import__("malthusjax.core.fitness", fromlist=["KnapsackConfig"]).KnapsackConfig(
                maximize=kw.get("maximize", True),
                weights=kw.get("weights", jnp.array([1.0, 2.0, 3.0])),
                values=kw.get("values", jnp.array([10.0, 20.0, 30.0])),
                capacity=kw.get("capacity", 5.0),
            )
        ),
        override=True,
    )
    evaluator = catalog.get("knapsack")
    assert isinstance(evaluator, KnapsackEvaluator)


# ---------------------------------------------------------------------------
# 6. Evosax strategy helpers return plain strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, expected_name",
    [
        ("evosax_simplega", "SimpleGA"),
        ("evosax_mr15", "MR15_GA"),
        ("evosax_de", "DifferentialEvolution"),
    ],
)
def test_evosax_strategy_strings(catalog: OperatorCatalog, key: str, expected_name: str) -> None:
    result = catalog.get(key)
    assert result == expected_name
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 7. Parameterised spec strings for newly-added operators
# ---------------------------------------------------------------------------


def test_scramble_with_params(catalog: OperatorCatalog) -> None:
    op = catalog.get("scramble:mutation_rate=0.3")
    assert isinstance(op, ScrambleMutation)
    assert op.mutation_rate == pytest.approx(0.3)


def test_swap_with_params(catalog: OperatorCatalog) -> None:
    op = catalog.get("swap:mutation_rate=0.2")
    assert isinstance(op, SwapMutation)
    assert op.mutation_rate == pytest.approx(0.2)


def test_bitflip_with_params(catalog: OperatorCatalog) -> None:
    op = catalog.get("bitflip:mutation_rate=0.05")
    assert isinstance(op, BitFlipMutation)
    assert op.mutation_rate == pytest.approx(0.05)


def test_elite_pool_with_params(catalog: OperatorCatalog) -> None:
    op = catalog.get("elite_pool:num_selections=10,elite_k=5")
    assert isinstance(op, ElitePoolSelection)
    assert op.num_selections == 10
    assert op.elite_k == 5


def test_tournament_defaults(catalog: OperatorCatalog) -> None:
    """Defaults from the registration table should apply."""
    op = catalog.get("tournament")
    assert isinstance(op, TournamentSelection)
    assert op.num_selections == 4
    assert op.tournament_size == 3


def test_tournament_override(catalog: OperatorCatalog) -> None:
    """User spec overrides registry defaults."""
    op = catalog.get("tournament:num_selections=50,tournament_size=7")
    assert isinstance(op, TournamentSelection)
    assert op.num_selections == 50
    assert op.tournament_size == 7


def test_gaussian_with_rate(catalog: OperatorCatalog) -> None:
    op = catalog.get("gaussian:mutation_rate=0.05")
    assert isinstance(op, GaussianMutation)
    assert op.mutation_rate == pytest.approx(0.05)


def test_blend_with_alpha(catalog: OperatorCatalog) -> None:
    op = catalog.get("blend:alpha=0.7")
    assert isinstance(op, BlendCrossover)
    assert op.alpha == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 8. _registry module unit tests
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    from malthusjax.composer._registry import _OPERATOR_REGISTRY, register

    key = "__test_unique_op__"
    try:
        register(key, lambda **kw: "dummy", {"a": 1})
        assert key in _OPERATOR_REGISTRY
        factory, defaults = _OPERATOR_REGISTRY[key]
        assert factory(a=1) == "dummy"
        assert defaults == {"a": 1}
    finally:
        _OPERATOR_REGISTRY.pop(key, None)


def test_registry_duplicate_raises() -> None:
    from malthusjax.composer._registry import _OPERATOR_REGISTRY, register

    key = "__test_dup__"
    try:
        register(key, lambda **kw: None)
        with pytest.raises(KeyError, match="already registered"):
            register(key, lambda **kw: None)
    finally:
        _OPERATOR_REGISTRY.pop(key, None)


def test_registry_override_flag() -> None:
    from malthusjax.composer._registry import _OPERATOR_REGISTRY, register

    key = "__test_override__"
    try:
        register(key, lambda **kw: "first")
        register(key, lambda **kw: "second", override=True)
        factory, _ = _OPERATOR_REGISTRY[key]
        assert factory() == "second"
    finally:
        _OPERATOR_REGISTRY.pop(key, None)


def test_get_registry_returns_copy() -> None:
    from malthusjax.composer._registry import _OPERATOR_REGISTRY, get_registry

    copy = get_registry()
    assert copy == _OPERATOR_REGISTRY
    # Mutations to the copy must not affect the original
    copy["__phantom__"] = (lambda: None, {})
    assert "__phantom__" not in _OPERATOR_REGISTRY


# ---------------------------------------------------------------------------
# 9. Catalog.register at runtime still works
# ---------------------------------------------------------------------------


def test_runtime_register(catalog: OperatorCatalog) -> None:
    """Runtime .register() should be usable and visible immediately."""
    from malthusjax.composer._registry import _OPERATOR_REGISTRY

    key = "__runtime_test__"
    try:
        catalog.register(key, lambda **kw: "runtime_ok")
        assert catalog.get(key) == "runtime_ok"
        assert key in catalog.list_available()
    finally:
        _OPERATOR_REGISTRY.pop(key, None)

def test_catalog_get_with_data_registry():
    """Test data_id resolution"""
    catalog = OperatorCatalog()
    data_reg = {"sphere_10": {"source": "synthetic", "dim": 10}}
    evaluator = catalog.get("sphere:data_id=sphere_10", data_registry=data_reg)
    assert evaluator is not None

def test_catalog_get_backward_compat_no_registry():
    """Test old behavior unchanged"""
    catalog = OperatorCatalog()
    evaluator = catalog.get("sphere:dim=10")
    assert evaluator is not None

def test_data_registry_missing_id_raises():
    """Test error handling"""
    catalog = OperatorCatalog()
    data_reg = {}
    with pytest.raises(KeyError, match="not in registry"):
        catalog.get("sphere:data_id=missing", data_registry=data_reg)
