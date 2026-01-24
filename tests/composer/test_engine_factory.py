import jax.random as jr

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.engine_factory import (
    GeneticEngineAdapter,
    build_engine,
    build_engine_from_catalog,
)
from malthusjax.engine.genetic_fastengine import GeneticEngine


def test_build_engine_basic():
    """Test basic engine construction from operators."""
    catalog = OperatorCatalog()

    # Get operators from catalog
    fitness = catalog.get("sphere:dim=5")
    selection = catalog.get("tournament:num_selections=10,tournament_size=3")
    crossover = catalog.get("blend")
    mutation = catalog.get("gaussian:mutation_rate=0.1")

    # Build engine
    adapter = build_engine(
        fitness_evaluator=fitness,
        selection_op=selection,
        crossover_op=crossover,
        mutation_op=mutation,
        genome_type="real",
        pop_size=10,
        generations=5,
        genome_length=5
    )

    assert isinstance(adapter, GeneticEngineAdapter)
    assert isinstance(adapter.genetic_engine, GeneticEngine)


def test_engine_adapter_run_once():
    """Test that adapter produces BenchmarkRunner-compatible output."""
    catalog = OperatorCatalog()

    # Build small test engine
    adapter = build_engine(
        fitness_evaluator=catalog.get("sphere:dim=3"),
        selection_op=catalog.get("tournament:num_selections=6,tournament_size=2"),
        crossover_op=catalog.get("blend"),
        mutation_op=catalog.get("gaussian:mutation_rate=0.2"),
        genome_type="real",
        pop_size=6,
        generations=3,
        genome_length=3
    )

    # Run engine
    key = jr.PRNGKey(42)
    result = adapter.run_once(key)

    # Check result format
    assert isinstance(result, dict)
    assert "history" in result
    assert "summary" in result
    assert "timings" in result

    # Check history format
    history = result["history"]
    assert isinstance(history, list)
    assert len(history) == 3  # 3 generations

    for gen_data in history:
        assert "generation" in gen_data
        assert "best_fitness" in gen_data
        assert "mean_fitness" in gen_data

    # Check summary format
    summary = result["summary"]
    assert "best_fitness" in summary
    assert "final_generation" in summary
    assert "total_evaluations" in summary


def test_build_engine_from_catalog():
    """Test building engine from catalog operator dict."""
    catalog = OperatorCatalog()

    operators = {
        "fitness": catalog.get("sphere:dim=4"),
        "selection": catalog.get("tournament:num_selections=8,tournament_size=2"),
        "crossover": catalog.get("blend"),
        "mutation": catalog.get("gaussian:mutation_rate=0.15")
    }

    config = {
        "genome_type": "real",
        "pop_size": 8,
        "generations": 2,
        "genome_length": 4,
        "bounds": (-10.0, 10.0)
    }

    adapter = build_engine_from_catalog(operators, config)

    assert isinstance(adapter, GeneticEngineAdapter)

    # Test run
    key = jr.PRNGKey(123)
    result = adapter.run_once(key)
    assert len(result["history"]) == 2


def test_engine_adapter_deterministic():
    """Test that adapter produces deterministic results with same seed."""
    catalog = OperatorCatalog()

    adapter = build_engine(
        fitness_evaluator=catalog.get("sphere:dim=2"),
        selection_op=catalog.get("tournament:num_selections=4,tournament_size=2"),
        crossover_op=catalog.get("blend"),
        mutation_op=catalog.get("gaussian:mutation_rate=0.1"),
        genome_type="real",
        pop_size=4,
        generations=2,
        genome_length=2
    )

    key = jr.PRNGKey(999)
    result1 = adapter.run_once(key)
    result2 = adapter.run_once(key)

    # Should be identical with same seed
    assert result1["summary"]["best_fitness"] == result2["summary"]["best_fitness"]
    assert len(result1["history"]) == len(result2["history"])
