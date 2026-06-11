import pytest
from malthusjax.composer.composer import Composer
from malthusjax.benchmarking.results import ExperimentResult, RunResult

def test_normalize_seeds_edge_cases():
    composer = Composer.create_default()
    
    # Test int > 0
    assert Composer._normalize_seeds(5) == (1, 2, 3, 4, 5)
    
    # Test int <= 0
    with pytest.raises(ValueError, match="seeds must be > 0"):
        Composer._normalize_seeds(0)
    
    # Test empty tuple
    with pytest.raises(ValueError, match="seeds must not be empty"):
        Composer._normalize_seeds([])
        
    # Test sequence of ints
    assert Composer._normalize_seeds([10, 20, 30]) == (10, 20, 30)

def test_quick_run_genome_fallback(monkeypatch):
    composer = Composer.create_default()
    
    # Mock the actual execution to return a dummy ExperimentResult
    def mock_run(*args, **kwargs):
        return ExperimentResult(name="test", runs=[])
    
    monkeypatch.setattr("malthusjax.benchmarking.runner.BenchmarkRunner.run", mock_run)
    
    # Trigger the genome fallback logic
    res = composer.quick_run(
        genome="real:dim=5,bounds=(-2.5,2.5)",
        seeds=[1],
        generations=1
    )
    assert res is not None

    # Test shape inference
    res = composer.quick_run(
        genome="real:shape=(3,)",
        seeds=[1],
        generations=1
    )
    assert res is not None

def test_populate_metrics_fallback():
    composer = Composer.create_default()
    
    res = ExperimentResult(name="test", runs=[
        RunResult(seed=1, status="success", metrics={}, history=[
            {"generation": 100, "best_fitness": 42.0}
        ]),
        RunResult(seed=2, status="success", metrics={}, history=[]), # Empty history
        RunResult(seed=3, status="success", metrics={}, history=[
            {"other_metric": 5} # Missing best_fitness and generation
        ]),
    ])
    
    composer._postprocess_experiment_final_from_history(res, force=True)
    
    assert res.runs[0].metrics["best_fitness"] == 42.0
    assert res.runs[0].metrics["final_generation"] == 100
    
    assert "best_fitness" not in res.runs[1].metrics
    assert "best_fitness" not in res.runs[2].metrics

def test_infer_genome_length():
    composer = Composer.create_default()
    
    assert composer._infer_genome_length({"genome_length": 15}) == 15
    assert composer._infer_genome_length({"fitness": "bbob:num_dims=12"}) == 12
    assert composer._infer_genome_length({}) == 10  # default
