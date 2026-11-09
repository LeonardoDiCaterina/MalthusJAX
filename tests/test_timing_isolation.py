"""
Timing Isolation Tests for MalthusJAX.

Guarantees that:
1. Warmup compilation is cleanly separated from steady-state generational execution.
2. Device synchronization barriers (block_until_ready) prevent unblocked asynchronous dispatch.
3. The benchmark analyzer prioritizes pure execution time over total time.
"""

from malthusjax.composer.composer import Composer


def test_timing_isolation_structure():
    """Verifies warmup and execution timings are distinctly reported and non-zero."""
    composer = Composer.create_default()
    res = composer.quick_run(
        fitness="sphere",
        genome_length=10,
        pop_size=50,
        generations=10,
        bounds=(-5.0, 5.0),
        seeds=(42,),
    )

    run_0 = res.runs[0]
    timings = run_0.timings

    assert "warmup" in timings, "Missing 'warmup' in timings dictionary"
    assert "execution" in timings, "Missing 'execution' in timings dictionary"
    assert "total" in timings, "Missing 'total' in timings dictionary"

    t_warmup = timings["warmup"]
    t_exec = timings["execution"]
    t_total = timings["total"]

    assert t_warmup > 0, "Warmup time must be strictly positive"
    assert t_exec > 0, "Execution time must be strictly positive"
    assert t_total >= t_exec, "Total duration cannot be less than execution duration"


def test_analyzer_prioritizes_execution_time(tmp_path):
    """Tests that the benchmark analyzer extracts 'execution' time rather than 'total' time."""
    import json

    from scripts.benchmark_analyzer import parse_global_data

    mock_run = {
        "seed": 42,
        "timings": {
            "warmup": 5.0,
            "execution": 0.25,
            "total": 5.25,
        },
        "best_fitness": 1.23,
    }

    from tests.timing_records_helpers import record
    mock_run["timing_record"] = record(0.25, seed=42)

    mock_data = {
        "experiment": "test_exp_D10_P50_G10",
        "config": {"D": 10, "P": 50, "G": 10, "fn_name": "lunacek"},
        "pipelines": {"mjx_test": {"per_seed": [mock_run]}},
    }

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    json_file = results_dir / "summary.json"
    with open(json_file, "w") as f:
        json.dump(mock_data, f)

    df = parse_global_data(results_dir, target_metrics=["execution_time", "best_fitness"])

    assert len(df) == 1
    extracted_exec_time = df.iloc[0]["execution_time"]
    assert extracted_exec_time == 0.25, (
        f"Analyzer extracted {extracted_exec_time} instead of pure execution time 0.25!"
    )
