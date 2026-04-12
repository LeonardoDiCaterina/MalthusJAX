from malthusjax.benchmarking.results import ExperimentResult, RunResult


def test_runresult_roundtrip():
    r = RunResult(
        seed=42,
        status="success",
        metrics={"best_fitness": 0.5},
        history=[{"gen": 0, "best": 0.9}, {"gen": 1, "best": 0.5}],
        artifacts={"trace": "trace.json"},
        duration_seconds=1.23,
        timings={"init": 0.1},
        error=None,
    )

    d = r.to_dict()
    assert d["seed"] == 42
    assert "created_at" in d

    # JSON round-trip
    s = r.to_json()
    r2 = RunResult.from_json(s)
    assert r2.seed == r.seed
    assert r2.status == r.status
    assert r2.metrics == r.metrics


def test_experiment_combined_history_and_aggregates():
    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
    )
    r2 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 0.5},
        history=[{"gen": 0, "best": 0.5}],
        artifacts={},
    )

    exp = ExperimentResult(name="ex", runs=[r1, r2])

    ch = exp.combined_history()
    assert any(row.get("seed") == 0 for row in ch)
    assert any(row.get("seed") == 1 for row in ch)

    agg = exp.aggregated_summary()
    assert "best_fitness" in agg
    assert agg["best_fitness"]["mean"] == 0.75
    assert agg["best_fitness"]["median"] == 0.75


def test_comparison_timing_data_and_boxplot():
    from malthusjax.benchmarking.results import ComparisonResult

    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
        duration_seconds=1.2,
        timings={"initialization": 0.1, "evolution": 1.1},
    )
    r2 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 0.5},
        history=[{"gen": 0, "best": 0.5}],
        artifacts={},
        duration_seconds=1.4,
        timings={"initialization": 0.2, "evolution": 1.2},
    )
    exp = ExperimentResult(name="ex", runs=[r1, r2])
    comparison = ComparisonResult(pipelines={"GA": exp})

    duration_data = comparison.timing_data()
    assert duration_data == {"GA": [1.2, 1.4]}

    init_data = comparison.timing_data(timing_key="initialization")
    assert init_data == {"GA": [0.1, 0.2]}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        ax = comparison.plot_timing_boxplot(timing_key="duration_seconds")
        assert ax.get_ylabel() == "duration_seconds (seconds)"
        assert ax.get_title() == "Timing boxplot"

def test_comparison_final_metric_data_and_boxplot_negation():
    from malthusjax.benchmarking.results import ComparisonResult

    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
    )
    r2 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 0.5},
        history=[{"gen": 0, "best": 0.5}],
        artifacts={},
    )
    exp = ExperimentResult(name="ex", runs=[r1, r2])
    comparison = ComparisonResult(pipelines={"Evosax": exp}, negate_map={"Evosax": True})

    final_data = comparison.final_metric_data()
    assert final_data == {"Evosax": [-1.0, -0.5]}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        ax = comparison.plot_final_metric_boxplot(metric_key="best_fitness")
        assert ax.get_ylabel() == "Best Fitness"
        assert ax.get_title() == "Final best_fitness distribution"
