import json

from malthusjax.dash.plan import AnalysisPlan


def test_analysis_plan_plots(tmp_path):
    # Mock data catalog output using a JSON mock source
    mock_data = {
        "experiment": "test_exp",
        "config": {"fn_name": "Sphere", "D": 10, "P": 50, "G": 100},
        "pipelines": {
            "A": [
                {"seed": 1, "best_fitness": 0.01, "duration_seconds": 1.5},
                {"seed": 2, "best_fitness": 0.02, "duration_seconds": 1.6},
            ],
            "B": [
                {"seed": 1, "best_fitness": 0.05, "duration_seconds": 1.0},
                {"seed": 2, "best_fitness": 0.06, "duration_seconds": 1.1},
            ],
        },
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with open(data_dir / "benchmark_results.json", "w") as f:
        json.dump(mock_data, f)

    # Mock config
    config = {
        "sources": {"my_run": str(data_dir)},
        "style": {"width": 8, "grid": False},
        "plots": {
            "fitness_box": {
                "type": "boxplot",
                "y": "best_fitness",
                "title": "Fitness Comparison",
                "style": {"height": 5},
            },
            "time_scaling": {
                "type": "scaling",
                "y": "execution_time",
                "x": "D",
                "title": "Scaling",
            },
        },
    }

    out_dir = tmp_path / "output"
    plan = AnalysisPlan(config, output_dir=out_dir)
    plan.execute()

    assert (out_dir / "fitness_box.png").exists()
    assert (out_dir / "time_scaling.png").exists()


def test_analysis_plan_stats(tmp_path):
    # Same data
    mock_data = {
        "experiment": "test_exp",
        "config": {"fn_name": "Sphere", "D": 10, "P": 50, "G": 100},
        "pipelines": {
            "A": [{"seed": i, "best_fitness": 0.01} for i in range(15)],
            "B": [{"seed": i, "best_fitness": 0.05} for i in range(15)],
        },
    }

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "benchmark_results.json", "w") as f:
        json.dump(mock_data, f)

    config = {
        "sources": {"my_run": str(data_dir)},
        "comparisons": {
            "a_vs_b": {
                "kind": "paired",
                "left": "A",
                "right": "B",
                "group_by": ["fn_name", "D"],
                "spec": {"metric_name": "best_fitness", "min_paired_seeds": 10},
            }
        },
    }

    out_dir = tmp_path / "output"
    plan = AnalysisPlan(config, output_dir=out_dir)
    plan.execute()

    assert (out_dir / "a_vs_b_report.md").exists()
