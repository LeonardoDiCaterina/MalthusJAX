import pandas as pd
from malthusjax.stats import StatisticalComparisonSpec
from malthusjax.dash.engine import ComparisonEngine


def test_engine_compare_paired():
    df = pd.DataFrame({
        "fn_name": ["Sphere", "Sphere", "Sphere", "Sphere"],
        "D": [10, 10, 10, 10],
        "pipeline": ["A", "A", "B", "B"],
        "seed": [1, 2, 1, 2],
        "best_fitness": [0.1, 0.2, 0.15, 0.25]
    })
    
    engine = ComparisonEngine()
    spec = StatisticalComparisonSpec(
        metric_name="best_fitness",
        min_paired_seeds=2
    )
    
    suite = engine.compare_paired(
        df=df,
        left_pipeline="A",
        right_pipeline="B",
        group_by=["fn_name", "D"],
        spec=spec
    )
    
    assert len(suite.results) == 1
    res = suite.results[0]
    assert res.n_paired == 2
    assert res.metadata["fn_name"] == "Sphere"
    assert res.metadata["D"] == 10
    
def test_engine_empty_dataframe():
    engine = ComparisonEngine()
    spec = StatisticalComparisonSpec(metric_name="score")
    suite = engine.compare_paired(pd.DataFrame(), "A", "B", ["group"], spec)
    assert len(suite.results) == 0

def test_engine_missing_pipeline():
    df = pd.DataFrame({
        "group": [1, 1],
        "pipeline": ["A", "A"],
        "seed": [1, 2],
        "score": [1.0, 2.0]
    })
    engine = ComparisonEngine()
    spec = StatisticalComparisonSpec(metric_name="score")
    suite = engine.compare_paired(df, "A", "B", ["group"], spec)
    assert len(suite.results) == 0
