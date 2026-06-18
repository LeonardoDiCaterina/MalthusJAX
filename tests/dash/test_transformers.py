import numpy as np
import pandas as pd
import pytest

from malthusjax.dash.transformers import (
    DropWarmupTransformer,
    ScalingTransformer,
    InteractionTransformer,
    CategoricalEncodingTransformer,
)


def test_drop_warmup_transformer():
    df = pd.DataFrame({
        "pipeline": ["A", "A", "B", "B"],
        "run_index": [0, 1, 0, 1],
        "D": [10, 10, 10, 10],
        "execution_time": [2.0, 1.0, 2.5, 1.2],
    })
    
    spec = {"group_by": ["pipeline", "D"]}
    transformer = DropWarmupTransformer()
    out_df = transformer.transform(df, spec)
    
    # Drops where run_index == 0
    assert len(out_df) == 2
    assert set(out_df["run_index"]) == {1}
    assert out_df.iloc[0]["execution_time"] == 1.0


def test_drop_warmup_empty_df():
    df = pd.DataFrame(columns=["pipeline", "run_index"])
    transformer = DropWarmupTransformer()
    out_df = transformer.transform(df, {})
    assert out_df.empty


def test_drop_warmup_single_run():
    # If a group only has run_index == 0, it should be dropped completely
    df = pd.DataFrame({
        "pipeline": ["A"],
        "run_index": [0],
        "D": [10],
    })
    
    spec = {"group_by": ["pipeline", "D"]}
    transformer = DropWarmupTransformer()
    out_df = transformer.transform(df, spec)
    assert out_df.empty


def test_scaling_transformer_log_log():
    df = pd.DataFrame({
        "execution_time": [1.0, 10.0, 100.0],
        "D": [10.0, 20.0, 30.0],
    })
    
    spec = {
        "scaling": "log-log",
        "target": "execution_time",
        "features": ["D"]
    }
    
    transformer = ScalingTransformer()
    out_df = transformer.transform(df, spec)
    
    assert np.allclose(out_df["execution_time"], np.log(df["execution_time"] + 1e-9))
    assert np.allclose(out_df["D"], np.log(df["D"] + 1e-9))


def test_scaling_transformer_standardize():
    df = pd.DataFrame({
        "execution_time": [1.0, 2.0, 3.0],
    })
    
    spec = {
        "scaling": "standardize",
        "target": "execution_time"
    }
    
    transformer = ScalingTransformer()
    out_df = transformer.transform(df, spec)
    
    # Standardize should yield mean=0, std=1
    assert np.isclose(out_df["execution_time"].mean(), 0.0)
    assert np.isclose(out_df["execution_time"].std(), 1.0)


def test_interaction_transformer():
    df = pd.DataFrame({
        "D": [10, 20],
        "is_fast": [1.0, 0.0]
    })
    
    spec = {"interactions": [["D", "is_fast"]]}
    transformer = InteractionTransformer()
    out_df = transformer.transform(df, spec)
    
    assert "D_x_is_fast" in out_df.columns
    assert list(out_df["D_x_is_fast"]) == [10.0, 0.0]


def test_interaction_missing_column():
    df = pd.DataFrame({"D": [10]})
    spec = {"interactions": [["D", "is_fast"]]}
    transformer = InteractionTransformer()
    
    with pytest.raises(KeyError):
        transformer.transform(df, spec)
