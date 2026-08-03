import pandas as pd

from malthusjax.dash.filters import apply_filters


def test_apply_filters_empty_df():
    df = pd.DataFrame()
    out = apply_filters(df, {"col": "A"})
    assert out.empty


def test_apply_filters_empty_filters():
    df = pd.DataFrame({"A": [1, 2]})
    out = apply_filters(df, {})
    assert len(out) == 2


def test_apply_filters_single_value():
    df = pd.DataFrame({"fn_name": ["A", "B", "A"], "val": [1, 2, 3]})
    out = apply_filters(df, {"fn_name": "A"})
    assert len(out) == 2
    assert (out["fn_name"] == "A").all()


def test_apply_filters_list_value():
    df = pd.DataFrame({"pipeline": ["P1", "P2", "P3", "P4"]})
    out = apply_filters(df, {"pipeline": ["P1", "P3"]})
    assert len(out) == 2
    assert "P2" not in out["pipeline"].values


def test_apply_filters_missing_column_ignored():
    df = pd.DataFrame({"A": [1, 2]})
    out = apply_filters(df, {"B": 3})
    assert len(out) == 2
