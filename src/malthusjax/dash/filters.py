import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict[str, str | list[str]]) -> pd.DataFrame:
    """Filter a DataFrame based on equality or inclusion rules.

    If the value is a string, it maps to `col == value`.
    If the value is a list, it maps to `col.isin(value)`.

    Parameters
    ----------
    df : pd.DataFrame
        The data to filter.
    filters : dict
        A dictionary mapping column names to filter values.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    if df.empty or not filters:
        return df

    filtered = df.copy()
    for col, value in filters.items():
        if col not in filtered.columns:
            continue

        if isinstance(value, list):
            filtered = filtered[filtered[col].isin(value)]
        else:
            filtered = filtered[filtered[col] == value]

    return filtered
