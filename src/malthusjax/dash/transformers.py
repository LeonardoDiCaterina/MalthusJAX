from typing import Any, Protocol

import numpy as np
import pandas as pd


class DataTransformer(Protocol):
    """Protocol for DataFrame transformers in the Dash layer."""

    def transform(self, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame: ...


class DropWarmupTransformer(DataTransformer):
    """Drops the chronologically first run of each group to eliminate JIT overhead."""

    def transform(self, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        if df.empty or "run_index" not in df.columns:
            return df

        group_by = spec.get(
            "group_by", ["source", "experiment", "pipeline", "fn_name", "D", "P", "G"]
        )
        valid_groups = [col for col in group_by if col in df.columns]

        if not valid_groups:
            return df

        # We drop the row where run_index == 0 for each valid group
        def _drop_first(group: pd.DataFrame) -> pd.DataFrame:
            return group[group["run_index"] != 0]

        df_clean = df.groupby(valid_groups, group_keys=False).apply(_drop_first)
        return df_clean.reset_index(drop=True)


class ScalingTransformer(DataTransformer):
    """Applies mathematical scalings (log, standardize) to features and targets."""

    def transform(self, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df

        df_out = df.copy()
        mode = spec.get("scaling", "linear")
        y_col = spec.get("target")
        x_cols = spec.get("features", [])

        def apply_log(series: pd.Series) -> pd.Series:
            min_val = series.min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            return np.log(series + shift + 1e-9)

        def apply_standardize(series: pd.Series) -> pd.Series:
            std = series.std()
            if std == 0 or pd.isna(std):
                return series - series.mean()
            return (series - series.mean()) / std

        if mode == "linear":
            pass
        elif mode == "log-log":
            if y_col in df_out.columns:
                df_out[y_col] = apply_log(df_out[y_col])
            for x in x_cols:
                if x in df_out.columns:
                    df_out[x] = apply_log(df_out[x])
        elif mode == "log-linear":
            if y_col in df_out.columns:
                df_out[y_col] = apply_log(df_out[y_col])
        elif mode == "linear-log":
            for x in x_cols:
                if x in df_out.columns:
                    df_out[x] = apply_log(df_out[x])
        elif mode == "standardize":
            if y_col in df_out.columns:
                df_out[y_col] = apply_standardize(df_out[y_col])
            for x in x_cols:
                if x in df_out.columns:
                    df_out[x] = apply_standardize(df_out[x])
        else:
            raise ValueError(f"Unknown scaling mode: '{mode}'")

        return df_out


class InteractionTransformer(DataTransformer):
    """Computes interaction terms between columns."""

    def transform(self, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df

        df_out = df.copy()
        interactions = spec.get("interactions", [])

        for interaction in interactions:
            if len(interaction) != 2:
                continue
            col1, col2 = interaction
            if col1 not in df_out.columns or col2 not in df_out.columns:
                raise KeyError(f"Interaction features {col1} or {col2} not found in DataFrame.")

            new_col = f"{col1}_x_{col2}"
            df_out[new_col] = df_out[col1] * df_out[col2]

        return df_out


class CategoricalEncodingTransformer(DataTransformer):
    """Converts categorical variables (e.g. pipeline) to binary indicators."""

    def transform(self, df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df

        df_out = df.copy()
        treatment_column = spec.get("treatment_column")
        treatment_value = spec.get("treatment_value")

        if treatment_column and treatment_column in df_out.columns:
            if treatment_value:
                df_out[f"{treatment_column}_is_{treatment_value}"] = (
                    df_out[treatment_column] == treatment_value
                ).astype(float)
            else:
                unique_vals = df_out[treatment_column].unique()
                if len(unique_vals) == 2:
                    val = unique_vals[1]
                    df_out[f"{treatment_column}_is_{val}"] = (
                        df_out[treatment_column] == val
                    ).astype(float)

        return df_out
