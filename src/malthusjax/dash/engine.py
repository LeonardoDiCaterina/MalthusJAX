from typing import Any

import numpy as np
import pandas as pd

from malthusjax.stats import (
    StatisticalComparator,
    StatisticalComparisonSpec,
    StatisticalSuiteResult,
    PairedMetricDataset,
)

class ComparisonEngine:
    """Bridges Pandas DataFrames to the pure-math stats layer."""
    
    def __init__(self, comparator: StatisticalComparator | None = None) -> None:
        self.comparator = comparator or StatisticalComparator()
        
    def compare_paired(
        self,
        df: pd.DataFrame,
        left_pipeline: str,
        right_pipeline: str,
        group_by: list[str],
        spec: StatisticalComparisonSpec,
    ) -> StatisticalSuiteResult:
        """Run a paired comparison suite across multiple groups."""
        if df.empty:
            return StatisticalSuiteResult(spec=spec, results=[])
            
        # Ensure we only have the two pipelines of interest
        df_paired = df[df["pipeline"].isin([left_pipeline, right_pipeline])].copy()
        
        # Pivot the table so left and right are side-by-side per seed
        index_cols = group_by + ["seed"]
        try:
            pivoted = df_paired.pivot(
                index=index_cols,
                columns="pipeline",
                values=spec.metric_name
            ).dropna()
        except KeyError:
            # Missing metric or columns
            return StatisticalSuiteResult(spec=spec, results=[])
            
        if left_pipeline not in pivoted.columns or right_pipeline not in pivoted.columns:
            return StatisticalSuiteResult(spec=spec, results=[])
            
        # Reset index to make grouping easier
        pivoted = pivoted.reset_index()
        
        datasets: list[PairedMetricDataset] = []
        for group_keys, group_df in pivoted.groupby(group_by, dropna=False):
            if isinstance(group_keys, tuple):
                label_parts = [f"{k}={v}" for k, v in zip(group_by, group_keys)]
                group_label = ", ".join(label_parts)
                meta_dict = dict(zip(group_by, group_keys))
            else:
                group_label = f"{group_by[0]}={group_keys}"
                meta_dict = {group_by[0]: group_keys}
                
            left_vals = group_df[left_pipeline].values
            right_vals = group_df[right_pipeline].values
            seeds = group_df["seed"].tolist()
            
            if len(left_vals) < spec.min_paired_seeds:
                continue
                
            dataset = PairedMetricDataset(
                label=group_label,
                left_name=left_pipeline,
                right_name=right_pipeline,
                seeds=seeds,
                left_values=np.asarray(left_vals, dtype=float),
                right_values=np.asarray(right_vals, dtype=float),
                metric_name=spec.metric_name,
                metric_source="dataframe",
                metadata=meta_dict,
            )
            datasets.append(dataset)
            
        return self.comparator.compare_suite(datasets, spec)
