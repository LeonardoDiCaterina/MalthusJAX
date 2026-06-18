from pathlib import Path
from typing import Any

from malthusjax.dash.catalog import DataCatalog
from malthusjax.dash.engine import ComparisonEngine
from malthusjax.dash.filters import apply_filters
from malthusjax.dash.plotting import get_plot_generator, PlotStyle
from malthusjax.stats import StatisticalComparisonSpec
from malthusjax.dash.transformers import (
    DropWarmupTransformer, 
    ScalingTransformer, 
    InteractionTransformer, 
    CategoricalEncodingTransformer
)
from malthusjax.stats.regression import fit_ols
from malthusjax.stats.core import RegressionDataset
from malthusjax.stats.io import regression_to_markdown


class AnalysisPlan:
    """Orchestrates the Dash workflow from a TOML configuration dictionary."""

    def __init__(self, config: dict[str, Any], output_dir: str | Path = "./output") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = DataCatalog()
        self.engine = ComparisonEngine()

    def execute(self) -> None:
        """Execute the full TOML plan."""
        # 1. Load Sources
        sources = self.config.get("sources", {})
        for name, path in sources.items():
            self.catalog.add_source(name, path)
        self.catalog.load()

        df_master = self.catalog.data

        # 2. Generate Plots
        plots = self.config.get("plots", {})
        global_style_data = self.config.get("style", {})
        global_style = PlotStyle.from_dict(global_style_data)

        for plot_name, plot_cfg in plots.items():
            # Apply dataset filters
            df_plot = apply_filters(df_master, plot_cfg.get("filters", {}))

            # Resolve PlotStyle (Global cascaded into Local)
            local_style_data = plot_cfg.get("style", {})
            local_style = PlotStyle.from_dict(local_style_data)
            merged_style = global_style.merge(local_style)

            # Generate Plot
            plot_type = plot_cfg.get("type", "boxplot")
            generator = get_plot_generator(plot_type)

            fig = generator.render(df_plot, plot_cfg, merged_style)

            # Save Output
            out_file = self.output_dir / f"{plot_name}.png"
            dpi = plot_cfg.get("dpi", 300)
            fig.savefig(out_file, dpi=dpi, bbox_inches="tight")

            # Cleanup Matplotlib memory
            import matplotlib.pyplot as plt
            plt.close(fig)

        # 3. Run Statistical Comparisons (Optional)
        comparisons = self.config.get("comparisons", {})
        for comp_name, comp_cfg in comparisons.items():
            kind = comp_cfg.get("kind", "paired")
            
            if kind == "paired":
                df_comp = apply_filters(df_master, comp_cfg.get("filters", {}))
                spec = StatisticalComparisonSpec(**comp_cfg.get("spec", {}))
                
                suite = self.engine.compare_paired(
                    df=df_comp,
                    left_pipeline=comp_cfg.get("left"),
                    right_pipeline=comp_cfg.get("right"),
                    group_by=comp_cfg.get("group_by", ["fn_name", "D"]),
                    spec=spec,
                )
                
                # Write markdown report
                out_file = self.output_dir / f"{comp_name}_report.md"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# Statistical Comparison: {comp_name}\n\n")
                    f.write(suite.to_markdown())

        # 4. Run Regressions (Optional)
        regressions = self.config.get("regressions", {})
        for reg_name, reg_cfg in regressions.items():
            # Apply initial filters
            df_reg = apply_filters(df_master, reg_cfg.get("filters", {}))
            
            # Apply Transformers
            if reg_cfg.get("drop_warmup", True):
                df_reg = DropWarmupTransformer().transform(df_reg, reg_cfg)
                
            df_reg = CategoricalEncodingTransformer().transform(df_reg, reg_cfg)
            df_reg = InteractionTransformer().transform(df_reg, reg_cfg)
            df_reg = ScalingTransformer().transform(df_reg, reg_cfg)
            
            target = reg_cfg.get("target")
            features = reg_cfg.get("features", [])
            
            if target not in df_reg.columns:
                continue
                
            # Find any dynamically generated interaction columns or categorical columns to include
            # For simplicity, we just use whatever is explicitly asked for, plus interactions if they were just generated.
            interactions = reg_cfg.get("interactions", [])
            for interaction in interactions:
                if len(interaction) == 2:
                    features.append(f"{interaction[0]}_x_{interaction[1]}")
                    
            treatment_column = reg_cfg.get("treatment_column")
            if treatment_column:
                # Add all indicator columns generated
                indicator_cols = [c for c in df_reg.columns if c.startswith(f"{treatment_column}_is_")]
                features.extend(indicator_cols)
                
            # Filter out features that didn't make it or are invalid
            valid_features = [f for f in set(features) if f in df_reg.columns]
            
            # Build Regression Dataset
            import numpy as np
            X_dict = {feat: np.asarray(df_reg[feat]) for feat in valid_features}
            y_arr = np.asarray(df_reg[target])
            
            dataset = RegressionDataset(
                X=X_dict,
                y=y_arr,
                target_name=target,
                dataset_name=reg_name
            )
            
            robust = reg_cfg.get("robust", ["HC3"])
            result = fit_ols(dataset, robust=robust)
            
            # Save Output
            out_file = self.output_dir / f"{reg_name}_regression.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(regression_to_markdown(result))
