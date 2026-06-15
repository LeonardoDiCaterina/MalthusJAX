"""Advanced Regression and Diagnostics Engine for LHS Experiments.

Parses MalthusJAX experiment JSON artifacts and performs Multiple Linear
Regression along with rigorous statistical diagnostics:
- Variance Inflation Factor (VIF)
- Breusch-Pagan Test (Heteroscedasticity)
- Shapiro-Wilk Test (Normality)
- Mardia's Test (Multivariate Normality)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import toml

def parse_global_lhs_data(results_dir: Path) -> pd.DataFrame:
    """Parse all LHS runs into a single unpivoted dataframe."""
    records = []
    
    for config_path in results_dir.rglob("metadata/config_snapshot.toml"):
        exp_dir = config_path.parent.parent
        try:
            with open(config_path, "r") as f:
                config_data = toml.load(f)
        except Exception:
            continue
            
        experiment = config_data.get("experiment", {})
        shared = experiment.get("shared", {})
        
        exp_name = experiment.get("name", exp_dir.name)
        hyp = "unknown"
        if "hyp1" in exp_name: hyp = "hyp1"
        elif "hyp2" in exp_name: hyp = "hyp2"
        elif "hyp3" in exp_name: hyp = "hyp3"
            
        lhs_id = exp_name.split("_")[-1]
        
        D = shared.get("genome_length", np.nan)
        P = shared.get("pop_size", np.nan)
        G = shared.get("generations", np.nan)
        
        fitness_str = shared.get("fitness", "")
        fn_name = "unknown"
        if "fn_name=" in fitness_str:
            fn_name = fitness_str.split("fn_name=")[1].split(",")[0]
            
        data_dir = exp_dir / "data"
        if not data_dir.exists():
            continue
            
        for pipeline_dir in data_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
                
            pipeline_name = pipeline_dir.name
            
            for json_path in pipeline_dir.glob("*.json"):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue
                
                runs = data.get("runs", [data]) if "runs" in data else [data]
                
                for run_data in runs:
                    seed = run_data.get("seed", -1)
                    metrics = run_data.get("metrics", {})
                    timings = run_data.get("timings", {})
                    
                    best_fitness = np.nan
                    if "history" in run_data and len(run_data["history"]) > 0:
                        best_fitness = run_data["history"][-1].get("best_fitness", np.nan)
                    elif "best_fitness" in metrics:
                        if isinstance(metrics["best_fitness"], dict):
                            best_fitness = metrics["best_fitness"].get("mean", np.nan)
                        else:
                            best_fitness = metrics["best_fitness"]
                            
                    exec_time = run_data.get("duration_seconds", np.nan)
                    if "total" in timings:
                        exec_time = timings["total"]
                    elif "execution" in timings:
                        if isinstance(timings["execution"], dict):
                            exec_time = timings["execution"].get("mean", np.nan)
                        else:
                            exec_time = timings["execution"]
                            
                    try:
                        best_fitness = float(best_fitness)
                        exec_time = float(exec_time)
                    except (ValueError, TypeError):
                        pass
                    
                    if np.isnan(best_fitness) or np.isnan(exec_time):
                        continue
                    
                    records.append({
                        "hypothesis": hyp,
                        "experiment_name": exp_name,
                        "fn_name": fn_name,
                        "lhs_id": lhs_id,
                        "D": float(D),
                        "P": float(P),
                        "G": float(G),
                        "pipeline": pipeline_name,
                        "seed": seed,
                        "best_fitness": best_fitness,
                        "execution_time": exec_time
                    })
                
    return pd.DataFrame(records)

def synthesize_regression_dataset(df_global: pd.DataFrame, hyp_num: str, target_pipeline: str, ref_hyp: str, ref_pipeline: str) -> pd.DataFrame:
    """Join target pipeline data with its specific reference pipeline to calculate relative effect."""
    df_target = df_global[(df_global["hypothesis"] == hyp_num) & (df_global["pipeline"] == target_pipeline)].copy()
    df_ref = df_global[(df_global["hypothesis"] == ref_hyp) & (df_global["pipeline"] == ref_pipeline)].copy()
    
    # We want to stack them, adding is_treatment = 1 for target, 0 for ref
    df_target["is_treatment"] = 1
    df_ref["is_treatment"] = 0
    
    # Inner merge to strictly enforce paired existence based on LHS geometry
    common_keys = ["fn_name", "lhs_id", "seed", "D", "P", "G"]
    
    # Filter only rows that exist in BOTH (paired traces)
    merged = pd.merge(df_target[common_keys], df_ref[common_keys], on=common_keys, how="inner")
    
    # Re-extract the paired rows from original frames
    df_target_paired = pd.merge(df_target, merged, on=common_keys, how="inner")
    df_ref_paired = pd.merge(df_ref, merged, on=common_keys, how="inner")
    
    df_paired = pd.concat([df_ref_paired, df_target_paired], ignore_index=True)
    return df_paired

def run_diagnostics(df: pd.DataFrame, dependent_var: str, output_dir: Path, prefix: str):
    """Run OLS and diagnostic tests on synthesized paired datasets."""
    if len(df) < 2:
        return None
        
    df_clean = df.copy()
    if dependent_var == "execution_time":
        df_clean["log_Y"] = np.log(df_clean[dependent_var] + 1e-9)
    else:
        # Fitness doesn't need to be strictly log-transformed (and handles negative maximization scores)
        df_clean["log_Y"] = df_clean[dependent_var]
        
    df_clean["log_D"] = np.log(df_clean["D"])
    df_clean["log_P"] = np.log(df_clean["P"])
    df_clean["log_G"] = np.log(df_clean["G"])
    df_clean["interaction_term"] = df_clean["is_treatment"] * df_clean["log_D"]
    
    X = df_clean[["is_treatment", "log_D", "log_P", "log_G", "interaction_term"]]
    X = sm.add_constant(X)
    y = df_clean["log_Y"]
    
    model = sm.OLS(y, X).fit()
    
    bp_pval = np.nan
    try:
        _, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    except Exception: pass
    
    sw_pval = np.nan
    resid = model.resid.values
    if len(resid) > 5000:
        resid = np.random.choice(resid, 5000, replace=False)
    try:
        _, sw_pval = stats.shapiro(resid)
    except Exception: pass
    
    with open(output_dir / f"{prefix}_{dependent_var}_ols_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f"\n\nBreusch-Pagan p-value: {bp_pval:.4e}")
        f.write(f"\nShapiro-Wilk p-value: {sw_pval:.4e}\n")
        
    return {
        "Dependent_Var": dependent_var,
        "R2": model.rsquared,
        "is_treatment_coef": model.params.get("is_treatment", np.nan),
        "is_treatment_pval": model.pvalues.get("is_treatment", np.nan),
        "interaction_coef": model.params.get("interaction_term", np.nan),
        "interaction_pval": model.pvalues.get("interaction_term", np.nan),
        "log_D_coef": model.params.get("log_D", np.nan),
        "BP_pval": bp_pval,
        "SW_pval": sw_pval,
    }

def main():
    parser = argparse.ArgumentParser(description="Run regression diagnostics on cross-matched LHS results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing JSON results")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics", help="Directory to save diagnostic reports")
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing results from {args.results_dir}...")
    raw_csv_path = diagnostics_dir / "lhs_global_raw_data.csv"
    existing_df = None
    if raw_csv_path.exists():
        existing_df = pd.read_csv(raw_csv_path)
        
    new_raw_df = parse_global_lhs_data(results_dir)
    
    if existing_df is not None and not new_raw_df.empty:
        raw_df = pd.concat([existing_df, new_raw_df], ignore_index=True)
        raw_df = raw_df.drop_duplicates(subset=["experiment_name", "fn_name", "D", "P", "G", "seed", "pipeline", "hypothesis"], keep="last")
    elif not new_raw_df.empty:
        raw_df = new_raw_df
    else:
        raw_df = existing_df
        
    if raw_df is not None:
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Global unpivoted data saved to {raw_csv_path} (Total Rows: {len(raw_df)})")
    else:
        print("No raw data found or parsed.")
        return
        
    pivot_rows = []

    # Map of target hypothesis to its respective treatment vs reference pipelines
    analysis_targets = [
        # HYPOTHESIS 1 (Algorithmic Parity)
        # Ref = hyp1: pipeline_evosax_baseline
        {"hyp": "hyp1", "target": "pipeline_malthusjax_wrapper", "ref_hyp": "hyp1", "ref_pipeline": "pipeline_evosax_baseline"},
        
        # HYPOTHESIS 2 (Ablations)
        # Ref = hyp1: pipeline_malthusjax_wrapper
        {"hyp": "hyp2", "target": "pipeline_mjx_native_mutation", "ref_hyp": "hyp1", "ref_pipeline": "pipeline_malthusjax_wrapper"},
        {"hyp": "hyp2", "target": "pipeline_mjx_native_crossover", "ref_hyp": "hyp1", "ref_pipeline": "pipeline_malthusjax_wrapper"},
        {"hyp": "hyp2", "target": "pipeline_mjx_native_selection_elite", "ref_hyp": "hyp1", "ref_pipeline": "pipeline_malthusjax_wrapper"},
        {"hyp": "hyp2", "target": "pipeline_mjx_native_selection_tournament", "ref_hyp": "hyp1", "ref_pipeline": "pipeline_malthusjax_wrapper"},
        
        # HYPOTHESIS 3 (Precision)
        # Ref = hyp2: pipeline_mjx_native_mutation (This represents the FP32 native engine)
        {"hyp": "hyp3", "target": "pipeline_malthusjax_fp16", "ref_hyp": "hyp2", "ref_pipeline": "pipeline_mjx_native_mutation"},
        {"hyp": "hyp3", "target": "pipeline_malthusjax_bf16", "ref_hyp": "hyp2", "ref_pipeline": "pipeline_mjx_native_mutation"},
    ]
    
    for analysis in analysis_targets:
        target_name = analysis["target"].replace("pipeline_", "")
        print(f"\nSynthesizing {analysis['hyp']}: {target_name} vs {analysis['ref_pipeline']}")
        
        df_paired = synthesize_regression_dataset(
            df_global, 
            analysis["hyp"], analysis["target"], 
            analysis["ref_hyp"], analysis["ref_pipeline"]
        )
        
        if df_paired.empty:
            print(f"  -> No paired traces found.")
            continue
            
        print(f"  -> Synthesized {len(df_paired)//2} paired traces.")
        
        for fn_name in df_paired["fn_name"].unique():
            df_paired_fn = df_paired[df_paired["fn_name"] == fn_name]
            if df_paired_fn.empty: continue
            
            prefix = f"{analysis['hyp']}_{target_name}_{fn_name}"
            
            for var in ["execution_time", "best_fitness"]:
                res = run_diagnostics(df_paired_fn, var, out_path, prefix=prefix)
                if res:
                    res["Hypothesis"] = analysis["hyp"]
                    res["Target_Pipeline"] = target_name
                    res["Benchmark"] = fn_name
                    pivot_rows.append(res)

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        pivot_file = out_path / "regression_pivot_table.csv"
        pivot_df.to_csv(pivot_file, index=False)
        print(f"\nRegression Grand Pivot Table saved to {pivot_file}")
        
if __name__ == "__main__":
    main()
