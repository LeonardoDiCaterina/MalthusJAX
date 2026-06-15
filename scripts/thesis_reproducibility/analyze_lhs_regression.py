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

def parse_lhs_results(results_dir: str) -> pd.DataFrame:
    """Walk through the results directory and compile a DataFrame of runs."""
    records = []
    base_path = Path(results_dir)
    
    # Iterate through all experiment directories that contain metadata/config_snapshot.toml
    for config_path in base_path.rglob("metadata/config_snapshot.toml"):
        exp_dir = config_path.parent.parent
        try:
            with open(config_path, "r") as f:
                config_data = toml.load(f)
        except Exception as e:
            print(f"Failed to parse {config_path}: {e}")
            continue
            
        experiment = config_data.get("experiment", {})
        shared = experiment.get("shared", {})
        
        exp_name = experiment.get("name", exp_dir.name)
        hyp = "unknown"
        if "hyp1" in exp_name:
            hyp = "hyp1"
        elif "hyp2" in exp_name:
            hyp = "hyp2"
        elif "hyp3" in exp_name:
            hyp = "hyp3"
            
        D = shared.get("genome_length", np.nan)
        P = shared.get("pop_size", np.nan)
        G = shared.get("generations", np.nan)
        
        fitness_str = shared.get("fitness", "")
        fn_name = "unknown"
        if "fn_name=" in fitness_str:
            fn_name = fitness_str.split("fn_name=")[1].split(",")[0]
            
        # Parse data JSONs
        data_dir = exp_dir / "data"
        if not data_dir.exists():
            continue
            
        for pipeline_dir in data_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
                
            pipeline_name = pipeline_dir.name
            is_mjx = 1 if "malthusjax" in pipeline_name or "mjx" in pipeline_name else 0
            
            for json_path in pipeline_dir.glob("*.json"):
                try:
                    with open(json_path, "r") as f:
                        run_data = json.load(f)
                except Exception:
                    continue
                    
                metrics = run_data.get("metrics", {})
                timings = run_data.get("timings", {})
                
                best_fitness = np.nan
                if "best_fitness" in metrics:
                    if isinstance(metrics["best_fitness"], dict):
                        best_fitness = metrics["best_fitness"].get("mean", np.nan)
                    else:
                        best_fitness = metrics["best_fitness"]
                        
                exec_time = np.nan
                if "execution" in timings:
                    if isinstance(timings["execution"], dict):
                        exec_time = timings["execution"].get("mean", np.nan)
                    else:
                        exec_time = timings["execution"]
                        
                # Ensure values are float
                try:
                    best_fitness = float(best_fitness)
                    exec_time = float(exec_time)
                except (ValueError, TypeError):
                    pass
                
                records.append({
                    "hypothesis": hyp,
                    "fn_name": fn_name,
                    "experiment_name": exp_name,
                    "pipeline": pipeline_name,
                    "is_mjx": is_mjx,
                    "D": float(D),
                    "P": float(P),
                    "G": float(G),
                    "best_fitness": best_fitness,
                    "execution_time": exec_time
                })
                
    return pd.DataFrame(records)

def run_diagnostics(df: pd.DataFrame, dependent_var: str, output_dir: Path, prefix: str):
    """Run OLS and diagnostic tests."""
    # We want to model log(Y) ~ is_mjx + log(D) + log(P) + log(G) + is_mjx:log(D)
    
    # Filter missing
    df_clean = df.dropna(subset=["is_mjx", "D", "P", "G", dependent_var]).copy()
    if len(df_clean) == 0:
        print(f"[{prefix}] No valid data for {dependent_var}.")
        return
        
    df_clean["log_Y"] = np.log(df_clean[dependent_var] + 1e-9)
    df_clean["log_D"] = np.log(df_clean["D"])
    df_clean["log_P"] = np.log(df_clean["P"])
    df_clean["log_G"] = np.log(df_clean["G"])
    df_clean["is_mjx_x_log_D"] = df_clean["is_mjx"] * df_clean["log_D"]
    
    X = df_clean[["is_mjx", "log_D", "log_P", "log_G", "is_mjx_x_log_D"]]
    X = sm.add_constant(X)
    y = df_clean["log_Y"]
    
    # 1. Fit OLS Model
    model = sm.OLS(y, X).fit()
    
    # 2. VIF (Multicollinearity)
    # We calculate VIF for D, P, G to ensure LHS generated an orthogonal space
    X_vif = sm.add_constant(df_clean[["log_D", "log_P", "log_G"]])
    vif_data = []
    for i in range(1, X_vif.shape[1]):
        vif = variance_inflation_factor(X_vif.values, i)
        vif_data.append({"Feature": X_vif.columns[i], "VIF": vif})
    vif_df = pd.DataFrame(vif_data)
    
    # 3. Breusch-Pagan Test (Heteroscedasticity)
    _, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    
    # 4. Shapiro-Wilk Test (Normality of residuals)
    # Shapiro-Wilk may fail for N > 5000, so we sample if necessary
    resid = model.resid
    if len(resid) > 5000:
        resid = np.random.choice(resid, 5000, replace=False)
    _, sw_pval = stats.shapiro(resid)
    
    # 5. Mardia's Test (Multivariate Normality)
    # Testing multivariate normality on the independent variables D, P, G
    mardia_res = pg.multivariate_normality(df_clean[["log_D", "log_P", "log_G"]])
    
    # Save OLS summary
    with open(output_dir / f"{prefix}_{dependent_var}_ols_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write("\n\n--- Diagnostics ---\n")
        f.write("VIF Scores:\n")
        f.write(vif_df.to_string(index=False))
        f.write(f"\n\nBreusch-Pagan p-value: {bp_pval:.4e} (H0: Homoscedasticity)")
        f.write(f"\nShapiro-Wilk p-value: {sw_pval:.4e} (H0: Normality of residuals)")
        f.write(f"\nMardia's Test: {mardia_res}\n")
        
    print(f"[{prefix}] Regression & Diagnostics saved for {dependent_var}.")

    # Return metrics for Pivot Table
    return {
        "Dependent_Var": dependent_var,
        "R2": model.rsquared,
        "is_mjx_coef": model.params.get("is_mjx", np.nan),
        "is_mjx_pval": model.pvalues.get("is_mjx", np.nan),
        "interaction_coef": model.params.get("is_mjx_x_log_D", np.nan),
        "interaction_pval": model.pvalues.get("is_mjx_x_log_D", np.nan),
        "log_D_coef": model.params.get("log_D", np.nan),
        "log_D_pval": model.pvalues.get("log_D", np.nan),
        "BP_pval": bp_pval,
        "SW_pval": sw_pval,
        "Mardia_hz_pval": mardia_res.pval if hasattr(mardia_res, "pval") else np.nan
    }

def main():
    parser = argparse.ArgumentParser(description="Run regression diagnostics on LHS results.")
    parser.add_argument("--results-dir", type=str, default="results/thesis_lhs", help="Directory containing JSON results")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics", help="Directory to save diagnostic reports")
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing results from {args.results_dir}...")
    df = parse_lhs_results(args.results_dir)
    
    if df.empty:
        print("No valid JSON artifacts found. Please run the LHS configurations first.")
        return
        
    df.to_csv(out_path / "lhs_raw_data.csv", index=False)
    print(f"Raw LHS data saved to {out_path / 'lhs_raw_data.csv'}")
        
    pivot_rows = []

    # We analyze each hypothesis and benchmark function separately
    for hyp in ["hyp1", "hyp2", "hyp3"]:
        df_hyp = df[df["hypothesis"] == hyp]
        if df_hyp.empty:
            continue
            
        for fn_name in df_hyp["fn_name"].unique():
            df_hyp_fn = df_hyp[df_hyp["fn_name"] == fn_name]
            if df_hyp_fn.empty:
                continue
                
            print(f"\nProcessing {hyp} - {fn_name}...")
            prefix = f"{hyp}_{fn_name}"
            
            # Analyze Execution Time
            res_time = run_diagnostics(df_hyp_fn, "execution_time", out_path, prefix=prefix)
            if res_time:
                res_time["Hypothesis"] = hyp
                res_time["Benchmark"] = fn_name
                pivot_rows.append(res_time)
            
            # Analyze Best Fitness
            res_fit = run_diagnostics(df_hyp_fn, "best_fitness", out_path, prefix=prefix)
            if res_fit:
                res_fit["Hypothesis"] = hyp
                res_fit["Benchmark"] = fn_name
                pivot_rows.append(res_fit)

    if pivot_rows:
        pivot_df = pd.DataFrame(pivot_rows)
        pivot_file = out_path / "grand_pivot_table.csv"
        pivot_df.to_csv(pivot_file, index=False)
        print(f"\nGrand Pivot Table data saved to {pivot_file}")

if __name__ == "__main__":
    main()
