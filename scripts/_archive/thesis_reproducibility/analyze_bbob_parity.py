import argparse
import json
import pandas as pd
import pingouin as pg
import numpy as np
from pathlib import Path

def parse_parity_results(results_dir: Path) -> pd.DataFrame:
    """Parse all JSON results from the BBOB static grid summary.json files."""
    records = []
    
    for summary_file in results_dir.rglob("summary.json"):
        if "metadata" in summary_file.parts or "diagnostics" in summary_file.parts:
            continue
            
        exp_dir = summary_file.parent
        name = exp_dir.name
        
        # Extract fn_name and pipeline from directory name (e.g. sphere_mjx_evosax_exact)
        fn_name = "unknown"
        pipeline_name = name
        for split_token in ["_mjx", "_evosax", "_malthusjax"]:
            if split_token in name:
                parts = name.split(split_token)
                fn_name = parts[0]
                pipeline_name = split_token[1:] + split_token.join(parts[1:])
                break
                
        is_mjx = 1 if ("malthusjax" in pipeline_name or "mjx" in pipeline_name) else 0
        
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
        except Exception:
            continue
            
        # summary.json contains a 'runs' array
        runs = data.get("runs", [])
        for run in runs:
            seed = run.get("seed", -1)
            
            best_fitness = np.nan
            if "history" in run and len(run["history"]) > 0:
                best_fitness = run["history"][-1].get("best_fitness", np.nan)
            elif "metrics" in run and "best_fitness" in run["metrics"]:
                bf = run["metrics"]["best_fitness"]
                if isinstance(bf, dict):
                    best_fitness = bf.get("mean", np.nan)
                else:
                    best_fitness = float(bf)
                    
            exec_time = run.get("duration_seconds", np.nan)
            if "timings" in run and "total" in run["timings"]:
                exec_time = run["timings"]["total"]
                
            records.append({
                "experiment_name": f"{fn_name}_comparison",
                "fn_name": fn_name,
                "D": 10, # Assuming static D=10 for these grids based on previous scripts
                "pipeline": pipeline_name,
                "is_mjx": is_mjx,
                "seed": seed,
                "best_fitness": best_fitness,
                "execution_time": exec_time
            })
                
    if len(records) == 0:
        print(f"[DEBUG] Searched inside {results_dir}. Found 0 valid summary records.")
            
    return pd.DataFrame(records)

def run_tost_parity_suite(df: pd.DataFrame, delta: float = 1e-4) -> pd.DataFrame:
    """Run Pingouin TOST and Wilcoxon on the parsed data to prove pragmatic equivalence."""
    results = []
    
    # Filter only the exactly matched pipelines for the parity comparison!
    parity_pipelines = ["mjx_evosax_exact", "evosax_simplega"]
    df_filtered = df[df['pipeline'].isin(parity_pipelines)].copy()
    
    for exp_name in df_filtered['experiment_name'].unique():
        df_exp = df_filtered[df_filtered['experiment_name'] == exp_name]
        
        mjx_data = df_exp[df_exp['is_mjx'] == 1].sort_values("seed")['best_fitness'].values
        evo_data = df_exp[df_exp['is_mjx'] == 0].sort_values("seed")['best_fitness'].values
        
        if len(mjx_data) == 0 or len(evo_data) == 0:
            continue
            
        if len(mjx_data) != len(evo_data):
            print(f"[{exp_name}] Mismatch in seeds: MJX={len(mjx_data)} vs Evo={len(evo_data)}. Found pipelines: {df_exp['pipeline'].unique()}")
            continue
            
        # 1. Non-parametric Wilcoxon signed-rank test
        # Tests if the median of paired differences is zero
        wilcoxon_res = pg.wilcoxon(mjx_data, evo_data)
        w_pval = wilcoxon_res['p-val'].values[0]
        
        # 2. TOST (Two One-Sided Tests) for Equivalence
        # Proves that the difference is strictly within the [-delta, delta] bounds
        tost_res = pg.tost(x=mjx_data, y=evo_data, bound=delta, paired=True)
        tost_pval = tost_res['pval'].values[0]
        
        # Determine strict equivalence
        is_equivalent = tost_pval < 0.05
        
        # Absolute mean difference
        mean_diff = np.abs(np.mean(mjx_data) - np.mean(evo_data))
        
        fn_name = df_exp['fn_name'].iloc[0]
        D = df_exp['D'].iloc[0]
        
        results.append({
            "Function": fn_name,
            "Dims (D)": D,
            "Seeds": len(mjx_data),
            "Mean_Diff": f"{mean_diff:.2e}",
            "Wilcoxon_p": f"{w_pval:.4f}",
            "TOST_p": f"{tost_pval:.2e}",
            "Equivalent": is_equivalent
        })
        
    res_df = pd.DataFrame(results)
    return res_df

def main():
    parser = argparse.ArgumentParser(description="Analyze Pragmatic Equivalence for the BBOB Static Grids.")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to the bbob_allfunctions_match_... directory")
    parser.add_argument("--delta", type=float, default=1e-4, help="TOST equivalence margin (delta)")
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: {results_path} does not exist.")
        return
        
    print(f"Parsing BBOB grids from {results_path}...")
    df = parse_parity_results(results_path)
    print(f"Loaded {len(df)} independent evaluation traces.")
    
    print("\nRunning TOST and Wilcoxon Statistical Parity Tests...")
    tost_df = run_tost_parity_suite(df, delta=args.delta)
    
    print("\n" + "="*80)
    print(" " * 20 + "PRAGMATIC EQUIVALENCE SUMMARY (BBOB GRIDS)")
    print("="*80)
    print(tost_df.to_markdown(index=False))
    print("="*80 + "\n")
    
    # Save the markdown table
    out_table = results_path / "parity_summary_table.md"
    tost_df.to_markdown(open(out_table, "w"), index=False)
    print(f"Saved Markdown table to {out_table}")

if __name__ == "__main__":
    main()
