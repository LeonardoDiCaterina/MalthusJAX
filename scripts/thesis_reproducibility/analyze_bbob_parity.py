import argparse
import json
import pandas as pd
import pingouin as pg
import numpy as np
from pathlib import Path

def parse_parity_results(results_dir: Path) -> pd.DataFrame:
    """Parse all JSON results from the BBOB static grid experiments."""
    records = []
    
    # Iterate over all experiment subdirectories (e.g. sphere_D10, rosenbrock_D100)
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ["metadata", "diagnostics"]:
            continue
            
        data_dir = exp_dir / "data"
        if not data_dir.exists():
            continue
            
        # Typically we expect two pipelines: evosax_baseline and malthusjax_wrapper
        pipelines = [p for p in data_dir.iterdir() if p.is_dir()]
        if len(pipelines) < 2:
            continue
            
        # Parse function name and dimension from exp_dir.name, e.g. "sphere_D10"
        parts = exp_dir.name.split("_D")
        if len(parts) == 2:
            fn_name = parts[0]
            D = parts[1]
        else:
            fn_name = exp_dir.name
            D = "Unknown"
            
        for pipeline_dir in pipelines:
            pipeline_name = pipeline_dir.name
            is_mjx = 1 if "malthusjax" in pipeline_name else 0
            
            for seed_file in pipeline_dir.glob("*.json"):
                seed = seed_file.stem.split("_")[-1]
                
                with open(seed_file, "r") as f:
                    data = json.load(f)
                    
                # Extract best fitness and execution time
                best_fitness = data.get("best_fitness", np.nan)
                exec_time = data.get("execution_time", np.nan)
                
                records.append({
                    "experiment_name": exp_dir.name,
                    "fn_name": fn_name,
                    "D": D,
                    "pipeline": pipeline_name,
                    "is_mjx": is_mjx,
                    "seed": seed,
                    "best_fitness": best_fitness,
                    "execution_time": exec_time
                })
                
    return pd.DataFrame(records)

def run_tost_parity_suite(df: pd.DataFrame, delta: float = 1e-4) -> pd.DataFrame:
    """Run Pingouin TOST and Wilcoxon on the parsed data to prove pragmatic equivalence."""
    results = []
    
    for exp_name in df['experiment_name'].unique():
        df_exp = df[df['experiment_name'] == exp_name]
        
        mjx_data = df_exp[df_exp['is_mjx'] == 1].sort_values("seed")['best_fitness'].values
        evo_data = df_exp[df_exp['is_mjx'] == 0].sort_values("seed")['best_fitness'].values
        
        if len(mjx_data) == 0 or len(evo_data) == 0:
            continue
            
        if len(mjx_data) != len(evo_data):
            print(f"[{exp_name}] Mismatch in seeds: MJX={len(mjx_data)}, Evo={len(evo_data)}")
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
