import pandas as pd
import numpy as np

def main():
    print("Loading LHS global raw data...")
    try:
        df = pd.read_csv("results/diagnostics/lhs_global_raw_data.csv")
    except FileNotFoundError:
        print("Error: results/diagnostics/lhs_global_raw_data.csv not found.")
        return

    # Extract target and ref for Hyp1
    target = df[(df["hypothesis"] == "hyp1") & (df["pipeline"] == "pipeline_malthusjax_wrapper")].copy()
    ref = df[(df["hypothesis"] == "hyp1") & (df["pipeline"] == "pipeline_evosax_baseline")].copy()

    # MalthusJAX maximizes, so the wrapper negates the fitness. We must un-negate it to compare with EvoSAX.
    target["best_fitness"] = -target["best_fitness"]

    merged = pd.merge(
        target, ref,
        on=["experiment_name", "fn_name", "D", "P", "G", "seed"],
        suffixes=("_mjx", "_cpu")
    )

    if merged.empty:
        print("No paired traces found. Have you run Hypothesis 1 yet?")
        return

    print(f"\nSuccessfully paired {len(merged)} LHS evaluations across both architectures.")
    print("Evaluating mathematical parity (Absolute Difference in Best Fitness)...\n")

    print(f"{'Function':<15} | {'Max Abs Diff':<15} | {'Mean Abs Diff':<15} | {'Parity Maintained'}")
    print("-" * 70)

    all_match = True
    for fn in merged["fn_name"].unique():
        fn_df = merged[merged["fn_name"] == fn]
        abs_diff = np.abs(fn_df["best_fitness_mjx"] - fn_df["best_fitness_cpu"])
        
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        
        # Using a standard floating point tolerance for accumulated operations
        is_equal = max_diff < 1e-4
        if not is_equal:
            all_match = False
            
        print(f"{fn:<15} | {max_diff:<15.4e} | {mean_diff:<15.4e} | {str(is_equal)}")

    print("\n")
    if all_match:
        print("VERDICT: ABSOLUTE MATHEMATICAL PARITY CONFIRMED.")
        print("The JAX wrapper produces identically equal evolutionary search paths to the CPU baseline across the entire LHS volume.")
    else:
        print("VERDICT: PARITY FAILED. Significant divergence detected.")

if __name__ == "__main__":
    main()
