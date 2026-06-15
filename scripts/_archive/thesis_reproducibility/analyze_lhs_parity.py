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

    import pingouin as pg
    from scipy import stats

    print(f"\nSuccessfully paired {len(merged)} LHS evaluations across both architectures.")
    print("Evaluating Algorithmic Parity (Statistical Equivalence TOST & Wilcoxon)...\n")

    print(f"{'Function':<15} | {'Mean Diff':<15} | {'Wilcoxon p':<15} | {'TOST p':<15} | {'Equivalent'}")
    print("-" * 80)

    all_match = True
    for fn in merged["fn_name"].unique():
        fn_df = merged[merged["fn_name"] == fn]
        
        target_fitness = fn_df["best_fitness_mjx"].values
        ref_fitness = fn_df["best_fitness_cpu"].values
        
        # Calculate standard mean difference
        mean_diff = np.abs(np.mean(target_fitness) - np.mean(ref_fitness))
        
        # 1. Wilcoxon Signed-Rank Test (Testing for differences)
        _, wilcox_p = stats.wilcoxon(target_fitness, ref_fitness)
        
        # 2. TOST Equivalence Test (Testing for parity)
        # Margin of equivalence: 1.0 (a tight absolute margin for final fitness)
        tost_res = pg.tost(x=target_fitness, y=ref_fitness, bound=1.0, paired=True)
        tost_p = tost_res['pval'].values[0] if 'pval' in tost_res else 1.0
        
        # They are equivalent if Wilcoxon is non-significant (p > 0.05) OR TOST is significant (p < 0.05)
        # Often in evolutionary algos, we rely strictly on TOST.
        is_equivalent = tost_p < 0.05
        if not is_equivalent:
            all_match = False
            
        print(f"{fn:<15} | {mean_diff:<15.4e} | {wilcox_p:<15.4e} | {tost_p:<15.4e} | {str(is_equivalent)}")

    print("\n")
    if all_match:
        print("VERDICT: STATISTICAL PARITY CONFIRMED.")
        print("The JAX wrapper produces statistically identical convergence distributions to the CPU baseline across the LHS volume.")
    else:
        print("VERDICT: PARITY FAILED. Significant distributional divergence detected.")

if __name__ == "__main__":
    main()
