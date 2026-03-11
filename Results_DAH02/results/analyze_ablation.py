import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def compare_engines(csv_file):
    df = pd.read_csv(csv_file)
    
    # Extract data for Normal vs Ablation
    normal = df[df['engine'] == 'Standard_GA']['warm_ms']
    ablation = df[df['engine'] == 'Standard_GA_Ablation']['warm_ms']
    
    if normal.empty or ablation.empty:
        print(f"Skipping {csv_file.name}: Missing engine data.")
        return

    # Calculate Speedup (H1/H2 Hypothesis)
    mean_normal = normal.mean()
    mean_ablation = ablation.mean()
    speedup = mean_ablation / mean_normal
    
    # Statistical significance (α = 0.001)
    t_stat, p_val = ttest_ind(normal, ablation)
    
    print(f"\n--- Analysis for {csv_file.name} ---")
    print(f"Normal GA:   {mean_normal:.4f} ms")
    print(f"Ablation GA: {mean_ablation:.4f} ms")
    print(f"🚀 Speedup:  {speedup:.2f}x")
    print(f"P-value:     {p_val:.6f} ({'SIGNIFICANT' if p_val < 0.001 else 'NOT SIGNIFICANT'})")

if __name__ == "__main__":
    from pathlib import Path
    # Analyze everything in clean_data
    for file in Path("results/ablation/clean_data").glob("*.csv"):
        compare_engines(file)