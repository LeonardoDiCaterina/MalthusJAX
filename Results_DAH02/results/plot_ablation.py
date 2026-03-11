import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_unroll_sensitivity(clean_data_dir="results/ablation/clean_data"):
    # Load all harvested CSVs
    all_files = list(Path(clean_data_dir).glob("*_stats.csv"))
    if not all_files:
        print("No data found. Run harvest_results.py first!")
        return
        
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        # Standardize column names if they differ
        if 'Mean (ms)' in temp_df.columns:
            temp_df = temp_df.rename(columns={'Mean (ms)': 'warm_ms'})
        elif 'mean' in temp_df.columns:
            temp_df = temp_df.rename(columns={'mean': 'warm_ms'})
        elif 'warm_mean' in temp_df.columns:
            temp_df = temp_df.rename(columns={'warm_mean': 'warm_ms'})
            
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # FILTER: Only plot the data from your latest 128-pop sweep to keep Figure 1 clean
    df = df[df['pop_size'] == 128]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Phase 5.1: Plotting Speedup vs. Unroll Factor
    sns.lineplot(data=df, x='unroll', y='warm_ms', hue='engine', marker='o', linewidth=2.5)
    
    plt.title('Impact of Static Entropy Allocation (Pop=128, Dim=50)', fontsize=14)
    plt.xlabel('Unroll Factor', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.xticks([1, 2, 4, 8, 16])
    plt.legend(title='Engine Implementation')
    
    output_file = "results/ablation/figure1_unroll_impact.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Figure 1 saved to {output_file}")

if __name__ == "__main__":
    plot_unroll_sensitivity()