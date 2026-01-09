"""
Analyze fitness tuning results and generate visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def load_results(csv_path):
    """Load fitness tuning results."""
    df = pd.read_csv(csv_path)
    return df


def plot_hyperparameter_effects(df, output_dir="results/fitness_tuning/plots"):
    """Plot how each hyperparameter affects fitness."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify hyperparameter columns
    hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(hyperparam_cols[:4]):  # Plot first 4
        param_name = col.replace('hyperparam_', '')
        
        # Group by this hyperparameter and compute mean fitness
        grouped = df.groupby(col)['Mean_Fitness'].agg(['mean', 'std', 'count'])
        
        ax = axes[i]
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                    marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Mean Fitness (lower is better)', fontsize=12)
        ax.set_title(f'Effect of {param_name} on Fitness', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hyperparam_effects.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/hyperparam_effects.png")
    plt.close()


def plot_top_configs(df, top_n=10, output_dir="results/fitness_tuning/plots"):
    """Visualize top N configurations."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df_sorted = df.sort_values('Mean_Fitness').head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_sorted))
    ax.barh(x, df_sorted['Mean_Fitness'], xerr=df_sorted['Fitness_Std'], 
            capsize=5, color='steelblue', edgecolor='black')
    
    # Create labels with hyperparameters
    labels = []
    for idx, row in df_sorted.iterrows():
        hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
        param_str = ', '.join([f"{col.split('_')[-1]}={row[col]}" 
                               for col in hyperparam_cols])
        labels.append(f"{row['Task']} D={int(row['Dim'])}\n{param_str}")
    
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean Fitness (lower is better)', fontsize=12)
    ax.set_title(f'Top {top_n} Configurations', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_configs.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/top_configs.png")
    plt.close()


def plot_heatmap(df, param1, param2, output_dir="results/fitness_tuning/plots"):
    """Plot fitness as a heatmap for two hyperparameters."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    col1 = f'hyperparam_{param1}'
    col2 = f'hyperparam_{param2}'
    
    if col1 not in df.columns or col2 not in df.columns:
        print(f"   Warning: {param1} or {param2} not found in data, skipping heatmap")
        return
    
    # Create pivot table
    pivot = df.pivot_table(values='Mean_Fitness', 
                          index=col1, 
                          columns=col2, 
                          aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Mean Fitness'}, ax=ax)
    ax.set_xlabel(param2, fontsize=12)
    ax.set_ylabel(param1, fontsize=12)
    ax.set_title(f'Fitness Heatmap: {param1} vs {param2}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_{param1}_{param2}.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/heatmap_{param1}_{param2}.png")
    plt.close()


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FITNESS TUNING SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Best fitness achieved: {df['Mean_Fitness'].min():.6f}")
    print(f"Worst fitness: {df['Mean_Fitness'].max():.6f}")
    print(f"Mean fitness: {df['Mean_Fitness'].mean():.6f}")
    print(f"Std fitness: {df['Mean_Fitness'].std():.6f}")
    
    print("\n" + "-" * 80)
    print("TOP 5 CONFIGURATIONS:")
    print("-" * 80)
    
    df_sorted = df.sort_values('Mean_Fitness').head(5)
    hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n{i}. Fitness: {row['Mean_Fitness']:.6f} ± {row['Fitness_Std']:.6f}")
        print(f"   Task: {row['Task']}, Dim: {int(row['Dim'])}, Pop: {int(row['Pop_Size'])}")
        print(f"   Hyperparameters:")
        for col in hyperparam_cols:
            param_name = col.replace('hyperparam_', '')
            print(f"      {param_name}: {row[col]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze fitness tuning results")
    parser.add_argument("csv", help="Path to results CSV file")
    parser.add_argument("--output-dir", default="results/fitness_tuning/plots",
                       help="Directory for output plots")
    args = parser.parse_args()
    
    print(f"📊 Loading results from: {args.csv}")
    df = load_results(args.csv)
    
    print(f"📈 Generating visualizations...")
    plot_hyperparameter_effects(df, args.output_dir)
    plot_top_configs(df, top_n=10, output_dir=args.output_dir)
    
    # Generate heatmaps for key parameter pairs
    plot_heatmap(df, 'mutation_rate', 'sigma', args.output_dir)
    plot_heatmap(df, 'mutation_rate', 'elite_ratio', args.output_dir)
    
    print_summary(df)
    
    print(f"\n✅ Analysis complete. Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
