import json
import argparse
import numpy as np
from pathlib import Path

def get_stats(data_dir, pipeline):
    pipe_dir = Path(data_dir) / f"pipeline_{pipeline}"
    fitnesses = []
    durations = []
    for fpath in pipe_dir.glob("seed_*.json"):
        with open(fpath, "r") as f:
            data = json.load(f)
            fitnesses.append(data["metrics"]["best_fitness"])
            durations.append(data["duration_seconds"])
            
    # Remove highest duration (JIT warmup)
    if len(durations) > 1:
        max_idx = np.argmax(durations)
        durations.pop(max_idx)
        
    return np.mean(fitnesses), np.std(fitnesses), np.mean(durations), np.std(durations)

def generate_latex(results_dir, out_path):
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Parity results between EvoSAX and MalthusJAX across BBOB problems.}",
        r"    \label{tab:parity_results}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lrrrrrr}",
        r"        \toprule",
        r"        \textbf{Problem} & \textbf{EvoSAX Fitness} & \textbf{MalthusJAX Fitness} & \textbf{$\Delta$ Fitness} & \textbf{EvoSAX Time (s)} & \textbf{MalthusJAX Time (s)} & \textbf{Speedup} \\",
        r"        \midrule"
    ]
    
    mapping = {
        "parity_sphere": "P1 (Sphere)",
        "parity_rastrigin": "P2 (Rastrigin)",
        "parity_rosenbrock": "P3 (Rosenbrock)",
        "parity_griewank_rosenbrock": "P4 (Griewank-Rosenbrock)",
        "parity_ellipsoidal": "P5 (Ellipsoidal)"
    }
    
    for prob, prob_name in mapping.items():
        data_dir = Path(results_dir).parent / prob / "data"
        if not data_dir.exists():
            continue
            
        # EvoSAX
        evo_fit_mean, evo_fit_std, evo_time_mean, evo_time_std = get_stats(data_dir, "evosax_ga")
        
        # MalthusJAX
        mjax_fit_mean, mjax_fit_std, mjax_time_mean, mjax_time_std = get_stats(data_dir, "malthusjax_ga")
        
        delta_fit = mjax_fit_mean - evo_fit_mean
        speedup = evo_time_mean / mjax_time_mean if mjax_time_mean > 0 else 0
        
        delta_str = f"+{delta_fit:.2f}" if delta_fit > 0 else f"{delta_fit:.2f}"
        
        row = f"        {prob_name} & {evo_fit_mean:.2f} $\\pm$ {evo_fit_std:.2f} & {mjax_fit_mean:.2f} $\\pm$ {mjax_fit_std:.2f} & {delta_str} & {evo_time_mean:.2f} $\\pm$ {evo_time_std:.2f} & {mjax_time_mean:.2f} $\\pm$ {mjax_time_std:.2f} & \\textbf{{{speedup:.2f}x}} \\\\"
        lines.append(row)
        
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"\end{table}"
    ])
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, required=True, help="Directory of the aggregated suite (e.g. results/thesis_suite)")
    parser.add_argument("--out", type=str, required=True, help="Output .tex file path")
    args = parser.parse_args()
    
    generate_latex(args.suite_dir, args.out)
