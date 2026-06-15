"""Generate LHS configuration files for the MalthusJAX thesis.

This script uses Latin Hypercube Sampling to explore the parameter space
(Genome Size, Population Size, Generations) for three thesis hypotheses:
1. Parity (MalthusJAX vs Evosax)
2. Mutation Superiority (Native vs Wrapper)
3. Precision (FP32 vs BF16)
"""

import argparse
import os
import toml
from pathlib import Path
from scipy.stats import qmc

def generate_seeds(num_seeds: int = 100) -> list[int]:
    return list(range(1, num_seeds + 1))

def create_hyp1_config(name: str, out_dir: str, fn_name: str, D: int, P: int, G: int, seeds: list[int]) -> dict:
    elite_k = max(2, int(P * 0.16666666666666666))
    return {
        "experiment": {
            "name": name,
            "output_dir": out_dir,
            "shared": {
                "fitness": f"bbob:fn_name={fn_name},num_dims={D},maximize=false",
                "selection": f"elite_pool:num_selections={P},elite_k={elite_k}",
                "pop_size": P,
                "generations": G,
                "genome_length": D,
                "bounds": [-5.0, 5.0],
                "seeds": seeds,
                "serialize_history": False
            }
        },
        "pipelines": {
            "malthusjax_wrapper": {
                "backend": "malthusjax",
                "elitism": 0,
                "crossover": "evosax_uniform_crossover:crossover_rate=0.3",
                "mutation": "evosax_gaussian:mutation_strength=0.05"
            },
            "evosax_baseline": {
                "backend": "evosax",
                "evosax_strategy": "SimpleGA",
                "strategy_params": {
                    "crossover_rate": 0.3,
                    "elite_ratio": 0.16666666666666666
                }
            }
        }
    }

def create_hyp2_config(name: str, out_dir: str, fn_name: str, D: int, P: int, G: int, seeds: list[int]) -> dict:
    elite_k = max(2, int(P * 0.16666666666666666))
    return {
        "experiment": {
            "name": name,
            "output_dir": out_dir,
            "shared": {
                "fitness": f"bbob:fn_name={fn_name},num_dims={D},maximize=false",
                "selection": f"elite_pool:num_selections={P},elite_k={elite_k}",
                "pop_size": P,
                "generations": G,
                "genome_length": D,
                "bounds": [-5.0, 5.0],
                "seeds": seeds,
                "backend": "malthusjax",
                "elitism": 0,
                "crossover": "uniform_real:crossover_rate=0.3",
                "serialize_history": False
            }
        },
        "pipelines": {
            "mjx_native_mutation": {
                "mutation": "gaussian:sigma=0.05"
            },
            "mjx_evosax_mutation_wrapper": {
                "mutation": "evosax_gaussian:mutation_strength=0.05"
            }
        }
    }

def create_hyp3_config(name: str, out_dir: str, fn_name: str, D: int, P: int, G: int, seeds: list[int]) -> dict:
    return {
        "experiment": {
            "name": name,
            "output_dir": out_dir,
            "shared": {
                "fitness": f"bbob:fn_name={fn_name},num_dims={D},maximize=false",
                "selection": f"tournament:num_selections={P},tournament_size=4",
                "mutation": "gaussian:sigma=0.1",
                "crossover": "uniform_real:crossover_rate=0.5",
                "pop_size": P,
                "generations": G,
                "genome_length": D,
                "bounds": [-5.0, 5.0],
                "seeds": seeds,
                "backend": "malthusjax",
                "elitism": 0,
                "serialize_history": False
            }
        },
        "pipelines": {
            "malthusjax_fp32": {
                "dtype": "float32"
            },
            "malthusjax_bf16": {
                "dtype": "bfloat16"
            }
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate LHS configs for thesis experiments")
    parser.add_argument("--k-samples", type=int, default=30, help="Number of LHS samples per benchmark function (default: 30, which is 10 x D)")
    parser.add_argument("--out-dir", type=str, default="configs/lhs_experiments", help="Output directory for generated configs")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of seeds to run per configuration")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Parameter Ranges
    # D: Genome Size [2, 100]
    # P: Population Size [10, 1000]
    # G: Generations [10, 1000]
    l_bounds = [2, 10, 10]
    u_bounds = [100, 1000, 1000]

    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=args.k_samples)
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    benchmarks = ["sphere", "rosenbrock", "rastrigin"]
    seeds = generate_seeds(args.num_seeds)

    config_count = 0
    
    # Save the LHS mapping for later analysis
    lhs_points_file = out_path / "lhs_points.csv"
    with open(lhs_points_file, "w") as f:
        f.write("point_idx,genome_size,pop_size,generations\\n")
        for i, point in enumerate(scaled_sample):
            D, P, G = int(round(point[0])), int(round(point[1])), int(round(point[2]))
            f.write(f"{i},{D},{P},{G}\\n")

    for i, point in enumerate(scaled_sample):
        # Round to integers
        D = int(round(point[0]))
        P = int(round(point[1]))
        G = int(round(point[2]))

        for fn_name in benchmarks:
            # Hyp 1
            h1_name = f"hyp1_{fn_name}_lhs{i}"
            h1_dir = f"results/thesis_lhs/hyp1/{fn_name}/{h1_name}"
            h1_conf = create_hyp1_config(h1_name, h1_dir, fn_name, D, P, G, seeds)
            with open(out_path / f"{h1_name}.toml", "w") as f:
                toml.dump(h1_conf, f)
            config_count += 1

            # Hyp 2
            h2_name = f"hyp2_{fn_name}_lhs{i}"
            h2_dir = f"results/thesis_lhs/hyp2/{fn_name}/{h2_name}"
            h2_conf = create_hyp2_config(h2_name, h2_dir, fn_name, D, P, G, seeds)
            with open(out_path / f"{h2_name}.toml", "w") as f:
                toml.dump(h2_conf, f)
            config_count += 1

            # Hyp 3
            h3_name = f"hyp3_{fn_name}_lhs{i}"
            h3_dir = f"results/thesis_lhs/hyp3/{fn_name}/{h3_name}"
            h3_conf = create_hyp3_config(h3_name, h3_dir, fn_name, D, P, G, seeds)
            with open(out_path / f"{h3_name}.toml", "w") as f:
                toml.dump(h3_conf, f)
            config_count += 1

    print(f"Generated {config_count} LHS configuration files in '{args.out_dir}'.")
    print(f"LHS parameter mapping saved to '{lhs_points_file}'.")

if __name__ == "__main__":
    main()
