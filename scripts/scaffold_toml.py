#!/usr/bin/env python3
"""
MalthusJAX TOML Scaffolding Tool

Automates the generation of compliant TOML configuration files for:
- Single-run experiments
- Multi-operator ablations
- Cross-backend parity and speed benchmarks
- Quality-Diversity (MAP-Elites) setups
- Composable Reinforcement Learning pipelines
- Combinatorial optimization with data registries
- High-throughput Cartesian grid & Latin Hypercube Sampling (LHS) benchmark suites
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from string import Template
from typing import Any, Dict

# --- RECIPE TEMPLATES ---

LOGGING_TEMPLATE = """[logging]
level = "INFO"
format_type = "color"
show_timestamps = false
log_interval = 10
log_nan_watchdog = true"""

SINGLE_RUN_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Single-run experiment: ${fitness} with ${pop_size} pop, ${generations} gens"
seeds = ${seeds}

${logging_block}

[pipelines.main]
fitness = "${fitness}"
selection = "tournament:num_selections=${num_selections},tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"
pop_size = ${pop_size}
generations = ${generations}
maximize = false
""")

ABLATION_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Operator ablation study on ${fitness}"
seeds = ${seeds}

${logging_block}

[experiment.shared]
fitness = "${fitness}"
pop_size = ${pop_size}
generations = ${generations}
seeds = ${seeds}
maximize = false
elitism = 2
selection = "tournament:num_selections=${num_selections},tournament_size=3"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.15"

[pipelines.uniform_crossover]
description = "Uniform real crossover (50% gene swap probability)"
crossover = "uniform_real"

[pipelines.blend_alpha]
description = "Blend-alpha crossover (BLX-0.5 expanded search)"
crossover = "blend:alpha=0.5"

[pipelines.simulated_binary]
description = "Simulated Binary Crossover (SBX eta=20)"
crossover = "simulated_binary:eta=20.0"
""")

PARITY_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Statistical parity alignment: MalthusJAX vs EvoSAX SimpleGA"
seeds = ${seeds}

${logging_block}

[experiment.shared]
fitness = "${fitness}"
pop_size = ${pop_size}
generations = ${generations}
seeds = ${seeds}

[pipelines.evosax_reference]
backend = "evosax"
evosax_strategy = "SimpleGA"

[pipelines.evosax_reference.strategy_params]
crossover_rate = 0.3
elite_ratio = 0.15625

[pipelines.malthusjax_clone]
backend = "malthusjax"
selection = "evosax_mimic_selection:num_selections=${pop_size},elite_k=${elite_k}"
crossover = "evosax_uniform_crossover:crossover_rate=0.3"
mutation = "evosax_gaussian:mutation_strength=1.0"
elitism = 0
""")

BACKEND_COMPARISON_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Multi-backend comparison across evolutionary algorithms"
seeds = ${seeds}

${logging_block}

[experiment.shared]
fitness = "${fitness}"
pop_size = ${pop_size}
generations = ${generations}
seeds = ${seeds}

[pipelines.malthusjax_native]
backend = "malthusjax"
engine_type = "fast_ga"
selection = "tournament:num_selections=${num_selections},tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"
elitism = 2

[pipelines.evosax_cmaes]
backend = "evosax"
evosax_strategy = "CMA_ES"

[pipelines.evosax_cmaes.strategy_params]
sigma_init = 0.5
elite_ratio = 0.5

[pipelines.evosax_openes]
backend = "evosax"
evosax_strategy = "OpenES"

[pipelines.evosax_openes.strategy_params]
sigma_init = 0.1
learning_rate = 0.05
""")

QUALITY_DIVERSITY_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Quality-Diversity MAP-Elites parameter study"
seeds = ${seeds}

${logging_block}

[experiment.shared]
fitness = "${fitness}"
pop_size = ${pop_size}
generations = ${generations}
seeds = ${seeds}

[pipelines.qdax_map_elites]
backend = "qdax"
qdax_strategy = "MAPElites"
grid_shape = [50, 50]
bd_extractors = ["identity", "identity"]

[pipelines.malthusjax_map_elites]
backend = "malthusjax"
engine_type = "map_elites"
map_elites_emitter = "mixing"
grid_shape = [50, 50]
""")

COMPOSABLE_RL_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Composable Reinforcement Learning with Brax and Neural Interpreter"
seeds = ${seeds}

${logging_block}

[pipelines.brax_ant_openes]
backend = "composable_evosax"
strategy = "OpenES"
pop_size = ${pop_size}
generations = ${generations}
maximize = true

[pipelines.brax_ant_openes.fitness]
type = "RLEvaluator"
max_steps = 1000

[pipelines.brax_ant_openes.fitness.env]
type = "BraxEnv"
env_name = "ant"

[pipelines.brax_ant_openes.fitness.interpreter]
type = "MLPInterpreter"
hidden = [64, 64]
activation = "swish"

[pipelines.brax_ant_openes.fitness.output]
type = "ScalarOutput"
maximize = true
""")

DATA_REGISTRY_TEMPLATE = Template("""[experiment]
name = "${name}"
output_dir = "results/${name}"
description = "Combinatorial Traveling Salesperson Problem on synthetic instances"
seeds = ${seeds}

${logging_block}

[experiment.shared]
genome_type = "real"
pop_size = ${pop_size}
generations = ${generations}
seeds = ${seeds}
maximize = false
elitism = 2

[data.synthetic_30]
source = "synthetic"
num_cities = 30
random_seed = 42
description = "Synthetic TSP: 30 random cities"

[pipelines.tournament_swap]
fitness = "tsp:data_id=synthetic_30"
selection = "tournament:num_selections=${num_selections},tournament_size=3"
mutation = "swap:mutation_rate=0.1"

[pipelines.elite_pool_swap]
fitness = "tsp:data_id=synthetic_30"
selection = "elite_pool:num_selections=${num_selections},elite_k=${elite_k}"
mutation = "swap:mutation_rate=0.15"
""")

BENCHMARK_CARTESIAN_TEMPLATE = Template("""[suite]
name = "${name}"
mode = "cartesian"
output_dir = "results/${name}"
num_seeds = ${num_seeds}
aggregation_level = "summary_only"

[grid]
functions = ["sphere", "rastrigin", "rosenbrock"]
dims = [10, 30, 50]
pops = [64, 256, 1024]
gens = [100, 500]

[analysis]
reference_pipeline = "malthusjax_baseline"
target_metrics = ["best_fitness", "execution_time"]

[pipelines.malthusjax_baseline]
backend = "malthusjax"
selection = "tournament:num_selections={pop_size},tournament_size=3"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"
elitism = 2

[pipelines.evosax_baseline]
backend = "evosax"
evosax_strategy = "SimpleGA"

[pipelines.evosax_baseline.strategy_params]
crossover_rate = 0.3
elite_ratio = 0.15625
""")

BENCHMARK_LHS_TEMPLATE = Template("""[suite]
name = "${name}"
mode = "lhs"
output_dir = "results/${name}"
num_seeds = ${num_seeds}
aggregation_level = "summary_only"

[grid]
functions = ["sphere", "rosenbrock", "rastrigin", "schwefel", "lunacek"]
dims_min = 10
dims_max = 500
pops_min = 64
pops_max = 2048
gens_min = 50
gens_max = 500
num_samples = 30

[analysis]
reference_pipeline = "malthusjax_candidate"
target_metrics = ["best_fitness", "execution_time", "warmup_time"]

[pipelines.malthusjax_candidate]
backend = "malthusjax"
selection = "evosax_mimic_selection:num_selections={pop_size},elite_k={elite_k}"
crossover = "batched_evosax_uniform:crossover_rate=0.3"
mutation = "batched_evosax_gaussian:mutation_strength=1.0"
elitism = 0

[pipelines.evosax_baseline]
backend = "evosax"
evosax_strategy = "SimpleGA"

[pipelines.evosax_baseline.strategy_params]
crossover_rate = 0.3
elite_ratio = 0.15625
""")

RECIPES: Dict[str, Dict[str, Any]] = {
    "single_run": {
        "template": SINGLE_RUN_TEMPLATE,
        "type": "composer",
        "description": "Quick single-pipeline experiment with JIT logging",
    },
    "ablation": {
        "template": ABLATION_TEMPLATE,
        "type": "composer",
        "description": "Multi-operator ablation study (uniform vs. blend vs. SBX)",
    },
    "parity": {
        "template": PARITY_TEMPLATE,
        "type": "composer",
        "description": "Cross-framework statistical parity test (MalthusJAX vs. EvoSAX)",
    },
    "backend_comparison": {
        "template": BACKEND_COMPARISON_TEMPLATE,
        "type": "composer",
        "description": "Multi-backend comparison (MalthusJAX vs. CMA-ES vs. OpenES)",
    },
    "quality_diversity": {
        "template": QUALITY_DIVERSITY_TEMPLATE,
        "type": "composer",
        "description": "MAP-Elites quality-diversity archive study",
    },
    "composable_rl": {
        "template": COMPOSABLE_RL_TEMPLATE,
        "type": "composer",
        "description": "Composable Reinforcement Learning pipeline (Brax + MLP)",
    },
    "data_registry": {
        "template": DATA_REGISTRY_TEMPLATE,
        "type": "composer",
        "description": "Combinatorial problem (TSP / Knapsack) with [data.*] registry",
    },
    "benchmark_cartesian": {
        "template": BENCHMARK_CARTESIAN_TEMPLATE,
        "type": "suite",
        "description": "Systematic Cartesian grid sweep benchmark suite",
    },
    "benchmark_lhs": {
        "template": BENCHMARK_LHS_TEMPLATE,
        "type": "suite",
        "description": "Latin Hypercube Sampling (LHS) scaling benchmark suite",
    },
}


def main(args_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MalthusJAX Automated TOML Configuration Scaffolder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Scaffold a single-run test config:
  python scripts/scaffold_toml.py -r single_run -o configs/test_sphere.toml

  # Scaffold an operator ablation study:
  python scripts/scaffold_toml.py -r ablation -o configs/ablation_rastrigin.toml -f "rastrigin:dim=30"

  # Scaffold a large-scale Latin Hypercube Sampling suite:
  python scripts/scaffold_toml.py -r benchmark_lhs -o configs/cluster_lhs.toml -n cluster_scaling -s 50
""",
    )
    parser.add_argument(
        "-r",
        "--recipe",
        choices=list(RECIPES.keys()),
        default="single_run",
        help="Task recipe template to generate (default: single_run)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Target output TOML filepath (required unless --list-recipes)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the experiment or suite (defaults to output file stem)",
    )
    parser.add_argument(
        "-f",
        "--fitness",
        type=str,
        default="sphere:dim=20",
        help="Fitness function specification string (default: 'sphere:dim=20')",
    )
    parser.add_argument(
        "-p",
        "--pop-size",
        type=int,
        default=64,
        help="Population size (default: 64)",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        help="Generation limit (default: 100)",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=str,
        default="[1, 2, 3]",
        help="Seeds list (e.g. '[1, 2, 3]' or integer count, default: '[1, 2, 3]')",
    )
    parser.add_argument(
        "--with-logging",
        action="store_true",
        default=True,
        help="Include pre-configured [logging] section (default: True)",
    )
    parser.add_argument(
        "--no-logging",
        action="store_false",
        dest="with_logging",
        help="Omit [logging] section from output",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="Display all available recipes and exit",
    )

    args = parser.parse_args(args_list)

    if args.list_recipes:
        print("\nAvailable MalthusJAX TOML Scaffolding Recipes:")
        print("-" * 75)
        for key, info in RECIPES.items():
            kind = f"[{info['type'].upper()}]"
            print(f"  {key:22s} {kind:12s} {info['description']}")
        print("-" * 75)
        return 0

    if not args.output:
        parser.error("the following arguments are required: -o/--output")

    exp_name = args.name or args.output.stem
    recipe_info = RECIPES[args.recipe]
    template = recipe_info["template"]

    # Formatting seeds
    seeds_val = args.seeds.strip()
    if seeds_val.isdigit():
        num_seeds = int(seeds_val)
        seeds_repr = str(list(range(1, num_seeds + 1)))
    else:
        seeds_repr = seeds_val
        num_seeds = 3

    num_selections = max(2, int(args.pop_size / 2))
    elite_k = max(2, int(args.pop_size / 6))
    logging_block = LOGGING_TEMPLATE if args.with_logging else ""

    rendered = template.substitute(
        name=exp_name,
        fitness=args.fitness,
        pop_size=args.pop_size,
        generations=args.generations,
        seeds=seeds_repr,
        num_seeds=num_seeds,
        num_selections=num_selections,
        elite_k=elite_k,
        logging_block=logging_block,
    )

    # Clean up empty lines if logging was omitted
    if not args.with_logging:
        rendered = "\n".join(line for line in rendered.splitlines() if line.strip()) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)

    print(f"\nSuccessfully scaffolded TOML configuration!")
    print(f"  Recipe:         {args.recipe} ({recipe_info['type'].upper()})")
    print(f"  Output File:    {args.output}")
    print(f"  Experiment:     {exp_name}")
    print(f"  Fitness:        {args.fitness}")
    print(f"  Population:     {args.pop_size} | Generations: {args.generations}")
    print(f"  Logging:        {'Pre-configured via [logging]' if args.with_logging else 'Disabled'}")

    if recipe_info["type"] == "composer":
        print(f"\nTo run this experiment:")
        print(f"  mjax run {args.output}")
        print(f"  # or programmatically:")
        print(f"  # from malthusjax.composer import Composer; res = Composer.from_toml('{args.output}')")
    else:
        print(f"\nTo execute this benchmarking suite:")
        print(f"  python scripts/benchmark_runner.py {args.output}")
        print(f"  # or run a quick smoke test:")
        print(f"  python scripts/benchmark_runner.py {args.output} --smoke")

    return 0


if __name__ == "__main__":
    sys.exit(main())
