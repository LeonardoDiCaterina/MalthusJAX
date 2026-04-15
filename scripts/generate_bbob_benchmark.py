#!/usr/bin/env python3
"""
Generate BBOB benchmark TOML files and launcher script.

This script generates:
1. TOML files for all 24 BBOB functions × 2 dimensions × 8 pop sizes × 7 strategies
2. A launcher script (bash) that runs all experiments in nohup with RAM cleanup
3. A results aggregator

Usage:
    python generate_bbob_benchmark.py --output-dir examples/_DEMO_COMPOSER/bbob_benchmark
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple


def generate_seeds(count: int = 100, start: int = 1) -> List[int]:
    """Generate seed list."""
    return list(range(start, start + count))


def generate_toml_content(
    fn: int,
    dimensions: List[int],
    pop_sizes: List[int],
    seeds: List[int],
    strategies: Dict[str, Dict],
    generations: int = 500,
) -> str:
    """Generate TOML content for a single BBOB function."""
    
    seeds_str = ", ".join(str(s) for s in seeds)
    dim_list = ", ".join(str(d) for d in dimensions)
    
    toml_lines = [
        "[experiment]",
        f'name = "bbob_fn{fn:02d}_sweep"',
        f'output_dir = "results/bbob_benchmark/fn{fn:02d}"',
        f'description = "BBOB Function {fn}: sweep across dims={{{dim_list}}}, pop_sizes={{{pop_sizes[0]}-{pop_sizes[-1]}}}, strategies"',
        "",
        "[experiment.shared]",
        f'fitness = "bbob:fn={fn}"',
        "engine_type = \"ga\"",
        "genome_type = \"real\"",
        "bounds = [-5.0, 5.0]",
        f"elitism = 0",
        f"track_best = 0",
        f"generations = {generations}",
        f"seeds = [{seeds_str}]",
        "",
    ]
    
    # Generate pipelines
    pipeline_count = 0
    for dim in dimensions:
        for pop_size in pop_sizes:
            for strat_name, strat_config in strategies.items():
                pipeline_name = f"{strat_name}_{dim}d_{pop_size}"
                
                toml_lines.append(f"[pipelines.{pipeline_name}]")
                toml_lines.append(f"genome_length = {dim}")
                toml_lines.append(f"pop_size = {pop_size}")
                
                # Apply strategy-specific overrides
                for key, value in strat_config.items():
                    if isinstance(value, str):
                        toml_lines.append(f'{key} = "{value}"')
                    elif isinstance(value, bool):
                        toml_lines.append(f'{key} = {"true" if value else "false"}')
                    else:
                        toml_lines.append(f'{key} = {value}')
                
                toml_lines.append("")
                pipeline_count += 1
    
    return "\n".join(toml_lines)


def generate_launcher_script(
    toml_dir: Path,
    toml_files: List[str],
    nohup_dir: Path,
    log_dir: Path,
    python_exec: str = "python",
) -> str:
    """Generate bash launcher script that runs all TOML files with RAM cleanup."""
    
    bash_lines = [
        "#!/bin/bash",
        "# Auto-generated BBOB benchmark launcher",
        "# Runs all TOML experiments in nohup with RAM cleanup between runs",
        "",
        f"TOML_DIR={toml_dir}",
        f"NOHUP_DIR={nohup_dir}",
        f"LOG_DIR={log_dir}",
        "PYTHON_EXEC=" + python_exec,
        "",
        "mkdir -p \"$NOHUP_DIR\" \"$LOG_DIR\"",
        "",
        "echo \"Starting BBOB benchmark suite at $(date)\"",
        "echo \"TOML directory: $TOML_DIR\"",
        "echo \"Nohup directory: $NOHUP_DIR\"",
        "echo \"Log directory: $LOG_DIR\"",
        "echo \"\"",
        "",
    ]
    
    for i, toml_file in enumerate(toml_files, 1):
        toml_path = toml_dir / toml_file
        log_file = log_dir / f"{toml_file.replace('.toml', '')}.log"
        nohup_file = nohup_dir / f"{toml_file.replace('.toml', '')}.out"
        
        bash_lines.append(f"# Run {i}/{len(toml_files)}: {toml_file}")
        bash_lines.append("echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"")
        bash_lines.append(f"echo \"[{i}/{len(toml_files)}] Starting: {toml_file}\"")
        bash_lines.append("echo \"Time: $(date)\"")
        bash_lines.append("echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"")
        bash_lines.append("")
        
        # Run in nohup with output logging
        bash_lines.append(
            f"nohup $PYTHON_EXEC -c "
            f'"from malthusjax.composer import Composer; '
            f'Composer.from_toml(\\"{toml_path}\\") '
            f"\" > \\\"$NOHUP_FILE\\\" 2>&1 &"
        )
        bash_lines.append("")
        bash_lines.append("NOHUP_PID=$!")
        bash_lines.append("echo \"Process PID: $NOHUP_PID\"")
        bash_lines.append("echo \"Nohup output: $NOHUP_FILE\"")
        bash_lines.append("")
        
        # Wait for process and log completion
        bash_lines.append("wait $NOHUP_PID")
        bash_lines.append("EXIT_CODE=$?")
        bash_lines.append("echo \"[{i}/{len(toml_files)}] Completed: {toml_file} (exit code: $EXIT_CODE)\" | tee -a \"$LOG_DIR/completion.log\"")
        bash_lines.append("")
        
        # RAM cleanup
        bash_lines.append("# Clean up RAM")
        bash_lines.append("echo \"Cleaning up RAM...\"")
        bash_lines.append("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true")
        bash_lines.append("sleep 2")
        bash_lines.append("echo \"\"")
        bash_lines.append("")
    
    bash_lines.append("echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"")
    bash_lines.append("echo \"All experiments completed at $(date)\"")
    bash_lines.append("echo \"Check logs in: $LOG_DIR\"")
    bash_lines.append("")
    
    return "\n".join(bash_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BBOB benchmark TOML files and launcher"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/_DEMO_COMPOSER/bbob_benchmark"),
        help="Output directory for TOML files",
    )
    parser.add_argument(
        "--fn-range",
        type=int,
        nargs=2,
        default=[1, 24],
        metavar=("START", "END"),
        help="BBOB function range (inclusive)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[10, 50],
        help="Dimensions to test",
    )
    parser.add_argument(
        "--pop-sizes",
        type=int,
        nargs="+",
        default=[1023, 1024, 1025, 1026, 511, 512, 513, 515],
        help="Population sizes to test",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of seeds per experiment",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Number of generations per run",
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default="python",
        help="Python executable path",
    )
    parser.add_argument(
        "--create-launcher",
        action="store_true",
        help="Generate launcher script (requires all TOML files to exist)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fn_start, fn_end = args.fn_range
    dimensions = args.dimensions
    pop_sizes = args.pop_sizes
    num_seeds = args.num_seeds
    generations = args.generations
    
    seeds = generate_seeds(num_seeds)
    
    # Define strategies (4 MalthusJAX + 3 Evosax)
    strategies = {
        "malthusjax_default": {
            "backend": "malthusjax",
            "selection": "elite_pool:num_selections=100,elite_k=50",
            "crossover": "uniform_real",
            "mutation": "gaussian:mutation_rate=0.1",
        },
        "malthusjax_roulette": {
            "backend": "malthusjax",
            "selection": "roulette:num_selections=100,temperature=1.0",
            "crossover": "uniform_real",
            "mutation": "gaussian:mutation_rate=0.1",
        },
        "malthusjax_tournament": {
            "backend": "malthusjax",
            "selection": "tournament:num_selections=100,tournament_size=3",
            "crossover": "uniform_real",
            "mutation": "gaussian:mutation_rate=0.1",
        },
        "malthusjax_evosaxops": {
            "backend": "malthusjax",
            "selection": "elite_pool:num_selections=100,elite_k=50",
            "crossover": "evosax_uniform_crossover",
            "mutation": "evosax_gaussian",
        },
        "evosax_simplega": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
        },
        "evosax_de": {
            "backend": "evosax",
            "evosax_strategy": "DifferentialEvolution",
        },
        "evosax_mr15ga": {
            "backend": "evosax",
            "evosax_strategy": "MR15_GA",
        },
    }
    
    # Generate TOML files
    print(f"Generating TOML files for BBOB functions {fn_start}-{fn_end}...")
    print(f"  Dimensions: {dimensions}")
    print(f"  Pop sizes: {pop_sizes}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Strategies: {len(strategies)}")
    print()
    
    toml_files = []
    for fn in range(fn_start, fn_end + 1):
        toml_filename = f"bbob_fn{fn:02d}.toml"
        toml_path = output_dir / toml_filename
        
        content = generate_toml_content(
            fn=fn,
            dimensions=dimensions,
            pop_sizes=pop_sizes,
            seeds=seeds,
            strategies=strategies,
            generations=generations,
        )
        
        toml_path.write_text(content)
        toml_files.append(toml_filename)
        
        num_pipelines = len(dimensions) * len(pop_sizes) * len(strategies)
        print(f"✓ {toml_filename} ({num_pipelines} pipelines)")
    
    print()
    print(f"✓ Generated {len(toml_files)} TOML files in {output_dir}")
    print()
    
    # Generate launcher script
    if args.create_launcher:
        nohup_dir = output_dir.parent / "nohup"
        log_dir = output_dir.parent / "logs"
        
        launcher_content = generate_launcher_script(
            toml_dir=output_dir,
            toml_files=toml_files,
            nohup_dir=nohup_dir,
            log_dir=log_dir,
            python_exec=args.python_exec,
        )
        
        launcher_path = output_dir.parent / "launch_bbob_benchmark.sh"
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)
        
        print(f"✓ Generated launcher script: {launcher_path}")
        print()
        print("To run the benchmark suite:")
        print(f"  chmod +x {launcher_path}")
        print(f"  {launcher_path}")
        print()
    
    # Print summary
    total_pipelines = len(dimensions) * len(pop_sizes) * len(strategies) * len(toml_files)
    total_runs = total_pipelines * num_seeds
    
    print("=" * 70)
    print("BENCHMARK CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"BBOB Functions:        {fn_start}-{fn_end} (24 functions)")
    print(f"Dimensions:            {dimensions} ({len(dimensions)} dims)")
    print(f"Population Sizes:      {pop_sizes} ({len(pop_sizes)} sizes)")
    print(f"Strategies:            {len(strategies)} ({', '.join(strategies.keys())})")
    print(f"Seeds:                 {num_seeds}")
    print(f"Generations:           {generations}")
    print()
    print(f"Pipelines per function: {len(dimensions) * len(pop_sizes) * len(strategies)}")
    print(f"Total pipelines:        {total_pipelines:,}")
    print(f"Total evolutionary runs: {total_runs:,}")
    print()
    print(f"Output directory:      {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
