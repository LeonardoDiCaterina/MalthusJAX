#!/usr/bin/env python3
"""
Complete BBOB Benchmark Workflow: Generate TOML files, launch, and analyze results.

This script provides a unified interface for:
1. Generating TOML files (if not present)
2. Launching all experiments with parallel execution
3. Aggregating and analyzing results

Usage:
    # Full workflow (generate + launch)
    python bbob_benchmark_workflow.py --generate --launch --max-parallel 2

    # Just generate TOML files
    python bbob_benchmark_workflow.py --generate

    # Launch pre-generated TOML files
    python bbob_benchmark_workflow.py --launch --max-parallel 2

    # Analyze completed results
    python bbob_benchmark_workflow.py --analyze
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json


def run_command(cmd: list, description: str) -> int:
    """Run a command and report status."""
    print()
    print("=" * 70)
    print(f"▶ {description}")
    print("=" * 70)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
    
    return result.returncode


def generate_toml_files(
    output_dir: Path,
    fn_start: int = 1,
    fn_end: int = 24,
    dimensions: list = None,
    pop_sizes: list = None,
    num_seeds: int = 100,
    generations: int = 500,
) -> int:
    """Generate TOML files."""
    if dimensions is None:
        dimensions = [10, 50]
    if pop_sizes is None:
        pop_sizes = [1023, 1024, 1025, 1026, 511, 512, 513, 515]
    
    cmd = [
        "python",
        "scripts/generate_bbob_benchmark.py",
        "--output-dir",
        str(output_dir),
        "--fn-range",
        str(fn_start),
        str(fn_end),
        "--dimensions",
        *[str(d) for d in dimensions],
        "--pop-sizes",
        *[str(p) for p in pop_sizes],
        "--num-seeds",
        str(num_seeds),
        "--generations",
        str(generations),
        "--create-launcher",
    ]
    
    return run_command(cmd, "Generate TOML files")


def launch_experiments(
    toml_dir: Path,
    output_dir: Path = None,
    max_parallel: int = 1,
    cleanup_ram: bool = True,
) -> int:
    """Launch all experiments."""
    cmd = [
        "python",
        "scripts/launch_bbob_benchmark.py",
        "--toml-dir",
        str(toml_dir),
        "--max-parallel",
        str(max_parallel),
    ]
    
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    
    if cleanup_ram:
        cmd.append("--cleanup-ram")
    else:
        cmd.append("--no-cleanup-ram")
    
    return run_command(cmd, "Launch BBOB experiments")


def analyze_results(results_dir: Path) -> int:
    """Analyze completed results."""
    print()
    print("=" * 70)
    print("▶ Analyze Results")
    print("=" * 70)
    
    # Find all result directories
    result_dirs = sorted(results_dir.glob("bbob_benchmark/fn*/results/traces"))
    
    if not result_dirs:
        print("❌ No results found in:", results_dir / "bbob_benchmark")
        return 1
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Count completed runs
    total_runs = 0
    for result_dir in result_dirs:
        num_files = len(list(result_dir.glob("*.json")))
        total_runs += num_files
    
    print(f"Total trace files: {total_runs}")
    
    # TODO: Add more sophisticated analysis
    # - Aggregated statistics per function
    # - Performance plots
    # - Convergence analysis
    
    print("✅ Result analysis completed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="BBOB Benchmark Workflow: Generate, Launch, and Analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/_DEMO_COMPOSER"),
        help="Base output directory",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate TOML files",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch experiments",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze results",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum parallel experiments",
    )
    parser.add_argument(
        "--cleanup-ram",
        action="store_true",
        default=True,
        help="Clean up RAM after each experiment",
    )
    parser.add_argument(
        "--no-cleanup-ram",
        action="store_false",
        dest="cleanup_ram",
        help="Disable RAM cleanup",
    )
    parser.add_argument(
        "--fn-start",
        type=int,
        default=1,
        help="Starting BBOB function",
    )
    parser.add_argument(
        "--fn-end",
        type=int,
        default=24,
        help="Ending BBOB function",
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
        help="Number of seeds",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Number of generations",
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not (args.generate or args.launch or args.analyze):
        parser.print_help()
        print()
        print("Examples:")
        print("  # Full workflow (generate + launch with 2 parallel jobs)")
        print("  python bbob_benchmark_workflow.py --generate --launch --max-parallel 2")
        print()
        print("  # Just generate TOML files for functions 1-5")
        print("  python bbob_benchmark_workflow.py --generate --fn-start 1 --fn-end 5")
        print()
        print("  # Launch pre-generated experiments")
        print("  python bbob_benchmark_workflow.py --launch --max-parallel 2")
        print()
        return 0
    
    toml_dir = args.output_dir / "bbob_benchmark"
    exit_code = 0
    
    # Generate phase
    if args.generate:
        exit_code = generate_toml_files(
            output_dir=toml_dir,
            fn_start=args.fn_start,
            fn_end=args.fn_end,
            dimensions=args.dimensions,
            pop_sizes=args.pop_sizes,
            num_seeds=args.num_seeds,
            generations=args.generations,
        )
        if exit_code != 0:
            return exit_code
    
    # Launch phase
    if args.launch:
        exit_code = launch_experiments(
            toml_dir=toml_dir,
            output_dir=args.output_dir,
            max_parallel=args.max_parallel,
            cleanup_ram=args.cleanup_ram,
        )
        if exit_code != 0:
            return exit_code
    
    # Analyze phase
    if args.analyze:
        exit_code = analyze_results(results_dir=args.output_dir)
        if exit_code != 0:
            return exit_code
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
