#!/usr/bin/env python3
"""
JAX Dispatch Timing CLI for MalthusJAX

Analyzes JAX dispatch overhead, compilation times, and per-operator timing
using JAX's tracing facilities and Perfetto profiler.

Usage:
    python cli_dispatch.py config.toml              # Run full analysis
    python cli_dispatch.py config.toml --quick      # Quick smoke test
    python cli_dispatch.py config.toml --trace-only # Generate traces only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import tomli

# MalthusJAX imports
from malthusjax.core.genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.operators.crossover import BlendCrossover
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.selection import TournamentSelection


# =============================================================================
# Helper Functions
# =============================================================================


def create_bbob_evaluator(task: str, dim: int):
    """Create a BBOB evaluator for the given task and dimension."""
    bbob_config = BBOBConfig(fn_name=task, num_dims=dim, maximize=False)
    return BBOBEvaluator.create(bbob_config)


def get_fitness_fn_from_evaluator(evaluator: BBOBEvaluator):
    """
    Extract a pure fitness function from a BBOBEvaluator.
    
    Returns a function that takes a single genome array and returns fitness.
    """
    # Capture the evosax problem and state in a closure
    problem = evaluator.evosax_problem
    state = evaluator.evosax_state
    maximize = evaluator.config.maximize
    
    def fitness_fn(genome_values: jnp.ndarray) -> jnp.ndarray:
        """Pure fitness function for single genome (values array)."""
        # Expand dims for evosax batch interface
        x = genome_values[None, :]
        rng = jr.PRNGKey(0)
        fitness, _, _ = problem.eval(rng, x, state)
        result = fitness[0]
        if maximize:
            return -result
        return result
    
    return fitness_fn


# =============================================================================
# Dispatch Timing Utilities
# =============================================================================


@dataclass
class DispatchTimingResult:
    """Results from a single dispatch timing measurement."""

    name: str
    cold_compile_ms: float
    warm_dispatch_ms: float
    execution_ms: float
    total_cold_ms: float
    total_warm_ms: float
    shape_info: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class OperatorTimings:
    """Per-operator timing breakdown."""

    selection_ms: float = 0.0
    crossover_ms: float = 0.0
    mutation_ms: float = 0.0
    fitness_ms: float = 0.0
    total_step_ms: float = 0.0


def measure_dispatch_timing(
    fn: Callable,
    *args,
    name: str = "unnamed",
    warmup_runs: int = 3,
    timed_runs: int = 10,
    **kwargs,
) -> DispatchTimingResult:
    """
    Measure dispatch and execution timing for a JIT-compiled function.

    Args:
        fn: The function to measure (will be JIT-compiled if not already)
        *args: Arguments to pass to the function
        name: Name for logging
        warmup_runs: Number of warmup runs after first compilation
        timed_runs: Number of timed runs for averaging
        **kwargs: Keyword arguments for the function

    Returns:
        DispatchTimingResult with timing breakdown
    """
    # Ensure function is JIT-compiled
    jitted_fn = jax.jit(fn) if not hasattr(fn, "_cache_size") else fn

    # Cold run (includes compilation)
    jax.block_until_ready(args)  # Ensure inputs are ready
    cold_start = time.perf_counter()
    result = jitted_fn(*args, **kwargs)
    jax.block_until_ready(result)
    cold_end = time.perf_counter()
    cold_total_ms = (cold_end - cold_start) * 1000

    # Warmup runs (compilation cached, but may have other startup costs)
    for _ in range(warmup_runs):
        result = jitted_fn(*args, **kwargs)
        jax.block_until_ready(result)

    # Timed runs for warm dispatch measurement
    warm_times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = jitted_fn(*args, **kwargs)
        jax.block_until_ready(result)
        end = time.perf_counter()
        warm_times.append((end - start) * 1000)

    warm_avg_ms = sum(warm_times) / len(warm_times)

    # Estimate dispatch vs execution (dispatch is the overhead above minimum)
    min_warm_ms = min(warm_times)
    dispatch_overhead_ms = warm_avg_ms - min_warm_ms

    # Estimate cold compilation time
    compile_time_ms = cold_total_ms - warm_avg_ms

    return DispatchTimingResult(
        name=name,
        cold_compile_ms=max(0, compile_time_ms),
        warm_dispatch_ms=dispatch_overhead_ms,
        execution_ms=min_warm_ms,
        total_cold_ms=cold_total_ms,
        total_warm_ms=warm_avg_ms,
        shape_info={
            "input_shapes": [
                getattr(a, "shape", None) for a in args if hasattr(a, "shape")
            ]
        },
        metadata={"warmup_runs": warmup_runs, "timed_runs": timed_runs},
    )


def create_named_operator(fn: Callable, name: str) -> Callable:
    """Wrap an operator with jax.named_call for trace visibility."""
    return jax.named_call(fn, name=name)


# =============================================================================
# Traced Engine Components
# =============================================================================


def create_traced_evolution_step(
    selection_op,
    crossover_op,
    mutation_op,
    fitness_fn,
    genome_config: RealGenomeConfig,
    pop_size: int,
    num_elites: int,
):
    """
    Create a simplified evolution step with named_call annotations for tracing.

    This creates a traceable step that exercises all operators for timing analysis.
    """

    @jax.named_call
    def traced_selection(key, fitness_values):
        return selection_op(key, fitness_values)

    @jax.named_call
    def traced_fitness(population):
        return jax.vmap(fitness_fn)(population)

    def evolution_step(key, population, fitness_values):
        """
        Simplified evolution step for dispatch timing analysis.
        
        Uses direct operations that exercise the same kernels as full GA,
        but with simpler logic to avoid shape mismatches.
        """
        k1, k2, k3, k4 = jr.split(key, 4)

        # Selection - exercise selection operator
        selected_indices = traced_selection(k1, fitness_values)

        # Get selected individuals for breeding
        num_offspring = pop_size - num_elites
        
        # Simple parent selection: use first N selected indices
        parent_indices = selected_indices[:num_offspring]
        parents = population[parent_indices]

        # Mutation-only approach (simpler, exercises mutation kernel)
        # Generate mutation noise directly
        noise = jr.normal(k2, shape=parents.shape) * 0.1
        mutated = parents + noise
        
        # Clip to bounds
        min_val, max_val = genome_config.bounds
        mutated = jnp.clip(mutated, min_val, max_val)

        # Elitism - keep best individuals  
        elite_indices = jnp.argsort(fitness_values)[-num_elites:]
        elites = population[elite_indices]

        # Combine elites and offspring
        new_population = jnp.concatenate([elites, mutated], axis=0)

        # Evaluate fitness
        new_fitness = traced_fitness(new_population)

        return new_population, new_fitness

    return evolution_step


# =============================================================================
# Profiling with Perfetto
# =============================================================================


def run_with_perfetto_trace(
    fn: Callable,
    args: tuple,
    trace_dir: Path,
    trace_name: str,
    num_steps: int = 10,
) -> Path:
    """
    Run function with Perfetto tracing enabled.

    Args:
        fn: Function to trace (should be the evolution step)
        args: Initial arguments (key, population, fitness)
        trace_dir: Directory to save trace files
        trace_name: Base name for trace file

    Returns:
        Path to the generated trace file
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{trace_name}.json.gz"

    # Start profiler
    jax.profiler.start_trace(str(trace_dir))

    try:
        key, population, fitness = args

        # Run evolution steps
        for i in range(num_steps):
            key, subkey = jr.split(key)
            population, fitness = fn(subkey, population, fitness)
            jax.block_until_ready(fitness)

    finally:
        # Stop profiler and save trace
        jax.profiler.stop_trace()

    return trace_path


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_unroll_impact(
    config: dict,
    pop_size: int,
    dim: int,
    task: str,
    unroll_factors: list[int],
    key: jax.Array,
) -> list[dict]:
    """
    Analyze dispatch overhead at different unroll factors.

    Returns timing data for each unroll factor.
    """
    results = []

    # Create genome config
    genome_config = RealGenomeConfig(length=dim, bounds=(-5.0, 5.0))

    # Create fitness evaluator using BBOB
    evaluator = create_bbob_evaluator(task, dim)
    fitness_fn = get_fitness_fn_from_evaluator(evaluator)

    # Create operators
    mutation_rate = config.get("mutation_rate", 0.1)
    sigma = config.get("sigma", 0.1)
    crossover_rate = config.get("crossover_rate", 0.9)

    selection_op = TournamentSelection(num_selections=pop_size, tournament_size=3)
    crossover_op = BlendCrossover(
        alpha=0.5, crossover_rate=crossover_rate
    )
    mutation_op = GaussianMutation(
        mutation_rate=mutation_rate, mutation_strength=sigma
    )

    num_elites = max(1, int(pop_size * config.get("elite_ratio", 0.1)))

    for unroll in unroll_factors:
        print(f"  Testing unroll={unroll}...")

        # Create traced evolution step
        evolution_step = create_traced_evolution_step(
            selection_op,
            crossover_op,
            mutation_op,
            fitness_fn,
            genome_config,
            pop_size,
            num_elites,
        )

        # Create unrolled version
        def make_unrolled_step(step_fn, n_unroll):
            def unrolled(key, pop, fit):
                for _ in range(n_unroll):
                    key, subkey = jr.split(key)
                    pop, fit = step_fn(subkey, pop, fit)
                return pop, fit

            return unrolled

        unrolled_step = make_unrolled_step(evolution_step, unroll)

        # Initialize population
        key, init_key = jr.split(key)
        population = RealPopulation.init_random(init_key, genome_config, pop_size).genes.values
        fitness = jax.vmap(fitness_fn)(population)

        # Measure timing
        key, run_key = jr.split(key)
        timing = measure_dispatch_timing(
            unrolled_step,
            run_key,
            population,
            fitness,
            name=f"unroll_{unroll}",
            warmup_runs=3,
            timed_runs=10,
        )

        results.append(
            {
                "unroll_factor": unroll,
                "cold_compile_ms": timing.cold_compile_ms,
                "warm_dispatch_ms": timing.warm_dispatch_ms,
                "execution_ms": timing.execution_ms,
                "total_cold_ms": timing.total_cold_ms,
                "total_warm_ms": timing.total_warm_ms,
                "steps_per_dispatch": unroll,
                "ms_per_step": timing.total_warm_ms / unroll,
            }
        )

    return results


def analyze_operator_breakdown(
    config: dict,
    pop_size: int,
    dim: int,
    task: str,
    key: jax.Array,
) -> dict:
    """
    Measure dispatch timing for fundamental operations used in evolution.
    
    Instead of measuring the complex operator objects directly, we measure
    the underlying JAX operations that dominate dispatch timing.
    """
    # Create genome config
    genome_config = RealGenomeConfig(length=dim, bounds=(-5.0, 5.0))

    # Create fitness evaluator using BBOB
    evaluator = create_bbob_evaluator(task, dim)
    fitness_fn = get_fitness_fn_from_evaluator(evaluator)

    # Initialize test data
    key, init_key = jr.split(key)
    population = RealPopulation.init_random(init_key, genome_config, pop_size).genes.values
    fitness_values = jax.vmap(fitness_fn)(population)

    results = {}

    # Measure selection (tournament selection core operation)
    def selection_kernel(key, fitness):
        """Core selection operation: argmax over random tournament groups."""
        tournament_size = 3
        num_selections = pop_size
        # Generate random indices for tournaments
        indices = jr.randint(key, (num_selections, tournament_size), 0, len(fitness))
        # Get fitness values for tournaments
        tournament_fitness = fitness[indices]
        # Select winner (index with max fitness in each tournament)
        winners = indices[jnp.arange(num_selections), jnp.argmax(tournament_fitness, axis=1)]
        return winners

    key, sel_key = jr.split(key)
    sel_timing = measure_dispatch_timing(
        selection_kernel, sel_key, fitness_values, name="selection"
    )
    results["selection"] = {
        "cold_ms": sel_timing.total_cold_ms,
        "warm_ms": sel_timing.total_warm_ms,
        "compile_ms": sel_timing.cold_compile_ms,
    }

    # Measure crossover (blend crossover core operation)
    def crossover_kernel(key, parents1, parents2):
        """Core crossover: blend between parent pairs."""
        alpha = 0.5
        diff = jnp.abs(parents1 - parents2)
        cmin = jnp.minimum(parents1, parents2) - alpha * diff
        cmax = jnp.maximum(parents1, parents2) + alpha * diff
        # Sample uniformly in the blend range
        offspring = jr.uniform(key, parents1.shape, minval=cmin, maxval=cmax)
        return offspring

    key, cx_key = jr.split(key)
    cx_timing = measure_dispatch_timing(
        crossover_kernel, cx_key, population[::2], population[1::2], name="crossover"
    )
    results["crossover"] = {
        "cold_ms": cx_timing.total_cold_ms,
        "warm_ms": cx_timing.total_warm_ms,
        "compile_ms": cx_timing.cold_compile_ms,
    }

    # Measure mutation (gaussian mutation core operation)
    def mutation_kernel(key, genomes):
        """Core mutation: add gaussian noise with rate."""
        mutation_rate = 0.1
        mutation_strength = 0.1
        k1, k2 = jr.split(key)
        # Mutation mask
        mask = jr.bernoulli(k1, mutation_rate, genomes.shape)
        # Gaussian noise
        noise = jr.normal(k2, genomes.shape) * mutation_strength
        # Apply masked mutation
        mutated = jnp.where(mask, genomes + noise, genomes)
        # Clip to bounds
        return jnp.clip(mutated, -5.0, 5.0)

    key, mut_key = jr.split(key)
    mut_timing = measure_dispatch_timing(
        mutation_kernel, mut_key, population, name="mutation"
    )
    results["mutation"] = {
        "cold_ms": mut_timing.total_cold_ms,
        "warm_ms": mut_timing.total_warm_ms,
        "compile_ms": mut_timing.cold_compile_ms,
    }

    # Measure fitness evaluation (full population)
    batch_fitness = jax.vmap(fitness_fn)
    fit_timing = measure_dispatch_timing(batch_fitness, population, name="fitness")
    results["fitness"] = {
        "cold_ms": fit_timing.total_cold_ms,
        "warm_ms": fit_timing.total_warm_ms,
        "compile_ms": fit_timing.cold_compile_ms,
    }

    # Measure combined evolution step (full generation)
    def evolution_kernel(key, pop, fit):
        """One full evolution step."""
        k1, k2, k3, k4 = jr.split(key, 4)
        
        # Selection
        selected_indices = selection_kernel(k1, fit)
        parents = pop[selected_indices]
        
        # Mutation-based offspring
        offspring = mutation_kernel(k2, parents)
        
        # Elitism (keep top 10%)
        num_elites = max(1, pop_size // 10)
        elite_indices = jnp.argsort(fit)[-num_elites:]
        elites = pop[elite_indices]
        
        # Combine
        new_pop = jnp.concatenate([elites, offspring[:-num_elites]], axis=0)
        
        # Evaluate
        new_fit = batch_fitness(new_pop)
        
        return new_pop, new_fit

    key, evo_key = jr.split(key)
    evo_timing = measure_dispatch_timing(
        evolution_kernel, evo_key, population, fitness_values, name="evolution_step"
    )
    results["evolution_step"] = {
        "cold_ms": evo_timing.total_cold_ms,
        "warm_ms": evo_timing.total_warm_ms,
        "compile_ms": evo_timing.cold_compile_ms,
    }

    return results


# =============================================================================
# Report Generation
# =============================================================================


def generate_dispatch_report(
    unroll_results: list[dict],
    operator_results: dict,
    output_path: Path,
    config_info: dict,
) -> None:
    """Generate a comprehensive dispatch timing report."""

    report_lines = [
        "=" * 80,
        "MalthusJAX Dispatch Timing Analysis Report",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        "Configuration:",
        f"  Task: {config_info.get('task', 'N/A')}",
        f"  Dimension: {config_info.get('dim', 'N/A')}",
        f"  Population Size: {config_info.get('pop_size', 'N/A')}",
        f"  Device: {jax.devices()[0]}",
        "",
        "-" * 80,
        "UNROLL FACTOR ANALYSIS",
        "-" * 80,
        "",
        f"{'Unroll':<10} {'Cold (ms)':<12} {'Warm (ms)':<12} {'Compile (ms)':<14} {'ms/step':<10}",
        "-" * 60,
    ]

    for r in unroll_results:
        report_lines.append(
            f"{r['unroll_factor']:<10} "
            f"{r['total_cold_ms']:<12.2f} "
            f"{r['total_warm_ms']:<12.2f} "
            f"{r['cold_compile_ms']:<14.2f} "
            f"{r['ms_per_step']:<10.3f}"
        )

    report_lines.extend(
        [
            "",
            "-" * 80,
            "PER-OPERATOR TIMING BREAKDOWN",
            "-" * 80,
            "",
            f"{'Operator':<20} {'Cold (ms)':<12} {'Warm (ms)':<12} {'Compile (ms)':<14}",
            "-" * 60,
        ]
    )

    for op_name, timings in operator_results.items():
        report_lines.append(
            f"{op_name:<20} "
            f"{timings['cold_ms']:<12.2f} "
            f"{timings['warm_ms']:<12.2f} "
            f"{timings['compile_ms']:<14.2f}"
        )

    report_lines.extend(
        [
            "",
            "-" * 80,
            "INSIGHTS",
            "-" * 80,
            "",
        ]
    )

    # Calculate insights
    if unroll_results:
        baseline = unroll_results[0]
        best = min(unroll_results, key=lambda x: x["ms_per_step"])

        speedup = baseline["ms_per_step"] / best["ms_per_step"] if best["ms_per_step"] > 0 else 1.0
        dispatch_reduction = (
            (baseline["total_warm_ms"] - best["total_warm_ms"]) / baseline["total_warm_ms"] * 100
            if baseline["total_warm_ms"] > 0
            else 0
        )

        report_lines.extend(
            [
                f"• Best unroll factor: {best['unroll_factor']} ({best['ms_per_step']:.3f} ms/step)",
                f"• Speedup vs unroll=1: {speedup:.2f}x",
                f"• Dispatch overhead reduction: {dispatch_reduction:.1f}%",
                "",
            ]
        )

    # Operator insights
    if operator_results:
        total_compile = sum(v["compile_ms"] for v in operator_results.values())
        slowest_op = max(operator_results.items(), key=lambda x: x[1]["warm_ms"])
        report_lines.extend(
            [
                f"• Total compilation time: {total_compile:.2f} ms",
                f"• Slowest operator (warm): {slowest_op[0]} ({slowest_op[1]['warm_ms']:.2f} ms)",
                "",
            ]
        )

    report_lines.append("=" * 80)

    # Write report
    report_text = "\n".join(report_lines)
    output_path.write_text(report_text)
    print(f"\nReport saved to: {output_path}")
    print("\n" + report_text)


def save_results_csv(
    unroll_results: list[dict],
    operator_results: dict,
    output_dir: Path,
    config_info: dict,
) -> None:
    """Save results to CSV files for further analysis."""

    # Save unroll analysis
    unroll_csv = output_dir / "unroll_analysis.csv"
    with open(unroll_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unroll_results[0].keys()))
        writer.writeheader()
        writer.writerows(unroll_results)
    print(f"Unroll analysis saved to: {unroll_csv}")

    # Save operator breakdown
    operator_csv = output_dir / "operator_breakdown.csv"
    with open(operator_csv, "w", newline="") as f:
        fieldnames = ["operator", "cold_ms", "warm_ms", "compile_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for op_name, timings in operator_results.items():
            writer.writerow({"operator": op_name, **timings})
    print(f"Operator breakdown saved to: {operator_csv}")

    # Save config info
    config_json = output_dir / "config_info.json"
    with open(config_json, "w") as f:
        json.dump(config_info, f, indent=2, default=str)


# =============================================================================
# Main CLI
# =============================================================================


def load_config(config_path: Path) -> dict:
    """Load TOML configuration file."""
    with open(config_path, "rb") as f:
        return tomli.load(f)


def run_dispatch_analysis(config: dict, args: argparse.Namespace) -> None:
    """Run the full dispatch timing analysis."""

    # Extract configuration
    experiment = config.get("experiment", {})
    output_dir = Path(experiment.get("output_dir", "results/dispatch_timing"))
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = config.get("grid", {})
    tasks = grid.get("tasks", ["sphere"])
    dimensions = grid.get("dimensions", [50])
    pop_sizes = grid.get("pop_sizes", [128])
    seeds = grid.get("seeds", [42])

    dispatch_config = config.get("dispatch", {})
    unroll_factors = dispatch_config.get("unroll_factors", [1, 2, 4, 8, 16])

    hyperparam = config.get("hyperparam", {})

    print("=" * 60)
    print("MalthusJAX Dispatch Timing Analysis")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Output directory: {output_dir}")
    print(f"Tasks: {tasks}")
    print(f"Dimensions: {dimensions}")
    print(f"Population sizes: {pop_sizes}")
    print(f"Unroll factors: {unroll_factors}")
    print()

    for task in tasks:
        for dim in dimensions:
            for pop_size in pop_sizes:
                for seed in seeds:
                    print("-" * 60)
                    print(f"Analyzing: task={task}, dim={dim}, pop={pop_size}, seed={seed}")
                    print("-" * 60)

                    key = jr.PRNGKey(seed)
                    config_info = {
                        "task": task,
                        "dim": dim,
                        "pop_size": pop_size,
                        "seed": seed,
                        "unroll_factors": unroll_factors,
                        "device": str(jax.devices()[0]),
                    }

                    # Run unroll analysis
                    print("\n[1/3] Analyzing unroll factor impact...")
                    key, analysis_key = jr.split(key)
                    unroll_results = analyze_unroll_impact(
                        hyperparam, pop_size, dim, task, unroll_factors, analysis_key
                    )

                    # Run operator breakdown
                    print("\n[2/3] Measuring per-operator timing...")
                    key, op_key = jr.split(key)
                    operator_results = analyze_operator_breakdown(
                        hyperparam, pop_size, dim, task, op_key
                    )

                    # Generate Perfetto trace if requested
                    if args.trace:
                        print("\n[3/3] Generating Perfetto trace...")
                        trace_dir = output_dir / "traces"
                        trace_name = f"{task}_d{dim}_p{pop_size}_s{seed}"

                        # Create traced step
                        genome_config = RealGenomeConfig(
                            length=dim, bounds=(-5.0, 5.0)
                        )
                        evaluator = create_bbob_evaluator(task, dim)
                        fitness_fn = get_fitness_fn_from_evaluator(evaluator)

                        selection_op = TournamentSelection(
                            num_selections=pop_size, tournament_size=3
                        )
                        crossover_op = BlendCrossover(
                            alpha=0.5, crossover_rate=0.9
                        )
                        mutation_op = GaussianMutation(
                            mutation_rate=0.1, mutation_strength=0.1
                        )
                        num_elites = max(1, int(pop_size * 0.1))

                        evolution_step = create_traced_evolution_step(
                            selection_op,
                            crossover_op,
                            mutation_op,
                            fitness_fn,
                            genome_config,
                            pop_size,
                            num_elites,
                        )
                        evolution_step = jax.jit(evolution_step)

                        # Initialize and trace
                        key, init_key = jr.split(key)
                        population = RealPopulation.init_random(init_key, genome_config, pop_size).genes.values
                        fitness = jax.vmap(fitness_fn)(population)

                        trace_path = run_with_perfetto_trace(
                            evolution_step,
                            (key, population, fitness),
                            trace_dir,
                            trace_name,
                            num_steps=20,
                        )
                        print(f"  Trace saved to: {trace_path}")
                    else:
                        print("\n[3/3] Skipping Perfetto trace (use --trace to enable)")

                    # Save results
                    run_dir = output_dir / f"{task}_d{dim}_p{pop_size}_s{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    save_results_csv(unroll_results, operator_results, run_dir, config_info)
                    generate_dispatch_report(
                        unroll_results,
                        operator_results,
                        run_dir / "dispatch_report.txt",
                        config_info,
                    )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="JAX Dispatch Timing Analysis for MalthusJAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_dispatch.py dispatch_config.toml
  python cli_dispatch.py dispatch_config.toml --trace
  python cli_dispatch.py dispatch_config.toml --quick
        """,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Generate Perfetto traces for detailed analysis",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test with reduced settings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    config = load_config(args.config)

    # Apply quick mode overrides
    if args.quick:
        config.setdefault("grid", {})
        config["grid"]["dimensions"] = [10]
        config["grid"]["pop_sizes"] = [32]
        config.setdefault("dispatch", {})
        config["dispatch"]["unroll_factors"] = [1, 2, 4]

    # Apply output dir override
    if args.output_dir:
        config.setdefault("experiment", {})
        config["experiment"]["output_dir"] = str(args.output_dir)

    run_dispatch_analysis(config, args)
    return 0


if __name__ == "__main__":
    exit(main())
