#!/usr/bin/env python3
"""
MalthusJAX Adapter Profiler with Multi-Run Statistics

Profiles MalthusJAX engines from the registry with statistical analysis
over multiple runs. Generates Perfetto traces and timing statistics.

Usage:
    python profile_adapters.py --task sphere --dim 10 --pop-size 64 --gens 50
    python profile_adapters.py --task sphere --dim 50 --runs 30 --engines Standard_GA MR15_GA
    python profile_adapters.py --task sphere --dim 10 --gens 50 --unroll 1 4 8 --runs 30
    python profile_adapters.py --list-engines
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import jax.random as jr
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from framework.adapters import setup_bbob_instances
from framework.registry import ComparisonRegistry

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunResult:
    """Single run timing result."""
    run_id: int
    engine: str
    unroll: int
    task: str
    dim: int
    pop_size: int
    num_gens: int
    cold_ms: float
    warm_ms: float
    compile_ms: float
    ms_per_step: float


@dataclass
class StatsSummary:
    """Statistical summary for a configuration."""
    engine: str
    unroll: int
    task: str
    dim: int
    pop_size: int
    num_gens: int
    n_runs: int
    # Warm execution stats
    warm_mean: float
    warm_std: float
    warm_min: float
    warm_max: float
    warm_p5: float
    warm_p50: float
    warm_p95: float
    warm_ci_low: float
    warm_ci_high: float
    warm_cv: float  # Coefficient of variation
    # Compile stats
    compile_mean: float
    compile_std: float


# =============================================================================
# Timing Functions
# =============================================================================

def measure_single_run(
    fn: Callable,
    *args,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> tuple[float, float, float]:
    """
    Measure timing for a single profiling run.

    Returns:
        (cold_ms, warm_avg_ms, compile_ms)
    """
    jitted_fn = jax.jit(fn)

    # Cold run (includes compilation)
    jax.block_until_ready(args)
    cold_start = time.perf_counter()
    result = jitted_fn(*args)
    jax.block_until_ready(result)
    cold_end = time.perf_counter()
    cold_ms = (cold_end - cold_start) * 1000

    # Warmup
    for _ in range(warmup_runs):
        result = jitted_fn(*args)
        jax.block_until_ready(result)

    # Timed runs
    warm_times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = jitted_fn(*args)
        jax.block_until_ready(result)
        end = time.perf_counter()
        warm_times.append((end - start) * 1000)

    warm_avg = np.mean(warm_times)
    compile_ms = max(0, cold_ms - warm_avg)

    return cold_ms, warm_avg, compile_ms


# =============================================================================
# Engine Functions
# =============================================================================

def get_available_engines() -> list[str]:
    """Get all MalthusJAX engines from registry."""
    engines = []
    for name in ComparisonRegistry._registry.keys():
        spec = ComparisonRegistry.get(name)
        if spec.malthus_factory is not None:
            engines.append(name)
    return engines


def build_adapter(engine_name: str, task: str, dim: int, pop_size: int, seed: int):
    """Build adapter from registry."""
    evaluator, _ = setup_bbob_instances(task, dim, seed)
    spec = ComparisonRegistry.get(engine_name)
    return spec.malthus_factory(pop_size, dim, seed, spec.default_hypers, evaluator)


# =============================================================================
# Multi-Run Profiling
# =============================================================================

def profile_configuration(
    engine_name: str,
    task: str,
    dim: int,
    pop_size: int,
    num_gens: int,
    unroll: int,
    n_runs: int,
    seed: int,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> list[RunResult]:
    """
    Profile a single engine/unroll configuration over multiple runs.
    """
    results = []

    for run_id in range(n_runs):
        # Use same seed for algorithm, different key split for isolation
        adapter = build_adapter(engine_name, task, dim, pop_size, seed)

        key = jr.PRNGKey(seed + run_id)  # Vary key per run for system variance
        init_state = adapter.init(key)
        step_fn = adapter.make_step_fn()

        def evolution_loop(state):
            final, _ = jax.lax.scan(step_fn, state, None, length=num_gens, unroll=unroll)
            return final

        cold_ms, warm_ms, compile_ms = measure_single_run(
            evolution_loop,
            init_state,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

        results.append(RunResult(
            run_id=run_id,
            engine=engine_name,
            unroll=unroll,
            task=task,
            dim=dim,
            pop_size=pop_size,
            num_gens=num_gens,
            cold_ms=cold_ms,
            warm_ms=warm_ms,
            compile_ms=compile_ms,
            ms_per_step=warm_ms / num_gens,
        ))

        print(".", end="", flush=True)

    return results


def compute_statistics(results: list[RunResult]) -> StatsSummary:
    """Compute statistical summary from multiple runs."""
    if not results:
        raise ValueError("No results to summarize")

    r0 = results[0]
    warm_times = [r.warm_ms for r in results]
    compile_times = [r.compile_ms for r in results]

    warm_arr = np.array(warm_times)
    compile_arr = np.array(compile_times)

    # 95% confidence interval
    ci = stats.t.interval(
        0.95,
        len(warm_arr) - 1,
        loc=np.mean(warm_arr),
        scale=stats.sem(warm_arr)
    ) if len(warm_arr) > 1 else (np.mean(warm_arr), np.mean(warm_arr))

    return StatsSummary(
        engine=r0.engine,
        unroll=r0.unroll,
        task=r0.task,
        dim=r0.dim,
        pop_size=r0.pop_size,
        num_gens=r0.num_gens,
        n_runs=len(results),
        # Warm stats
        warm_mean=np.mean(warm_arr),
        warm_std=np.std(warm_arr, ddof=1) if len(warm_arr) > 1 else 0,
        warm_min=np.min(warm_arr),
        warm_max=np.max(warm_arr),
        warm_p5=np.percentile(warm_arr, 5),
        warm_p50=np.percentile(warm_arr, 50),
        warm_p95=np.percentile(warm_arr, 95),
        warm_ci_low=ci[0],
        warm_ci_high=ci[1],
        warm_cv=(
            np.std(warm_arr, ddof=1)
            / np.mean(warm_arr)
            * 100
            if np.mean(warm_arr) > 0
            else 0
        ),
        # Compile stats
        compile_mean=np.mean(compile_arr),
        compile_std=np.std(compile_arr, ddof=1) if len(compile_arr) > 1 else 0,
    )


# =============================================================================
# Perfetto Tracing
# =============================================================================

def run_perfetto_trace(
    engine_name: str,
    task: str,
    dim: int,
    pop_size: int,
    num_gens: int,
    unroll: int,
    seed: int,
    trace_dir: Path,
) -> Path:
    """Run engine with Perfetto tracing enabled."""
    adapter = build_adapter(engine_name, task, dim, pop_size, seed)

    key = jr.PRNGKey(seed)
    init_state = adapter.init(key)
    step_fn = adapter.make_step_fn()

    def traced_loop(state):
        final, _ = jax.lax.scan(step_fn, state, None, length=num_gens, unroll=unroll)
        return final

    # Compile first
    jit_loop = jax.jit(traced_loop)
    _ = jit_loop(init_state)
    jax.block_until_ready(_)

    # Create trace directory
    trace_subdir = trace_dir / f"{engine_name}_unroll{unroll}"
    trace_subdir.mkdir(parents=True, exist_ok=True)

    jax.profiler.start_trace(str(trace_subdir))
    try:
        result = jit_loop(init_state)
        jax.block_until_ready(result)
    finally:
        jax.profiler.stop_trace()

    return trace_subdir


# =============================================================================
# Output Generation
# =============================================================================

def save_raw_results(results: list[RunResult], output_path: Path):
    """Save all individual run results to CSV."""
    fieldnames = [
        "run_id", "engine", "unroll", "task", "dim", "pop_size",
        "num_gens", "cold_ms", "warm_ms", "compile_ms", "ms_per_step"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": r.run_id,
                "engine": r.engine,
                "unroll": r.unroll,
                "task": r.task,
                "dim": r.dim,
                "pop_size": r.pop_size,
                "num_gens": r.num_gens,
                "cold_ms": f"{r.cold_ms:.3f}",
                "warm_ms": f"{r.warm_ms:.3f}",
                "compile_ms": f"{r.compile_ms:.3f}",
                "ms_per_step": f"{r.ms_per_step:.4f}",
            })


def save_statistics(stats_list: list[StatsSummary], output_path: Path):
    """Save statistical summaries to CSV."""
    fieldnames = [
        "engine", "unroll", "task", "dim", "pop_size", "num_gens", "n_runs",
        "warm_mean", "warm_std", "warm_min", "warm_max",
        "warm_p5", "warm_p50", "warm_p95",
        "warm_ci_low", "warm_ci_high", "warm_cv",
        "compile_mean", "compile_std"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats_list:
            writer.writerow({
                "engine": s.engine,
                "unroll": s.unroll,
                "task": s.task,
                "dim": s.dim,
                "pop_size": s.pop_size,
                "num_gens": s.num_gens,
                "n_runs": s.n_runs,
                "warm_mean": f"{s.warm_mean:.3f}",
                "warm_std": f"{s.warm_std:.3f}",
                "warm_min": f"{s.warm_min:.3f}",
                "warm_max": f"{s.warm_max:.3f}",
                "warm_p5": f"{s.warm_p5:.3f}",
                "warm_p50": f"{s.warm_p50:.3f}",
                "warm_p95": f"{s.warm_p95:.3f}",
                "warm_ci_low": f"{s.warm_ci_low:.3f}",
                "warm_ci_high": f"{s.warm_ci_high:.3f}",
                "warm_cv": f"{s.warm_cv:.2f}",
                "compile_mean": f"{s.compile_mean:.3f}",
                "compile_std": f"{s.compile_std:.3f}",
            })


def print_statistics_report(stats_list: list[StatsSummary]):
    """Print a formatted statistics report."""
    print("\n" + "=" * 90)
    print("STATISTICAL SUMMARY")
    print("=" * 90)
    print(f"{'Engine':<22} {'Unroll':<8} {'Mean (ms)':<12} {'Std':<10} {'95% CI':<20} {'CV%':<8}")
    print("-" * 90)

    for s in stats_list:
        ci_str = f"[{s.warm_ci_low:.2f}, {s.warm_ci_high:.2f}]"
        print(
            f"{s.engine:<22} {s.unroll:<8} {s.warm_mean:<12.3f} "
            f"{s.warm_std:<10.3f} {ci_str:<20} {s.warm_cv:<8.2f}"
        )

    print("-" * 90)

    # Find best configuration
    if stats_list:
        best = min(stats_list, key=lambda x: x.warm_mean)
        most_stable = min(stats_list, key=lambda x: x.warm_cv)
        print(
            f"\nFastest: {best.engine} (unroll={best.unroll}) - "
            f"{best.warm_mean:.3f} ms"
        )
        print(
            f"Most stable: {most_stable.engine} (unroll={most_stable.unroll}) - "
            f"CV={most_stable.warm_cv:.2f}%"
        )

    print("=" * 90)


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MalthusJAX Adapter Profiler with Multi-Run Statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python profile_adapters.py --task sphere --dim 10 --pop-size 64 --gens 50
    python profile_adapters.py --task sphere --dim 50 --runs 30 --engines Standard_GA MR15_GA
    python profile_adapters.py --task sphere --dim 10 --unroll 1 4 8 --runs 30
    python profile_adapters.py --list-engines
    python profile_adapters.py --task sphere --dim 10 --with-trace
        """
    )

    parser.add_argument("--task", default="sphere", help="BBOB problem (default: sphere)")
    parser.add_argument("--dim", type=int, default=10, help="Dimension (default: 10)")
    parser.add_argument(
        "--pop-size",
        type=int,
        default=64,
        help="Population size (default: 64)",
    )
    parser.add_argument(
        "--gens",
        type=int,
        default=50,
        help="Generations (default: 50)",
    )
    parser.add_argument(
        "--unroll",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Unroll factors (default: 1 4 8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of runs for statistics (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/profiler"),
        help="Output directory",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Engines to profile (default: all)",
    )
    parser.add_argument(
        "--with-trace",
        action="store_true",
        help="Generate Perfetto traces",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available engines",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup runs per measurement (default: 3)",
    )
    parser.add_argument(
        "--timed",
        type=int,
        default=10,
        help="Timed runs per measurement (default: 10)",
    )

    args = parser.parse_args()

    if args.list_engines:
        print("Available MalthusJAX engines:")
        for name in get_available_engines():
            print(f"  {name}")
        return

    # Get engines
    available = get_available_engines()
    if args.engines:
        engines = [e for e in args.engines if e in available]
        if not engines:
            print(f"No valid engines. Available: {available}")
            return
    else:
        engines = available

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{args.task}_d{args.dim}_p{args.pop_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MalthusJAX Adapter Profiler")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Task: {args.task} | Dim: {args.dim} | Pop: {args.pop_size} | Gens: {args.gens}")
    print(f"Unroll factors: {args.unroll}")
    print(f"Runs per config: {args.runs}")
    print(f"Engines: {engines}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_results = []
    all_stats = []

    for engine_name in engines:
        for unroll in args.unroll:
            print(f"\n{engine_name} [unroll={unroll}]: ", end="", flush=True)

            try:
                # Run profiling
                results = profile_configuration(
                    engine_name=engine_name,
                    task=args.task,
                    dim=args.dim,
                    pop_size=args.pop_size,
                    num_gens=args.gens,
                    unroll=unroll,
                    n_runs=args.runs,
                    seed=args.seed,
                    warmup_runs=args.warmup,
                    timed_runs=args.timed,
                )
                all_results.extend(results)

                # Compute stats
                stats_summary = compute_statistics(results)
                all_stats.append(stats_summary)

                # Clear XLA trace caches to prevent unfair compilation hits
                # from previous configurations skewing cold JIT timing
                jax.clear_caches()

                print(f" Mean: {stats_summary.warm_mean:.2f}ms ± {stats_summary.warm_std:.2f}ms")

                # Perfetto trace (optional)
                if args.with_trace:
                    trace_dir = run_perfetto_trace(
                        engine_name, args.task, args.dim, args.pop_size,
                        args.gens, unroll, args.seed, output_dir / "traces"
                    )
                    print(f"    Trace: {trace_dir}")

            except Exception as e:
                print(f" ERROR: {e}")

    # Save results
    save_raw_results(all_results, output_dir / "raw_results.csv")
    save_statistics(all_stats, output_dir / "statistics.csv")

    # Print summary
    print_statistics_report(all_stats)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - raw_results.csv: {len(all_results)} individual measurements")
    print(f"  - statistics.csv: {len(all_stats)} configuration summaries")
    if args.with_trace:
        print("  - traces/: Perfetto traces (view at https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
