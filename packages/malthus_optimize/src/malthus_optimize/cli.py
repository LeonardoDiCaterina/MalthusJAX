"""Command-line interface for malthus-optimize (mopt)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, List, Optional

import numpy as np

from malthus_optimize.api.differential_evolution import differential_evolution
from malthus_optimize.api.minimize import minimize


def _benchmark_functions() -> dict[str, Any]:
    """Standard global optimization benchmark functions."""
    return {
        "sphere": lambda x: float(np.sum(x**2)),
        "rosenbrock": lambda x: float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1.0 - x[:-1]) ** 2.0)
        ),
        "rastrigin": lambda x: float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))),
        "ackley": lambda x: float(
            -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(2.0 * np.pi * x)))
            + 20.0
            + np.e
        ),
    }


def handle_benchmarks(args: argparse.Namespace) -> int:
    """Compares malthus_optimize vs scipy.optimize on benchmark functions."""
    benchmarks = _benchmark_functions()
    target_fns = [args.function] if args.function != "all" else list(benchmarks.keys())

    print("=" * 75)
    print(
        f"  Malthus-Optimize Benchmark Suite (dim={args.dim}, popsize={args.popsize}, maxiter={args.maxiter})"
    )
    print("=" * 75)

    bounds = [(-5.0, 5.0)] * args.dim

    for fn_name in target_fns:
        fn = benchmarks[fn_name]
        print(f"\n--- Benchmark Function: {fn_name.upper()} ---")

        # 1. Run Malthus-Optimize Differential Evolution
        t0 = time.perf_counter()
        m_res = differential_evolution(
            fn,
            bounds=bounds,
            popsize=args.popsize,
            maxiter=args.maxiter,
            seed=args.seed,
        )
        t_malthus = time.perf_counter() - t0

        print("Malthus-Optimize (JAX/Composer):")
        print(f"  Best Value: {m_res.fun:.6e}")
        print(f"  Execution Time: {t_malthus:.4f} s ({m_res.nfev} evaluations)")

        # 2. Run SciPy Differential Evolution
        try:
            import scipy.optimize as sp_opt

            t0 = time.perf_counter()
            s_res = sp_opt.differential_evolution(
                fn,
                bounds=bounds,
                popsize=args.popsize,
                maxiter=args.maxiter,
                seed=args.seed,
                polish=False,
            )
            t_scipy = time.perf_counter() - t0

            speedup = t_scipy / max(t_malthus, 1e-6)
            print("SciPy (Sequential CPU):")
            print(f"  Best Value: {s_res.fun:.6e}")
            print(f"  Execution Time: {t_scipy:.4f} s ({s_res.nfev} evaluations)")
            print(f"  Speedup: {speedup:.2f}x")
        except ImportError:
            pass

    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Executes a single optimization run on a benchmark problem."""
    benchmarks = _benchmark_functions()
    if args.function not in benchmarks:
        print(f"Unknown benchmark: '{args.function}'. Available: {list(benchmarks.keys())}")
        return 1

    fn = benchmarks[args.function]
    bounds = [(-5.0, 5.0)] * args.dim

    print(
        f"Running {args.method.upper()} on {args.function} (dim={args.dim}, maxiter={args.maxiter})..."
    )
    res = minimize(
        fun=fn,
        bounds=bounds,
        method=args.method,
        options={"maxiter": args.maxiter, "popsize": args.popsize, "seed": args.seed, "disp": True},
    )

    print("\nResult:")
    print(f"  Optimal parameters x: {np.round(res.x, 5)}")
    print(f"  Minimum value:        {res.fun:.6e}")
    print(f"  Total evaluations:    {res.nfev}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mopt",
        description="Malthus-Optimize CLI: SciPy-compatible optimization powered by JAX.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Benchmarks command
    bench_p = subparsers.add_parser(
        "benchmarks", help="Run comparative benchmarks vs scipy.optimize"
    )
    bench_p.add_argument(
        "--function",
        "-f",
        default="all",
        choices=["all", "sphere", "rosenbrock", "rastrigin", "ackley"],
    )
    bench_p.add_argument("--dim", "-d", type=int, default=10, help="Search dimension count")
    bench_p.add_argument("--popsize", "-p", type=int, default=15, help="Population size multiplier")
    bench_p.add_argument("--maxiter", "-m", type=int, default=80, help="Maximum generations")
    bench_p.add_argument("--seed", "-s", type=int, default=42, help="PRNG seed")
    bench_p.set_defaults(handler=handle_benchmarks)

    # Run command
    run_p = subparsers.add_parser("run", help="Run optimization on a benchmark function")
    run_p.add_argument(
        "--function",
        "-f",
        default="sphere",
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
    )
    run_p.add_argument("--method", default="de", choices=["de", "cmaes", "pso", "ga"])
    run_p.add_argument("--dim", "-d", type=int, default=5, help="Search dimension count")
    run_p.add_argument("--popsize", "-p", type=int, default=20, help="Population size multiplier")
    run_p.add_argument("--maxiter", "-m", type=int, default=60, help="Maximum generations")
    run_p.add_argument("--seed", "-s", type=int, default=42, help="PRNG seed")
    run_p.set_defaults(handler=handle_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
