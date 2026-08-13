#!/usr/bin/env python3
"""
HLO Extraction and Comparison Tool for MalthusJAX.

Extracts optimized XLA HLO for each engine pipeline defined in a TOML config,
then writes a side-by-side Markdown comparison table.

For EvoSAX pipelines, the HLO is extracted by JIT-compiling the native
strategy.ask() + strategy.tell() step directly (bypassing the MJX adapter)
so you see the true EvoSAX kernel — not the adapter overhead.

Usage:
    python scripts/extract_hlo.py --toml configs/perf/h1_speed_vs_evosax.toml
    python scripts/extract_hlo.py --toml configs/perf/h1_speed_vs_evosax.toml \
        --dims 9 --pop 195 --gens 387 --out-dir results/perf/h1_speed_vs_evosax/hlo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hlo_stats(hlo_text: str) -> dict[str, int]:
    """Extract key XLA graph stats from HLO text."""
    return {
        "lines": len(hlo_text.splitlines()),
        "fusion_kernels": hlo_text.count("fusion"),
        "while_loops": hlo_text.count("while"),
        "broadcasts": hlo_text.count("broadcast"),
        "copies": hlo_text.count("copy"),
    }


def _lower_and_compile(fn: Any, *args: Any, optimize: bool = True) -> str:
    """JIT-lower a function, compile, and return HLO text."""
    jit_fn = jax.jit(fn)
    lowered = jit_fn.lower(*args)
    if optimize:
        compiled = lowered.compile()
        return compiled.as_text()  # type: ignore[return-value]
    return lowered.as_text()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# EvoSAX native step extraction
# ---------------------------------------------------------------------------


def extract_evosax_hlo(
    strategy_name: str,
    pop_size: int,
    num_dims: int,
    strategy_params: dict[str, Any] | None = None,
    optimize: bool = True,
) -> str:
    """
    Extract HLO for a native EvoSAX strategy step (ask + tell).

    Bypasses the MJX adapter entirely so the HLO reflects the true EvoSAX
    kernel — the gold-standard target for MJX performance parity.
    """
    try:
        from evosax.algorithms.population_based import simple_ga as sga_mod  # noqa: F401
        from evosax.algorithms.population_based.simple_ga import SimpleGA
    except ImportError as e:
        raise ImportError("evosax is not installed. Run: pip install evosax") from e

    init_solution = jnp.zeros(num_dims)

    # Resolve strategy class
    strategy_map = {"SimpleGA": SimpleGA}
    if strategy_name not in strategy_map:
        raise ValueError(
            f"Unknown EvoSAX strategy '{strategy_name}'. Supported: {list(strategy_map)}"
        )
    strategy_cls = strategy_map[strategy_name]
    strategy = strategy_cls(population_size=pop_size, solution=init_solution)
    params = strategy.default_params

    # Apply any overrides from TOML strategy_params
    if strategy_params:
        params_kwargs = {}
        for k, v in strategy_params.items():
            if hasattr(params, k):
                params_kwargs[k] = v
        if params_kwargs:
            params = params.replace(**params_kwargs)

    key = jax.random.PRNGKey(0)
    pop_init = jnp.zeros((pop_size, num_dims))
    fit_init = jnp.zeros(pop_size)
    state = strategy.init(key, pop_init, fit_init, params)

    # Sphere fitness for a realistic step trace
    def evosax_step(key: jax.Array, state: Any) -> Any:
        k_ask, k_tell = jax.random.split(key)
        population, new_state = strategy.ask(k_ask, state, params)
        fitness = jax.vmap(lambda x: jnp.sum(jnp.square(x)))(population)
        new_state, _ = strategy.tell(k_tell, population, fitness, new_state, params)
        return new_state

    return _lower_and_compile(evosax_step, key, state, optimize=optimize)


# ---------------------------------------------------------------------------
# MJX engine step extraction
# ---------------------------------------------------------------------------


def extract_mjx_hlo(
    pipeline_kwargs: dict[str, Any],
    pop_size: int,
    num_dims: int,
    gens: int,
    optimize: bool = True,
) -> str:
    """Extract HLO for a MalthusJAX pipeline using its built-in get_hlo_text()."""
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator

    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name="sphere", num_dims=num_dims, seed=42, maximize=False)
    )

    # Strip keys not understood by build_engine
    kwargs = {
        k: v
        for k, v in pipeline_kwargs.items()
        if k
        not in {
            "backend",
            "evosax_strategy",
            "strategy_params",
        }
    }

    catalog = OperatorCatalog()
    sel_str = kwargs.pop("selection", None)
    cross_str = kwargs.pop("crossover", None)
    mut_str = kwargs.pop("mutation", None)

    sel = catalog.get(sel_str) if sel_str else None
    cross = catalog.get(cross_str) if cross_str else None
    mut = catalog.get(mut_str) if mut_str else None

    engine_type = kwargs.pop("engine_type", "ga")

    # engine_factory.build_engine takes engine_cls, but we just use engine_type through Composer's EngineRegistry
    from malthusjax.composer.engine_catalog import EngineRegistry

    registry = EngineRegistry()

    adapter = registry.get(
        engine_type,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        genome_type="real",
        pop_size=pop_size,
        generations=gens,
        genome_shape=(num_dims,),
        bounds=(-5.0, 5.0),
        **kwargs,
    )
    engine = adapter.genetic_engine
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)
    hlo = engine.get_hlo_text(state, optimize=optimize, print_analysis=False)
    return hlo  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _write_summary(results: dict[str, dict[str, Any]], out_dir: Path) -> None:
    """Write a Markdown table comparing HLO stats for all pipelines."""
    rows = []
    for name, info in results.items():
        if "error" in info:
            rows.append((name, "ERROR", info["error"], "—", "—", "—", "—"))
        else:
            s = info["stats"]
            rows.append(
                (
                    name,
                    "ok",
                    "—",
                    s["lines"],
                    s["fusion_kernels"],
                    s["while_loops"],
                    s["copies"],
                )
            )

    summary_path = out_dir / "hlo_summary.md"
    with open(summary_path, "w") as f:
        f.write("# HLO Comparison Summary\n\n")
        f.write("| Pipeline | Status | IR Lines | Fusion Kernels | While Loops | Copies |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for name, status, error, lines, fusions, whiles, copies in rows:
            if status == "ERROR":
                f.write(f"| `{name}` | ❌ ERROR | {error[:60]} | — | — | — |\n")
            else:
                f.write(f"| `{name}` | ✅ | {lines:,} | {fusions:,} | {whiles:,} | {copies:,} |\n")
        f.write("\n")
        f.write(
            "_Fewer IR lines and while-loops generally means a tighter, more fused XLA kernel._\n"
        )

    print(f"\n  Summary written to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and compare HLO for all pipelines in a perf TOML."
    )
    parser.add_argument(
        "--toml",
        required=True,
        help="Path to the perf TOML config (e.g. configs/perf/h1_speed_vs_evosax.toml)",
    )
    parser.add_argument("--dims", type=int, default=9, help="Problem dimensionality (default: 9)")
    parser.add_argument("--pop", type=int, default=195, help="Population size (default: 195)")
    parser.add_argument(
        "--gens", type=int, default=387, help="Number of generations (default: 387)"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to <suite.output_dir>/hlo/)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip XLA optimization pass (faster, but less representative)",
    )
    args = parser.parse_args()

    optimize = not args.no_optimize

    # Load TOML
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            print("ERROR: Install tomli: pip install tomli", file=sys.stderr)
            sys.exit(1)

    toml_path = Path(args.toml)
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else Path(config["suite"]["output_dir"]) / "hlo"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipelines = config.get("pipelines", {})
    print(
        f"\nHLO Extraction  dims={args.dims}  pop={args.pop}  gens={args.gens}  optimize={optimize}"
    )
    print(f"Output: {out_dir}")
    print("=" * 70)

    results: dict[str, dict[str, Any]] = {}

    for name, pipe_def in pipelines.items():
        print(f"\n  [{name}] ", end="", flush=True)
        backend = pipe_def.get("backend", "malthusjax")

        try:
            if backend == "evosax":
                strategy_name = pipe_def.get("evosax_strategy", "SimpleGA")
                strategy_params = pipe_def.get("strategy_params", {})
                print(f"(evosax/{strategy_name}) ... ", end="", flush=True)
                hlo = extract_evosax_hlo(
                    strategy_name=strategy_name,
                    pop_size=args.pop,
                    num_dims=args.dims,
                    strategy_params=strategy_params,
                    optimize=optimize,
                )
            else:
                # MalthusJAX pipeline
                fmt_vars = dict(
                    pop_size=args.pop,
                    genome_length=args.dims,
                    generations=args.gens,
                    elite_k=max(2, args.pop // 6),
                )
                formatted_pipe = {
                    k: (v.format(**fmt_vars) if isinstance(v, str) else v)
                    for k, v in pipe_def.items()
                }
                print("(malthusjax) ... ", end="", flush=True)
                hlo = extract_mjx_hlo(
                    pipeline_kwargs=formatted_pipe,
                    pop_size=args.pop,
                    num_dims=args.dims,
                    gens=args.gens,
                    optimize=optimize,
                )

            # Save HLO text
            hlo_file = out_dir / f"{name}.hlo.txt"
            hlo_file.write_text(hlo)
            stats = _hlo_stats(hlo)
            results[name] = {"stats": stats}
            print(
                f"done  ({stats['lines']:,} IR lines, "
                f"{stats['fusion_kernels']} fusions, "
                f"{stats['while_loops']} whiles)"
            )

        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"ERROR: {exc}")

    _write_summary(results, out_dir)

    # Print summary table to stdout too
    print("\n" + "=" * 70)
    print(f"{'Pipeline':<40} {'Lines':>8} {'Fusions':>9} {'Whiles':>8}")
    print("-" * 70)
    for name, info in results.items():
        if "error" in info:
            print(f"  {name:<38} {'ERROR':>27}")
        else:
            s = info["stats"]
            print(f"  {name:<38} {s['lines']:>8,} {s['fusion_kernels']:>9,} {s['while_loops']:>8,}")


if __name__ == "__main__":
    main()
