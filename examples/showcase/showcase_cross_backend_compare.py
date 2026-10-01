#!/usr/bin/env python3
"""Cross-Backend & Operator Composition Showcase for MalthusJAX.

Demonstrates the ultimate promise of the Composer:
1. GA Operator Ablation: Isolate crossover and mutation operators on Rastrigin.
2. Cross-Framework Battle: MalthusJAX GA vs EvoSAX CMA-ES vs Sep_CMA vs OpenES vs Native MAP-Elites.
3. Quality Diversity Cross-Validation: Native MAP-Elites vs QDAX MAP-Elites.
4. BBOB Hard Landscapes: Comparing search strategies across ill-conditioned, multimodal, and deceptive functions.
5. Level 3 Engine Parity: Validating declarative Composer reproduction of raw Engine loops.

Usage:
    python examples/showcase/showcase_cross_backend_compare.py --demo 1
    python examples/showcase/showcase_cross_backend_compare.py --demo 2
    python examples/showcase/showcase_cross_backend_compare.py --demo all
    python examples/showcase/showcase_cross_backend_compare.py --demo 2 --plot
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
import time
from pathlib import Path
from typing import Any, Dict

from malthusjax.composer.composer import Composer

# ==============================================================================
# Helper for plotting convergence curves
# ==============================================================================


def plot_comparison_results(result: Any, title: str, filename: str) -> None:
    try:
        import matplotlib.pyplot as plt

        os.makedirs("results/showcase", exist_ok=True)
        plt.figure(figsize=(10, 5))

        # If shared initial population was used, compute Gen 0 best fitness so curves share origin
        gen_0_best = None
        if getattr(result, "initial_population", None) is not None:
            try:
                import jax.numpy as jnp

                from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base
                from malthusjax.core.genome.real_genome import RealGenome, RealPopulation

                shared_cfg = getattr(result, "shared_config", {}) or {}
                fn_spec = shared_cfg.get("fitness")
                if fn_spec:
                    eval_seed = shared_cfg.get("seed", 42)
                    if isinstance(shared_cfg.get("seeds"), (list, tuple)) and shared_cfg["seeds"]:
                        eval_seed = shared_cfg["seeds"][0]
                    eval_obj = resolve_evaluator_base(
                        fn_spec,
                        maximize=shared_cfg.get("maximize", False),
                        seed=eval_seed,
                        num_dims=result.initial_population.shape[1],
                    )
                    pop = RealPopulation(
                        genes=RealGenome(values=result.initial_population),
                        fitness=jnp.zeros(len(result.initial_population)),
                        config=None,
                    )
                    eval_pop = eval_obj.evaluate_population(pop)
                    if shared_cfg.get("maximize", False):
                        gen_0_best = float(jnp.max(eval_pop.fitness))
                    else:
                        gen_0_best = float(jnp.min(eval_pop.fitness))
            except Exception:
                pass

        for name, run_res in result.pipelines.items():
            if not run_res.runs:
                continue
            history = run_res.runs[0].history
            gens = [h.get("generation", i) for i, h in enumerate(history)]
            best = [float(h.get("best_fitness", 0.0)) for h in history]
            if gen_0_best is not None and (not gens or gens[0] > 0):
                gens = [0] + gens
                best = [gen_0_best] + best
            plt.plot(gens, best, label=name, linewidth=2)

        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = Path("results/showcase") / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Plot saved to {out_path}")
    except Exception as e:
        print(f"    (Plotting skipped: {e})")


# ==============================================================================
# Showcase 1: GA Operator Ablation
# ==============================================================================


def run_showcase_1(composer: Composer, plot: bool = False):
    print("=" * 75)
    print("  SHOWCASE 1: GA Operator Ablation on 10D Rastrigin")
    print("=" * 75)
    print("  Hypothesis: SBX + Polynomial handles multimodal Rastrigin better than Blend.")

    pipelines = {
        "Blend + Gaussian": {
            "backend": "malthusjax",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.1",
            "selection": "tournament:tournament_size=3",
        },
        "SBX + Polynomial": {
            "backend": "malthusjax",
            "crossover": "simulated_binary:eta=2.0",
            "mutation": "polynomial:mutation_rate=0.1,eta=20.0",
            "selection": "tournament:tournament_size=3",
        },
        "Uniform + Gaussian": {
            "backend": "malthusjax",
            "crossover": "uniform_real:crossover_rate=0.5",
            "mutation": "gaussian:mutation_rate=0.1",
            "selection": "tournament:tournament_size=3",
        },
        "No Crossover (Mutation Only)": {
            "backend": "malthusjax",
            "crossover": "uniform_real:crossover_rate=0.0",
            "mutation": "gaussian:mutation_rate=0.15",
            "selection": "tournament:tournament_size=3",
        },
    }

    t0 = time.time()
    result = composer.compare(
        fitness="rastrigin:dim=10",
        pop_size=64,
        generations=50,
        seeds=(42,),
        shared_initial_population=True,  # Rigorous head-to-head fairness
        maximize=False,
        pipelines=pipelines,
    )
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.2f}s across {len(pipelines)} pipelines.\n")
    summary = result.summary_table()
    pprint.pprint(summary)

    if plot:
        plot_comparison_results(
            result, "GA Operator Ablation on 10D Rastrigin", "01_ablation_rastrigin.png"
        )


# ==============================================================================
# Showcase 2: Cross-Framework Comparison on Sphere
# ==============================================================================


def run_showcase_2(composer: Composer, plot: bool = False):
    print("\n" + "=" * 75)
    print("  SHOWCASE 2: Cross-Framework Battle on 10D Sphere")
    print("=" * 75)
    print(
        "  Competing Backends: MalthusJAX GA vs EvoSAX CMA-ES vs Sep_CMA_ES vs OpenES vs Native MAP-Elites"
    )

    pipelines: Dict[str, Dict[str, Any]] = {
        "MalthusJAX GA": {
            "backend": "malthusjax",
            "selection": "tournament:tournament_size=3",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.05",
        },
        "EvoSAX CMA-ES": {
            "backend": "evosax",
            "evosax_strategy": "CMA_ES",
        },
        "EvoSAX Sep_CMA_ES": {
            "backend": "evosax",
            "evosax_strategy": "Sep_CMA_ES",
        },
        "EvoSAX Open_ES": {
            "backend": "evosax",
            "evosax_strategy": "Open_ES",
        },
        "Native MAP-Elites": {
            "backend": "map_elites",
        },
    }

    t0 = time.time()
    result = composer.compare(
        fitness="sphere:dim=10",
        pop_size=64,
        generations=50,
        seeds=(42,),
        shared_initial_population=True,
        maximize=False,
        pipelines=pipelines,
    )
    elapsed = time.time() - t0

    print(f"\n  Cross-framework comparison finished in {elapsed:.2f}s.\n")
    summary = result.summary_table()
    pprint.pprint(summary)

    if plot:
        plot_comparison_results(
            result, "Cross-Framework Battle on 10D Sphere", "02_cross_framework_sphere.png"
        )


# ==============================================================================
# Showcase 3: Quality-Diversity Cross-Validation
# ==============================================================================


def run_showcase_3(composer: Composer, plot: bool = False, num_seeds: int = 1):
    print("\n" + "=" * 75)
    print("  SHOWCASE 3: Quality Diversity (Native MAP-Elites vs QDAX MAP-Elites)")
    print("=" * 75)

    pipelines = {
        "MalthusJAX Native MAP-Elites": {
            "backend": "map_elites",
            "qdax_num_descriptors": 2,
            "qdax_num_centroids": 64,
        },
        "QDAX MAP-Elites": {
            "backend": "qdax",
            "qdax_strategy": "MAPElites",
            "qdax_num_descriptors": 2,
            "qdax_num_centroids": 64,
            "qdax_mutation_sigma": 0.1,
        },
    }

    seeds = tuple(range(num_seeds)) if num_seeds > 1 else (42,)

    try:
        t0 = time.time()
        result = composer.compare(
            fitness="bbob:fn_name=rastrigin,num_dims=5",
            pop_size=64,
            generations=30,
            seeds=seeds,
            shared_initial_population=True,
            maximize=False,
            pipelines=pipelines,
        )
        elapsed = time.time() - t0
        print(
            f"\n  QD comparison ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}) completed in {elapsed:.2f}s.\n"
        )
        summary = result.summary_table()
        pprint.pprint(summary)

        if len(seeds) > 1:
            try:
                from scipy import stats

                native_runs = result.normalized_runs("MalthusJAX Native MAP-Elites")
                qdax_runs = result.normalized_runs("QDAX MAP-Elites")
                native_fit = [r.metrics["best_fitness"] for r in native_runs]
                qdax_fit = [r.metrics["best_fitness"] for r in qdax_runs]
                t_stat, p_val_fit = stats.ttest_rel(native_fit, qdax_fit)
                w_stat, p_val_wilcox = stats.wilcoxon(native_fit, qdax_fit)
                print(f"\n  --- Statistical Parity Tests (N={len(seeds)} seeds) ---")
                print(
                    f"    Best Fitness Paired t-test:       t={t_stat:.3f},  p-value={p_val_fit:.4f}"
                )
                print(
                    f"    Best Fitness Wilcoxon signed-rank: W={w_stat:.1f},  p-value={p_val_wilcox:.4f}"
                )
                if p_val_fit > 0.05:
                    print(
                        "    ✓ Null hypothesis holds: Performance is statistically indistinguishable (p > 0.05)."
                    )

                speedup = result.statistical_speedup(
                    "QDAX MAP-Elites", "MalthusJAX Native MAP-Elites"
                )
                print(
                    f"    Runtime Speedup (Native vs QDAX): {speedup['mean_speedup']:.2f}x [95% CI: {speedup['ci_lower']:.2f} - {speedup['ci_upper']:.2f}]"
                )
            except Exception as se:
                print(f"    (Statistical tests skipped: {se})")

        if plot:
            plot_comparison_results(
                result, "Native MAP-Elites vs QDAX MAP-Elites", "03_qd_cross_validation.png"
            )
    except Exception as e:
        print(f"  (QD cross-validation skipped or unavailable: {e})")


# ==============================================================================
# Showcase 5: BBOB Hard Landscapes
# ==============================================================================


def run_showcase_5(composer: Composer, plot: bool = False):
    print("\n" + "=" * 75)
    print("  SHOWCASE 5: BBOB Landscapes Comparison")
    print("=" * 75)

    # Functions: 8 = Rosenbrock (ill-conditioned valley), 15 = Rastrigin (multimodal)
    bbob_tests = [
        ("bbob:fn_name=rosenbrock,num_dims=5", "Rosenbrock (5D)"),
        ("bbob:fn_name=rastrigin,num_dims=5", "Rastrigin (5D)"),
    ]

    for fn_spec, label in bbob_tests:
        print(f"\n  --> Evaluating on BBOB {label}:")
        pipelines = {
            "MalthusJAX GA": {
                "backend": "malthusjax",
                "selection": "tournament:tournament_size=3",
                "crossover": "blend:alpha=0.5",
                "mutation": "gaussian:mutation_rate=0.1",
            },
            "EvoSAX CMA-ES": {
                "backend": "evosax",
                "evosax_strategy": "CMA_ES",
            },
            "EvoSAX DE": {
                "backend": "evosax",
                "evosax_strategy": "DifferentialEvolution",
            },
        }

        result = composer.compare(
            fitness=fn_spec,
            pop_size=64,
            generations=40,
            seeds=(42,),
            shared_initial_population=True,
            maximize=False,
            pipelines=pipelines,
        )
        summary = result.summary_table()
        for p_name, metrics in summary.items():
            best = metrics.get("best_fitness", "N/A")
            print(f"      {p_name:<20}: Best Fitness = {best}")


# ==============================================================================
# Showcase 7: Level 3 Engine Parity
# ==============================================================================


def run_showcase_7():
    print("\n" + "=" * 75)
    print("  SHOWCASE 7: Level 3 Engine Parity Reproduction")
    print("=" * 75)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from examples.showcase_engine_composer_parity import (
        showcase_custom_diversity_engine_parity,
        showcase_standard_engine_parity,
    )

    showcase_standard_engine_parity()
    showcase_custom_diversity_engine_parity()


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Cross-Backend & Strategy Showcase")
    parser.add_argument(
        "--demo",
        choices=[
            "1",
            "2",
            "3",
            "5",
            "7",
            "all",
            "ablation",
            "cross_backend",
            "qd",
            "bbob",
            "parity",
        ],
        default="2",
        help="Which showcase demo to run (default: 2)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to save convergence plots in results/showcase/",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds for statistical evaluation (default: 1)",
    )
    args = parser.parse_args()

    composer = Composer.create_default()

    demo = args.demo
    if demo in ("1", "ablation", "all"):
        run_showcase_1(composer, plot=args.plot)
    if demo in ("2", "cross_backend", "all"):
        run_showcase_2(composer, plot=args.plot)
    if demo in ("3", "qd", "all"):
        run_showcase_3(composer, plot=args.plot, num_seeds=args.seeds)
    if demo in ("5", "bbob", "all"):
        run_showcase_5(composer, plot=args.plot)
    if demo in ("7", "parity", "all"):
        run_showcase_7()

    print("\n" + "=" * 75)
    print("  SHOWCASE EXECUTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
