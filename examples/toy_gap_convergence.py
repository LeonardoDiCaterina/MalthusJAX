#!/usr/bin/env python3
"""Tiny BBOB benchmark that shows gap-to-optimum shrinking over time.

This merged example can run either backend or both side-by-side:
- evosax SimpleGA via ``--backend evosax``
- direct MalthusJAX engine via ``--backend malthusjax``
- both pipelines via ``--backend both`` (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
import optax

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy gap-to-optimum BBOB demo")
    parser.add_argument("--backend", choices=("evosax", "malthusjax", "both"), default="both")
    parser.add_argument("--function", type=str, default="rosenbrock")
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elite-k", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=0.1)
    parser.add_argument("--mutation-strength", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true", help="Save a gap plot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/toy_gap_convergence_parity.png"),
    )
    return parser.parse_args()


def _best_so_far(gap_history: list[float]) -> list[float]:
    return np.minimum.accumulate(gap_history).tolist()


def _print_report(label: str, gap_history: list[float], best_so_far: list[float]) -> None:
    print(f"\n== {label} ==")
    print("generation, gap_to_optimum, best_so_far")
    for generation, gap, best in zip(range(0, len(gap_history)), gap_history, best_so_far):
        print(f"{generation:>3d}, {gap:>12.6f}, {best:>12.6f}")

    print()
    print(f"initial gap: {gap_history[0]:.6f}")
    print(f"final gap:   {gap_history[-1]:.6f}")
    print(f"best-so-far:  {best_so_far[-1]:.6f}")


def _shared_initial_population(args: argparse.Namespace) -> jax.Array:
    """Sample one initial population and reuse it across both backends."""
    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )
    key = jr.PRNGKey(args.seed)
    initial_keys = jr.split(key, args.pop_size)
    return jax.vmap(evaluator.evosax_problem.sample)(initial_keys)


def run_evosax(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )

    # Ensure evosax uses the same effective elite pool size as MalthusJAX
    elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=args.generations,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params={
            "crossover_rate": args.crossover_rate,
            # Provide elite pool size via ratio only; some evosax strategy
            # implementations accept `elite_ratio` but not `elite_count`.
            "elite_ratio": float(elite_k) / float(args.pop_size),
            "std_schedule": optax.constant_schedule(args.mutation_strength),
        },
        initial_population=_shared_initial_population(args),
    )

    population_init = _shared_initial_population(args)
    key_eval = jr.PRNGKey(args.seed + 1)
    fitness_init, _, _ = evaluator.evosax_problem.eval(
        key_eval, population_init, evaluator.problem_state
    )
    initial_best_idx = int(np.argmin(np.asarray(fitness_init)))  # Minimize: pick lowest
    initial_gap = abs(float(fitness_init[initial_best_idx]) - float(evaluator.f_opt))

    result = adapter.run_once(jr.PRNGKey(args.seed), compile=False)
    gap_history = [initial_gap] + [
        abs(float(row["best_fitness"]) - float(evaluator.f_opt)) for row in result["history"]
    ]
    return gap_history, _best_so_far(gap_history)


def run_malthusjax(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)
    genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
    engine_params = GeneticEngineParams(
        pop_size=args.pop_size,
        elitism=0,
        num_generations=args.generations,
    )
    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1, crossover_rate=args.crossover_rate
        ),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=args.mutation_strength),
        enable_progress_bar=False,
    )

    shared_population = _shared_initial_population(args)
    initial_population = RealPopulation.from_array(shared_population, genome_config, axis=0)
    evaluated_population = evaluator.evaluate_population(initial_population)
    initial_best_idx = int(np.argmin(np.asarray(evaluated_population.fitness)))

    state = engine.init_state(jr.PRNGKey(args.seed)).replace(
        population=evaluated_population,
        best_genome=evaluated_population.genes[initial_best_idx],
        best_fitness=evaluated_population.fitness[initial_best_idx],
    )
    best_history = [float(state.best_fitness)]  # Now these are raw minimization values
    for _ in range(args.generations):
        state, output = engine.step(state)
        best_history.append(float(output.best_fitness))

    # Compute gaps using the convention flip:
    # fitness values are now positive minimization objectives
    gap_history = [abs(value - float(evaluator.f_opt)) for value in best_history]
    return gap_history, _best_so_far(gap_history)


def main() -> None:
    args = parse_args()
    runs: list[tuple[str, list[float], list[float]]] = []
    if args.backend in ("evosax", "both"):
        runs.append(("evosax", *run_evosax(args)))
    if args.backend in ("malthusjax", "both"):
        runs.append(("malthusjax", *run_malthusjax(args)))

    for label, gap_history, best_so_far in runs:
        _print_report(label, gap_history, best_so_far)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("matplotlib is required for --plot") from exc

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, gap_history, best_so_far in runs:
            ax.plot(range(1, len(gap_history) + 1), gap_history, marker="o", label=f"{label} gap")
            ax.plot(
                range(1, len(best_so_far) + 1),
                best_so_far,
                marker="s",
                linestyle="--",
                label=f"{label} best so far",
            )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Gap to optimum")
        ax.set_title(f"Toy parity run: gap to optimum over time ({args.function})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
