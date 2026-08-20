#!/usr/bin/env python3
"""Debug BBOB parity demo with per-step MalthusJAX population tracing.

This variant mirrors `examples/toy_gap_convergence.py` but uses the new
`debug_run()` path for the MalthusJAX backend so each generation prints the
current population length after `step()`.

Backends:
- evosax SimpleGA via `--backend evosax`
- direct MalthusJAX engine via `--backend malthusjax`
- both pipelines via `--backend both` (default)
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from malthusjax.composer.evosax_adapter import build_evosax_engine
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.resource_mapper import get_resource_summary, get_step_dimension_flow
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy gap-to-optimum BBOB debug demo")
    parser.add_argument("--backend", choices=("evosax", "malthusjax", "both"), default="both")
    parser.add_argument("--function", type=str, default="rosenbrock")
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elite-k", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=0.1)
    parser.add_argument("--mutation-strength", type=float, default=0.0)
    parser.add_argument(
        "--show-resource-summary",
        action="store_true",
        help="Print the resource map summary for the MalthusJAX pipeline",
    )
    parser.add_argument(
        "--show-dimension-flow",
        action="store_true",
        help="Print the exact phase dimension flow for one step",
    )
    parser.add_argument("--plot", action="store_true", help="Save a gap plot")
    parser.add_argument(
        "--compare-trials",
        type=int,
        default=100,
        help="Number of trials to run when comparing operators (default: 100)",
    )
    parser.add_argument(
        "--param-sweep",
        action="store_true",
        help="Run parameter sweep across elite_ratio/elite_k values",
    )
    parser.add_argument(
        "--capture-trials",
        type=int,
        default=100,
        help="Trials used to screen for largest crossover diffs when capturing traces",
    )
    parser.add_argument(
        "--capture-top-k",
        type=int,
        default=0,
        help="If >0, run screening and capture detailed traces for top-k differing trials",
    )
    parser.add_argument(
        "--replay-crossover-parity",
        action="store_true",
        help="Replay the captured Evosax crossover key split and compare it against MalthusJAX",
    )
    parser.add_argument(
        "--replay-selection-parity",
        action="store_true",
        help="Compare the live MalthusJAX selection step against the captured parent split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/toy_gap_convergence_debug.png"),
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


def _print_population_snapshot(label: str, population: RealPopulation) -> None:
    fitness = np.asarray(population.fitness)
    genes = np.asarray(population.genes)
    print(f"{label}: size={len(population)}, gene_shape={genes.shape[1:]}")
    print(
        f"{label}: fitness min/mean/max = "
        f"{fitness.min():.6f} / {fitness.mean():.6f} / {fitness.max():.6f}"
    )
    print(f"{label}: first_fitness={fitness[: min(3, len(fitness))].tolist()}")


def _format_value(value: object) -> str:
    if isinstance(value, (int, float, bool, str)):
        return repr(value)
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return repr(value.tolist())
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"array(shape={tuple(value.shape)}, dtype={value.dtype})"
    return repr(value)


def _print_operator_box(title: str, operator: object, result: object) -> None:
    fields = []
    if dataclasses.is_dataclass(operator):
        for field in dataclasses.fields(operator):
            fields.append((field.name, getattr(operator, field.name)))

    type_name = type(operator).__name__
    module_name = type(operator).__module__
    lines = [
        f"┌─ {title} ─{'─' * max(0, 44 - len(title))}┐",
        f"│ type: {module_name}.{type_name}",
        f"│ result: {_format_value(result)}",
        "│ parameters:",
    ]
    for name, value in fields:
        lines.append(f"│   {name}: {_format_value(value)}")
    lines.append(f"└{'─' * 58}┘")
    print("\n".join(lines))


def _mean_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between matching rows of two 2D arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch for distance computation: {a.shape} vs {b.shape}")
    if a.ndim == 1:
        return float(np.linalg.norm(a - b))
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


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


def _shared_strategy_params(args: argparse.Namespace, elite_k: int) -> dict[str, object]:
    return {
        "crossover_rate": args.crossover_rate,
        "elite_ratio": float(elite_k) / float(args.pop_size),
        "std_schedule": optax.constant_schedule(args.mutation_strength),
    }


def run_evosax(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )

    elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=args.generations,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params=_shared_strategy_params(args, elite_k),
        initial_population=_shared_initial_population(args),
    )

    pass

    population_init = _shared_initial_population(args)
    key_eval = jr.PRNGKey(args.seed + 1)
    fitness_init, _, _ = evaluator.evosax_problem.eval(
        key_eval, population_init, evaluator.problem_state
    )
    # Inspect evosax internals: init state and one ask() to see operator outputs
    strategy = adapter.strategy
    params = adapter.params
    rng = jr.PRNGKey(args.seed + 1)
    rng, key_init, key_ask = jr.split(rng, 3)
    state = strategy.init(key_init, population_init, fitness_init, params)
    offspring_pop, state_after = strategy.ask(key_ask, state, params)

    # Build a params dict for pretty-printing
    ev_params = {}
    try:
        ev_params.update(getattr(params, "__dict__", {}))
    except Exception:
        pass
    # Strategy-level attributes
    strat_attrs = {k: v for k, v in strategy.__dict__.items() if not k.startswith("_")}

    _print_operator_box(
        "evosax strategy (ask) output",
        strategy,
        {
            "offspring_shape": np.asarray(offspring_pop).shape,
            "first_offspring": np.asarray(offspring_pop)[
                : min(3, np.asarray(offspring_pop).shape[0])
            ].tolist(),
        },
    )

    _print_operator_box(
        "evosax strategy params",
        params,
        {
            "params": ev_params,
            "strategy_attrs": {k: _format_value(v) for k, v in strat_attrs.items()},
        },
    )
    initial_best_idx = int(np.argmin(np.asarray(fitness_init)))
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
        forward_presplit_keys=True,
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
            num_offspring=1,
            crossover_rate=args.crossover_rate,
        ),
        mutation=EvosaxGaussianWrapper(
            num_offspring=1,
            mutation_strength=args.mutation_strength,
        ),
        enable_progress_bar=False,
    )

    print(
        "\n== malthusjax debug config ==\n"
        f"function={args.function}, dims={args.dimensions}, pop_size={args.pop_size}, "
        f"generations={args.generations}, elite_k={elite_k}, "
        f"crossover_rate={args.crossover_rate}, mutation_strength={args.mutation_strength}"
    )

    shared_population = _shared_initial_population(args)
    initial_population = RealPopulation.from_array(shared_population, genome_config, axis=0)
    evaluated_population = evaluator.evaluate_population(initial_population)
    initial_best_idx = int(np.argmin(np.asarray(evaluated_population.fitness)))
    _print_population_snapshot("initial population", evaluated_population)

    init_state = engine.init_state(jr.PRNGKey(args.seed))

    if args.show_resource_summary:
        print(get_resource_summary(init_state.resource_map))

    if args.show_dimension_flow:
        print(
            get_step_dimension_flow(
                init_state.resource_map,
                elitism=engine_params.elitism,
                pop_symbol="n",
                genome_symbol="d",
                genome_width=args.dimensions,
            )
        )

    state = init_state.replace(
        population=evaluated_population,
        best_genome=evaluated_population.genes[initial_best_idx],
        best_fitness=evaluated_population.fitness[initial_best_idx],
    )
    best_history = [float(state.best_fitness)]
    # Inspect operators on the freshly-evaluated initial population
    k_sel, k_cross, k_mut, k_eval, k_next = engine._allocate_entropy(state)
    elites_genes, parent_indices = engine._selection_phase(
        k_sel, state.population, state.operators, engine.engine_params
    )

    # Prepare parents for crossover exactly as the engine does
    rmap = state.resource_map
    num_pairs = rmap.crossover.input_count // 2
    p1_idx = parent_indices[:num_pairs]
    p2_idx = parent_indices[num_pairs : num_pairs * 2]
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], state.population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], state.population.genes)
    dummy_fitness = jnp.zeros(num_pairs)
    p1_pop = state.population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = state.population.spawn_offspring(p2_genes, fitness=dummy_fitness)

    # Run crossover and mutation using the live baked operators
    crossover_offspring = state.operators.crossover(
        k_cross, p1_pop, p2_pop, engine.genome_config, generation=state.generation
    )
    mutated_offspring = state.operators.mutation(
        k_mut, crossover_offspring, engine.genome_config, generation=state.generation
    )

    # Pretty-print operator boxes
    _print_operator_box(
        "live crossover operator",
        state.operators.crossover,
        {
            "offspring_shape": np.asarray(
                jax.tree_util.tree_leaves(crossover_offspring.genes)[0]
            ).shape,
            "first_offspring": np.asarray(jax.tree_util.tree_leaves(crossover_offspring.genes)[0])[
                : min(
                    3, np.asarray(jax.tree_util.tree_leaves(crossover_offspring.genes)[0]).shape[0]
                )
            ].tolist(),
        },
    )

    _print_operator_box(
        "live mutation operator",
        state.operators.mutation,
        {
            "offspring_shape": np.asarray(
                jax.tree_util.tree_leaves(mutated_offspring.genes)[0]
            ).shape,
            "first_offspring": np.asarray(jax.tree_util.tree_leaves(mutated_offspring.genes)[0])[
                : min(3, np.asarray(jax.tree_util.tree_leaves(mutated_offspring.genes)[0]).shape[0])
            ].tolist(),
        },
    )
    final_state, history = engine.debug_run(state)
    _ = final_state
    best_history.extend(float(item.best_fitness) for item in history)

    _print_population_snapshot("final population", final_state.population)

    live_selection = final_state.operators.selection
    selection_parent_idx, selection_elite_idx = live_selection(state.rng_key, state.population)
    _print_operator_box(
        "live selection operator",
        live_selection,
        {
            "parent_idx": np.asarray(selection_parent_idx).tolist(),
            "elite_idx": np.asarray(selection_elite_idx).tolist(),
        },
    )

    gap_history = [abs(value - float(evaluator.f_opt)) for value in best_history]
    return gap_history, _best_so_far(gap_history)


def replay_crossover_parity(args: argparse.Namespace) -> None:
    """Reconstruct the Evosax key split and replay it through MalthusJAX.

    This is the narrow parity-check mode used to validate the wrapper contract:
    the same sampled parents and presplit per-pair keys are passed to both
    implementations, and the resulting offspring should match exactly.
    """
    trace_path = Path(f"results/crossover_trace_{args.seed}.npz")
    if trace_path.exists():
        data = np.load(trace_path)
        config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))

        def _fmt(value: object) -> str:
            arr = np.asarray(value)
            if arr.ndim == 0:
                return repr(arr.item())
            preview = arr.reshape(-1)[:6].tolist()
            return f"shape={arr.shape}, preview={preview}"

        def _compare(label: str, live: object, captured: object) -> None:
            live_arr = np.asarray(live)
            captured_arr = np.asarray(captured)
            if live_arr.shape == captured_arr.shape:
                diff = np.abs(live_arr - captured_arr)
                diff_text = (
                    f"max_diff={float(diff.max()):.6e}, "
                    f"mean_diff={float(diff.mean()):.6e}, "
                    f"allclose={bool(np.allclose(live_arr, captured_arr))}"
                )
            else:
                diff_text = f"shape_mismatch live={live_arr.shape} captured={captured_arr.shape}"
            print(f"{label}: live={_fmt(live_arr)} | captured={_fmt(captured_arr)} | {diff_text}")

        # Live step snapshot: reconstruct the same one-generation path the engine takes.
        evaluator = BBOBEvaluator.create(
            BBOBConfig(
                fn_name=args.function,
                num_dims=args.dimensions,
                seed=args.seed,
                maximize=False,
            )
        )
        genome_config = config
        elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)
        engine = GeneticEngine(
            engine_params=GeneticEngineParams(
                pop_size=args.pop_size,
                elitism=0,
                num_generations=1,
                forward_presplit_keys=True,
            ),
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=elite_k),
            crossover=EvosaxUniformCrossoverWrapper(
                num_offspring=1,
                crossover_rate=args.crossover_rate,
            ),
            mutation=EvosaxGaussianWrapper(
                num_offspring=1,
                mutation_strength=args.mutation_strength,
            ),
            enable_progress_bar=False,
        )

        live_state = engine.init_state(jr.PRNGKey(args.seed))
        live_k_sel, live_k_cross, live_k_mut, live_k_eval, live_k_next = engine._allocate_entropy(
            live_state
        )
        live_elites, live_parent_indices = engine._selection_phase(
            live_k_sel, live_state.population, live_state.operators, engine.engine_params
        )
        live_num_pairs = live_state.resource_map.crossover.input_count // 2
        live_p1_idx = live_parent_indices[:live_num_pairs]
        live_p2_idx = live_parent_indices[live_num_pairs : live_num_pairs * 2]
        live_p1_genes = jax.tree_util.tree_map(
            lambda x: x[live_p1_idx], live_state.population.genes
        )
        live_p2_genes = jax.tree_util.tree_map(
            lambda x: x[live_p2_idx], live_state.population.genes
        )
        live_dummy = jnp.zeros(live_num_pairs)
        live_p1_pop = live_state.population.spawn_offspring(live_p1_genes, fitness=live_dummy)
        live_p2_pop = live_state.population.spawn_offspring(live_p2_genes, fitness=live_dummy)
        live_mutants = engine._reproduction_phase(
            live_k_cross,
            live_k_mut,
            live_parent_indices,
            live_state.population,
            live_state.operators,
            live_state.resource_map,
            generation=live_state.generation,
        )
        live_next_genes = engine._merge(live_elites, live_mutants.genes, live_state)
        live_new_pop = engine._evaluate(live_next_genes, live_state)

        # Captured trace snapshot.
        captured_live_mj = np.asarray(data["mj_offspring"])
        captured_replay = np.asarray(data["ev_offspring"])
        captured_k_cross = np.asarray(data["k_cross_keys"])
        captured_p1_idx = np.asarray(data["p1_idx"])
        captured_p2_idx = np.asarray(data["p2_idx"])
        captured_p1 = np.asarray(data["p1_pop"])
        captured_p2 = np.asarray(data["p2_pop"])

        print("\n== live vs captured replay (side-by-side) ==")
        print(f"trace: {trace_path}")
        _compare("k_cross", live_k_cross, captured_k_cross)
        _compare("p1_idx", live_p1_idx, captured_p1_idx)
        _compare("p2_idx", live_p2_idx, captured_p2_idx)
        _compare("p1_first_leaf", jax.tree_util.tree_leaves(live_p1_pop.genes)[0], captured_p1)
        _compare("p2_first_leaf", jax.tree_util.tree_leaves(live_p2_pop.genes)[0], captured_p2)
        _compare(
            "live_offspring_vs_captured_mj",
            jax.tree_util.tree_leaves(live_mutants.genes)[0],
            captured_live_mj,
        )
        _compare(
            "live_offspring_vs_captured_replay",
            jax.tree_util.tree_leaves(live_mutants.genes)[0],
            captured_replay,
        )
        _compare(
            "live_next_genes_vs_trace_mj",
            jax.tree_util.tree_leaves(live_next_genes)[0],
            captured_live_mj,
        )
        _compare("live_fitness", live_new_pop.fitness, data["fitness_init"])
        return


def replay_selection_parity(args: argparse.Namespace) -> None:
    """Compare the live selection output against the captured parent split."""

    def _compare(label: str, live: object, captured: object) -> None:
        live_arr = np.asarray(live)
        captured_arr = np.asarray(captured)
        if live_arr.shape == captured_arr.shape:
            diff = np.abs(live_arr - captured_arr)
            diff_text = (
                f"max_diff={float(diff.max()):.6e}, "
                f"mean_diff={float(diff.mean()):.6e}, "
                f"allclose={bool(np.array_equal(live_arr, captured_arr))}"
            )
        else:
            diff_text = f"shape_mismatch live={live_arr.shape} captured={captured_arr.shape}"
        print(f"{label}: live={live_arr.shape}, captured={captured_arr.shape}, {diff_text}")

    trace_path = Path(f"results/crossover_trace_{args.seed}.npz")
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing captured trace: {trace_path}")

    data = np.load(trace_path)
    config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
    population_init = jnp.asarray(data["population_init"])
    fitness_init = jnp.asarray(data["fitness_init"])
    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )
    elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)
    engine = GeneticEngine(
        engine_params=GeneticEngineParams(
            pop_size=args.pop_size,
            elitism=0,
            num_generations=1,
            forward_presplit_keys=True,
        ),
        genome_config=config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1,
            crossover_rate=args.crossover_rate,
        ),
        mutation=EvosaxGaussianWrapper(
            num_offspring=1,
            mutation_strength=args.mutation_strength,
        ),
        enable_progress_bar=False,
    )

    live_state = engine.init_state(jr.PRNGKey(args.seed))
    evaluated_population = evaluator.evaluate_population(
        RealPopulation.from_array(np.asarray(population_init), config, axis=0)
    )
    live_state = live_state.replace(
        population=evaluated_population,
        best_genome=evaluated_population.genes[0],
        best_fitness=evaluated_population.fitness[0],
    )

    # Fetch the engine-allocated selection key slice so we can compare it
    # directly to the Evosax-derived parent draw keys used during capture.
    alloc_k_sel, _, _, _, _ = engine._allocate_entropy(live_state)

    # Prefer using saved parent-draw keys from the trace when available.
    if "k_parent_keys" in data.files:
        try:
            parent_keys = jnp.asarray(data["k_parent_keys"])  # shape (2,2)
            live_k_sel = parent_keys
        except Exception:
            # fallback to reconstructing keys from seed
            base_key, parent_key = jr.split(jr.PRNGKey(args.seed + 3), 2)
            k1, k2 = jr.split(parent_key, 2)
            live_k_sel = jnp.stack([k1, k2])
    else:
        # Reconstruct the Evosax-style parent draw keys used in capture (best-effort)
        base_key, parent_key = jr.split(jr.PRNGKey(args.seed + 3), 2)
        k1, k2 = jr.split(parent_key, 2)
        live_k_sel = jnp.stack([k1, k2])

    live_elites, live_parent_indices = engine._selection_phase(
        live_k_sel, live_state.population, live_state.operators, engine.engine_params
    )

    captured_p1_idx = np.asarray(data["p1_idx"])
    captured_p2_idx = np.asarray(data["p2_idx"])
    captured_p1 = np.asarray(data["p1_pop"])
    captured_p2 = np.asarray(data["p2_pop"])
    captured_parent_indices = np.concatenate([captured_p1_idx, captured_p2_idx])
    live_parent_indices_arr = np.asarray(live_parent_indices)

    print("\n== live vs captured selection (side-by-side) ==")
    print(f"trace: {trace_path}")
    print(
        f"live_parent_indices: shape={live_parent_indices_arr.shape}, "
        f"preview={live_parent_indices_arr[:8].tolist()}"
    )
    print(
        f"captured_parent_indices: shape={captured_parent_indices.shape}, "
        f"preview={captured_parent_indices[:8].tolist()}"
    )
    print(
        "selection compare: "
        f"max_diff={float(np.abs(live_parent_indices_arr - captured_parent_indices).max()):.6e}, "
        f"mean_diff={float(np.abs(live_parent_indices_arr - captured_parent_indices).mean()):.6e}, "
        f"allclose={bool(np.array_equal(live_parent_indices_arr, captured_parent_indices))}"
    )

    live_num_pairs = live_state.resource_map.crossover.input_count // 2
    live_p1_idx = live_parent_indices_arr[:live_num_pairs]
    live_p2_idx = live_parent_indices_arr[live_num_pairs : live_num_pairs * 2]
    print(
        f"live p1_idx preview={live_p1_idx[:8].tolist()} | "
        f"captured p1_idx preview={captured_p1_idx[:8].tolist()}"
    )
    print(
        f"live p2_idx preview={live_p2_idx[:8].tolist()} | "
        f"captured p2_idx preview={captured_p2_idx[:8].tolist()}"
    )

    live_p1_genes = jax.tree_util.tree_map(lambda x: x[live_p1_idx], live_state.population.genes)
    live_p2_genes = jax.tree_util.tree_map(lambda x: x[live_p2_idx], live_state.population.genes)
    live_dummy = jnp.zeros(live_num_pairs)
    live_p1_pop = live_state.population.spawn_offspring(live_p1_genes, fitness=live_dummy)
    live_p2_pop = live_state.population.spawn_offspring(live_p2_genes, fitness=live_dummy)

    live_p1_arr = np.asarray(jax.tree_util.tree_leaves(live_p1_pop.genes)[0])
    live_p2_arr = np.asarray(jax.tree_util.tree_leaves(live_p2_pop.genes)[0])
    _compare("p1_parent_genes", live_p1_arr, captured_p1)
    _compare("p2_parent_genes", live_p2_arr, captured_p2)
    live_elite_leaves = jax.tree_util.tree_leaves(live_elites)
    live_elite_preview = np.asarray(live_elite_leaves[0])[:3].tolist() if live_elite_leaves else []
    print(f"live elites preview={live_elite_preview}")
    print(
        f"fitness_init parity check: allclose={bool(np.array_equal(np.asarray(fitness_init), np.asarray(data['fitness_init'])))}"
    )
    return

    evaluator = BBOBEvaluator.create(
        BBOBConfig(
            fn_name=args.function,
            num_dims=args.dimensions,
            seed=args.seed,
            maximize=False,
        )
    )

    elite_k = args.elite_k if args.elite_k is not None else max(1, args.pop_size // 2)
    genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))

    strategy_params = _shared_strategy_params(args, elite_k)
    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=1,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params=strategy_params,
        initial_population=_shared_initial_population(args),
    )

    # Reconstruct the same initial population and key schedule used in the
    # detailed crossover capture path.
    key = jr.PRNGKey(args.seed)
    initial_keys = jr.split(key, args.pop_size)
    population_init = jax.vmap(evaluator.evosax_problem.sample)(initial_keys)
    key_eval = jr.PRNGKey(args.seed + 1)
    fitness_init, _, _ = evaluator.evosax_problem.eval(
        key_eval, population_init, evaluator.problem_state
    )

    strategy = adapter.strategy
    params = adapter.params
    state = strategy.init(jr.PRNGKey(args.seed + 2), population_init, fitness_init, params)
    sorted_idx = np.asarray(jnp.argsort(state.fitness))
    pop_size = args.pop_size
    strategy_elite_ratio = float(
        getattr(strategy, "elite_ratio", strategy_params.get("elite_ratio", 0.5))
    )
    k_elite = int(strategy_elite_ratio * pop_size)
    p_mask = jnp.arange(pop_size) < k_elite
    base_key, parent_key = jr.split(jr.PRNGKey(args.seed + 3), 2)
    k_cross = base_key
    cross_keys = jr.split(base_key, pop_size)
    k1, k2 = jr.split(parent_key, 2)

    # Evosax parents / crossover output.
    parents_1 = jax.random.choice(k1, population_init, (pop_size,), p=p_mask)
    parents_2 = jax.random.choice(k2, population_init, (pop_size,), p=p_mask)
    from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover_fn

    ev_offspring = jax.vmap(evosax_crossover_fn, in_axes=(0, 0, 0, None))(
        cross_keys, parents_1, parents_2, args.crossover_rate
    )

    # MalthusJAX parents / crossover output, using the same presplit keys.
    engine = GeneticEngine(
        engine_params=GeneticEngineParams(
            pop_size=args.pop_size,
            elitism=0,
            num_generations=1,
            forward_presplit_keys=True,
        ),
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1, crossover_rate=args.crossover_rate
        ),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=args.mutation_strength),
        enable_progress_bar=False,
    )

    evaluated_population = evaluator.evaluate_population(
        RealPopulation.from_array(population_init, genome_config, axis=0)
    )
    sel1 = np.asarray(jax.random.choice(k1, jnp.arange(pop_size), (pop_size,), p=p_mask))
    sel2 = np.asarray(jax.random.choice(k2, jnp.arange(pop_size), (pop_size,), p=p_mask))
    p1_idx = np.asarray(sorted_idx)[sel1]
    p2_idx = np.asarray(sorted_idx)[sel2]
    p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], evaluated_population.genes)
    p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], evaluated_population.genes)
    dummy_fitness = jnp.zeros(pop_size)
    p1_pop = evaluated_population.spawn_offspring(p1_genes, fitness=dummy_fitness)
    p2_pop = evaluated_population.spawn_offspring(p2_genes, fitness=dummy_fitness)
    mj_offspring = engine.crossover(
        jnp.expand_dims(k_cross, 0), p1_pop, p2_pop, engine.genome_config, generation=0
    )

    ev_arr = np.asarray(ev_offspring)
    mj_arr = np.asarray(jax.tree_util.tree_leaves(mj_offspring.genes)[0])
    diff = np.abs(ev_arr - mj_arr)
    print("\n== replay crossover parity ==")
    print(f"seed={args.seed}, pop_size={args.pop_size}, generations={args.generations}")
    print(f"offspring shape: {ev_arr.shape}")
    print(f"max diff: {float(diff.max()):.6e}")
    print(f"mean diff: {float(diff.mean()):.6e}")
    print(f"allclose: {bool(np.allclose(ev_arr, mj_arr))}")


def compare_selection(
    args: argparse.Namespace,
    trials: int = 100,
    *,
    elite_ratio_override: float | None = None,
    elite_k_override: int | None = None,
) -> None:
    """Run multiple trials comparing Evosax vs MalthusJAX selection on fitness stats.

    For each trial we sample a fresh initial population, run both selection
    mechanisms, and record the selected parents' fitness statistics (max,
    median, mean). After `trials` runs we perform paired statistical tests
    (t-test and Wilcoxon when available) to assess differences.
    """
    print(f"\n== selection operator comparison ({trials} trials) ==")
    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name=args.function, num_dims=args.dimensions, seed=args.seed, maximize=False)
    )

    # prepare engines/adapters once
    strategy_params = {
        "crossover_rate": args.crossover_rate,
        "elite_ratio": float(max(1, args.pop_size // 2)) / float(args.pop_size),
        "std_schedule": optax.constant_schedule(args.mutation_strength),
    }
    if elite_ratio_override is not None:
        strategy_params["elite_ratio"] = float(elite_ratio_override)

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=1,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params={
            **strategy_params,
        },
        initial_population=None,
    )

    genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
    engine_params = GeneticEngineParams(pop_size=args.pop_size, elitism=0, num_generations=1)
    # apply elite_k_override when constructing MalthusJAX selection
    m_elite_k = elite_k_override if elite_k_override is not None else max(1, args.pop_size // 2)

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=m_elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1, crossover_rate=args.crossover_rate
        ),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=args.mutation_strength),
        enable_progress_bar=False,
    )

    ev_diffs_max = []
    ev_diffs_med = []
    ev_diffs_mean = []
    ev_all = []
    mj_all = []

    try:
        from scipy import stats

        have_scipy = True
    except Exception:
        have_scipy = False

    for i in range(trials):
        # sample fresh population (vary seed per trial)
        seed_i = args.seed + 1000 + i
        key = jr.PRNGKey(seed_i)
        keys = jr.split(key, args.pop_size)
        population_init = jax.vmap(evaluator.evosax_problem.sample)(keys)

        # evaluate fitness
        key_eval = jr.PRNGKey(seed_i + 1)
        fitness_init, _, _ = evaluator.evosax_problem.eval(
            key_eval, population_init, evaluator.problem_state
        )

        # Evosax selection (replicate SimpleGA._ask sampling semantics)
        strategy = adapter.strategy
        params = adapter.params
        # init state with same population/fitness
        state = strategy.init(jr.PRNGKey(seed_i + 2), population_init, fitness_init, params)
        sorted_idx = np.asarray(jnp.argsort(state.fitness))
        pop_size = args.pop_size
        # respect elite_ratio_override when computing mask
        strategy_elite_ratio = float(
            getattr(strategy, "elite_ratio", strategy_params.get("elite_ratio", 0.5))
        )
        k_elite = int(strategy_elite_ratio * pop_size)
        p_mask = jnp.arange(pop_size) < k_elite
        # draw choices
        _, key_ask = jr.split(jr.PRNGKey(seed_i + 3), 2)
        k_cross, k_mut, k1, k2 = jax.random.split(key_ask, 4)
        sel1 = np.asarray(jax.random.choice(k1, jnp.arange(pop_size), (pop_size,), p=p_mask))
        sel2 = np.asarray(jax.random.choice(k2, jnp.arange(pop_size), (pop_size,), p=p_mask))
        ev_parent_idx = np.concatenate([sorted_idx[sel1], sorted_idx[sel2]])

        # MalthusJAX selection
        init_state = engine.init_state(jr.PRNGKey(seed_i))
        evaluated_population = evaluator.evaluate_population(
            RealPopulation.from_array(population_init, genome_config, axis=0)
        )
        state_m = init_state.replace(
            population=evaluated_population,
            best_genome=evaluated_population.genes[0],
            best_fitness=evaluated_population.fitness[0],
        )
        k_sel, _, _, _, _ = engine._allocate_entropy(state_m)
        _, mj_parent_idx = engine._selection_phase(
            k_sel, state_m.population, state_m.operators, engine.engine_params
        )
        mj_parent_idx = np.asarray(mj_parent_idx)

        # compute fitness stats for selected parents
        ev_parent_fitness = np.asarray(fitness_init)[ev_parent_idx]
        mj_parent_fitness = np.asarray(evaluated_population.fitness)[mj_parent_idx]

        ev_max = float(np.max(ev_parent_fitness))
        ev_med = float(np.median(ev_parent_fitness))
        ev_mean = float(np.mean(ev_parent_fitness))

        mj_max = float(np.max(mj_parent_fitness))
        mj_med = float(np.median(mj_parent_fitness))
        mj_mean = float(np.mean(mj_parent_fitness))

        ev_diffs_max.append(ev_max - mj_max)
        ev_diffs_med.append(ev_med - mj_med)
        ev_diffs_mean.append(ev_mean - mj_mean)
        ev_all.extend(ev_parent_fitness.tolist())
        mj_all.extend(mj_parent_fitness.tolist())

    ev_diffs_max = np.array(ev_diffs_max)
    ev_diffs_med = np.array(ev_diffs_med)
    ev_diffs_mean = np.array(ev_diffs_mean)

    def summarize(d: np.ndarray, name: str) -> None:
        mean = d.mean()
        std = d.std(ddof=1)
        cohens_d = mean / std if std > 0 else float("nan")
        print(f"\nMetric: {name}")
        print(f" mean difference (evosax - malthusjax): {mean:.6e}")
        print(f" std of differences: {std:.6e}")
        print(f" Cohen's d: {cohens_d:.4f}")
        if have_scipy:
            tstat, pval = stats.ttest_1samp(d, 0.0)
            try:
                wstat, w_p = stats.wilcoxon(d)
            except Exception:
                wstat, w_p = float("nan"), float("nan")
            print(f" paired t-test (one-sample vs 0): t={tstat:.4f}, p={pval:.4e}")
            print(f" wilcoxon: stat={wstat:.4f}, p={w_p:.4e}")
        else:
            # simple sign-flip permutation test for paired difference
            obs = mean
            rng = np.random.default_rng(0)
            perm_means = []
            for _ in range(10000):
                signs = rng.choice([-1, 1], size=d.shape[0])
                perm_means.append((d * signs).mean())
            perm_means = np.array(perm_means)
            pval = (np.abs(perm_means) >= abs(obs)).mean()
            print(f" permutation sign-flip p-value ~ {pval:.4e} (10000 samples)")

    summarize(ev_diffs_max, "max")
    summarize(ev_diffs_med, "median")
    summarize(ev_diffs_mean, "mean")

    # KS test on pooled selected-parent fitness distributions
    ev_all = np.asarray(ev_all)
    mj_all = np.asarray(mj_all)
    print("\nPooled selected-parent sample sizes:", ev_all.shape[0], mj_all.shape[0])
    if have_scipy:
        ks_stat, ks_p = stats.ks_2samp(ev_all, mj_all)
        print(f"KS test (two-sample): stat={ks_stat:.4f}, p={ks_p:.4e}")
    else:
        # naive KS implementation (two-sample)
        def ks_naive(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
            a = np.sort(a)
            b = np.sort(b)
            n, m = a.size, b.size
            data = np.concatenate([a, b])
            cdf_a = np.searchsorted(a, data, side="right") / n
            cdf_b = np.searchsorted(b, data, side="right") / m
            d = np.max(np.abs(cdf_a - cdf_b))
            # p-value approx for large n,m (not accurate for small samples)
            en = np.sqrt(n * m / (n + m))
            p = np.exp(-2 * (en * d) ** 2)
            return d, p

        ks_stat, ks_p = ks_naive(ev_all, mj_all)
        print(f"KS test (naive two-sample): stat={ks_stat:.4f}, approx_p={ks_p:.4e}")


def compare_mutation(
    args: argparse.Namespace,
    trials: int = 100,
    *,
    elite_ratio_override: float | None = None,
    elite_k_override: int | None = None,
) -> None:
    """Compare mutation operators across many trials (similar to compare_selection).

    Applies evosax's mutation primitive and MalthusJAX baked mutation to the same
    input populations, evaluates resulting fitnesses, and runs paired tests
    and a pooled KS test on selected metrics.
    """
    print(f"\n== mutation operator comparison ({trials} trials) ==")
    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name=args.function, num_dims=args.dimensions, seed=args.seed, maximize=False)
    )

    # Prepare adapters/engines like compare_selection
    strategy_params = {
        "crossover_rate": args.crossover_rate,
        "elite_ratio": float(max(1, args.pop_size // 2)) / float(args.pop_size),
        "std_schedule": optax.constant_schedule(args.mutation_strength),
    }
    if elite_ratio_override is not None:
        strategy_params["elite_ratio"] = float(elite_ratio_override)

    build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=1,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params=strategy_params,
        initial_population=None,
    )

    genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
    engine_params = GeneticEngineParams(pop_size=args.pop_size, elitism=0, num_generations=1)
    m_elite_k = elite_k_override if elite_k_override is not None else max(1, args.pop_size // 2)
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=m_elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1, crossover_rate=args.crossover_rate
        ),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=args.mutation_strength),
        enable_progress_bar=False,
    )

    ev_diffs_max = []
    ev_diffs_med = []
    ev_diffs_mean = []
    ev_diffs_dist = []
    ev_all = []
    mj_all = []
    ev_dist_parent = []
    mj_dist_parent = []

    try:
        from scipy import stats

        have_scipy = True
    except Exception:
        have_scipy = False

    for i in range(trials):
        seed_i = args.seed + 2000 + i
        key = jr.PRNGKey(seed_i)
        keys = jr.split(key, args.pop_size)
        population_init = jax.vmap(evaluator.evosax_problem.sample)(keys)

        # Evaluate initial fitness (not strictly needed but kept for parity)
        key_eval = jr.PRNGKey(seed_i + 1)
        fitness_init, _, _ = evaluator.evosax_problem.eval(
            key_eval, population_init, evaluator.problem_state
        )

        # Evosax mutation: apply primitive to each genome
        # Import evosax mutation function locally
        from evosax.algorithms.population_based.simple_ga import mutation as evosax_mutation_fn

        # derive per-individual subkeys
        k_mut = jr.PRNGKey(seed_i + 3)
        subkeys = jax.random.split(k_mut, args.pop_size)
        # vectorized application
        mutated_vals_ev = jax.vmap(
            lambda kk, sol: evosax_mutation_fn(kk, sol, args.mutation_strength)
        )(subkeys, population_init)
        ev_mut_pop = RealPopulation.from_array(np.asarray(mutated_vals_ev), genome_config, axis=0)
        ev_mut_eval = evaluator.evaluate_population(ev_mut_pop)

        # MalthusJAX mutation: use baked operator
        init_state = engine.init_state(jr.PRNGKey(seed_i))
        evaluated_population = evaluator.evaluate_population(
            RealPopulation.from_array(population_init, genome_config, axis=0)
        )
        state_m = init_state.replace(
            population=evaluated_population,
            best_genome=evaluated_population.genes[0],
            best_fitness=evaluated_population.fitness[0],
        )
        _, k_cross, k_mut_keys, _, _ = engine._allocate_entropy(state_m)
        # Call operator
        mj_mut_pop = state_m.operators.mutation(
            k_mut_keys, evaluated_population, engine.genome_config, generation=0
        )
        mj_mut_eval = evaluator.evaluate_population(mj_mut_pop)

        # Stats
        ev_f = np.asarray(ev_mut_eval.fitness)
        mj_f = np.asarray(mj_mut_eval.fitness)

        ev_max = float(np.max(ev_f))
        ev_med = float(np.median(ev_f))
        ev_mean = float(np.mean(ev_f))

        ev_dist = _mean_euclidean_distance(np.asarray(mutated_vals_ev), np.asarray(population_init))

        mj_max = float(np.max(mj_f))
        mj_med = float(np.median(mj_f))
        mj_mean = float(np.mean(mj_f))

        mj_dist = _mean_euclidean_distance(
            np.asarray(mj_mut_pop.genes if hasattr(mj_mut_pop, "genes") else mj_mut_pop),
            np.asarray(evaluated_population.genes),
        )

        ev_diffs_max.append(ev_max - mj_max)
        ev_diffs_med.append(ev_med - mj_med)
        ev_diffs_mean.append(ev_mean - mj_mean)
        ev_diffs_dist.append(ev_dist - mj_dist)
        ev_all.extend(ev_f.tolist())
        mj_all.extend(mj_f.tolist())
        ev_dist_parent.append(ev_dist)
        mj_dist_parent.append(mj_dist)

    ev_diffs_max = np.array(ev_diffs_max)
    ev_diffs_med = np.array(ev_diffs_med)
    ev_diffs_mean = np.array(ev_diffs_mean)
    ev_diffs_dist = np.array(ev_diffs_dist)

    def summarize(d: np.ndarray, name: str) -> None:
        mean = d.mean()
        std = d.std(ddof=1)
        cohens_d = mean / std if std > 0 else float("nan")
        print(f"\nMetric: {name}")
        print(f" mean difference (evosax - malthusjax): {mean:.6e}")
        print(f" std of differences: {std:.6e}")
        print(f" Cohen's d: {cohens_d:.4f}")
        if have_scipy:
            tstat, pval = stats.ttest_1samp(d, 0.0)
            try:
                wstat, w_p = stats.wilcoxon(d)
            except Exception:
                wstat, w_p = float("nan"), float("nan")
            print(f" paired t-test (one-sample vs 0): t={tstat:.4f}, p={pval:.4e}")
            print(f" wilcoxon: stat={wstat:.4f}, p={w_p:.4e}")
        else:
            rng = np.random.default_rng(0)
            obs = mean
            perm_means = []
            for _ in range(10000):
                signs = rng.choice([-1, 1], size=d.shape[0])
                perm_means.append((d * signs).mean())
            perm_means = np.array(perm_means)
            pval = (np.abs(perm_means) >= abs(obs)).mean()
            print(f" permutation sign-flip p-value ~ {pval:.4e} (10000 samples)")

    summarize(ev_diffs_max, "max")
    summarize(ev_diffs_med, "median")
    summarize(ev_diffs_mean, "mean")
    summarize(ev_diffs_dist, "avg_parent_distance")

    ev_all = np.asarray(ev_all)
    mj_all = np.asarray(mj_all)
    print("\nPooled mutation sample sizes:", ev_all.shape[0], mj_all.shape[0])
    print("Evosax mutation parent distance mean:", float(np.mean(ev_dist_parent)))
    print("MalthusJAX mutation parent distance mean:", float(np.mean(mj_dist_parent)))
    if have_scipy:
        ks_stat, ks_p = stats.ks_2samp(ev_all, mj_all)
        print(f"KS test (two-sample): stat={ks_stat:.4f}, p={ks_p:.4e}")
    else:

        def ks_naive(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
            a = np.sort(a)
            b = np.sort(b)
            n, m = a.size, b.size
            data = np.concatenate([a, b])
            cdf_a = np.searchsorted(a, data, side="right") / n
            cdf_b = np.searchsorted(b, data, side="right") / m
            d = np.max(np.abs(cdf_a - cdf_b))
            en = np.sqrt(n * m / (n + m))
            p = np.exp(-2 * (en * d) ** 2)
            return d, p

        ks_stat, ks_p = ks_naive(ev_all, mj_all)
        print(f"KS test (naive two-sample): stat={ks_stat:.4f}, approx_p={ks_p:.4e}")


def compare_crossover(
    args: argparse.Namespace,
    trials: int = 100,
    *,
    elite_ratio_override: float | None = None,
    elite_k_override: int | None = None,
) -> None:
    """Compare crossover operators across many trials.

    For each trial we sample an initial population, build parent pairs using
    the Evosax SimpleGA sampling semantics, run the Evosax crossover primitive
    and the MalthusJAX baked crossover operator, then compare offspring
    fitness statistics (max, median, mean) across trials.
    """
    print(f"\n== crossover operator comparison ({trials} trials) ==")
    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name=args.function, num_dims=args.dimensions, seed=args.seed, maximize=False)
    )

    strategy_params = {
        "crossover_rate": args.crossover_rate,
        "elite_ratio": float(max(1, args.pop_size // 2)) / float(args.pop_size),
        "std_schedule": optax.constant_schedule(args.mutation_strength),
    }
    if elite_ratio_override is not None:
        strategy_params["elite_ratio"] = float(elite_ratio_override)

    adapter = build_evosax_engine(
        strategy_name="SimpleGA",
        evaluator=evaluator,
        pop_size=args.pop_size,
        generations=1,
        bounds=(-5.0, 5.0),
        maximize=False,
        seed=args.seed,
        strategy_params=strategy_params,
        initial_population=None,
    )

    genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
    engine_params = GeneticEngineParams(pop_size=args.pop_size, elitism=0, num_generations=1)
    m_elite_k = elite_k_override if elite_k_override is not None else max(1, args.pop_size // 2)
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=args.pop_size, elite_k=m_elite_k),
        crossover=EvosaxUniformCrossoverWrapper(
            num_offspring=1, crossover_rate=args.crossover_rate
        ),
        mutation=EvosaxGaussianWrapper(num_offspring=1, mutation_strength=args.mutation_strength),
        enable_progress_bar=False,
    )

    ev_diffs_max = []
    ev_diffs_med = []
    ev_diffs_mean = []
    ev_diffs_dist = []
    ev_all = []
    mj_all = []
    ev_dist_parent1 = []
    ev_dist_parent2 = []
    mj_dist_parent1 = []
    mj_dist_parent2 = []

    try:
        from scipy import stats

        have_scipy = True
    except Exception:
        have_scipy = False

    from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover_fn

    for i in range(trials):
        seed_i = args.seed + 3000 + i
        key = jr.PRNGKey(seed_i)
        keys = jr.split(key, args.pop_size)
        population_init = jax.vmap(evaluator.evosax_problem.sample)(keys)

        # fitness (needed for selection sampling)
        key_eval = jr.PRNGKey(seed_i + 1)
        fitness_init, _, _ = evaluator.evosax_problem.eval(
            key_eval, population_init, evaluator.problem_state
        )

        # Evosax parent sampling (SimpleGA semantics)
        strategy = adapter.strategy
        params = adapter.params
        state = strategy.init(jr.PRNGKey(seed_i + 2), population_init, fitness_init, params)
        sorted_idx = np.asarray(jnp.argsort(state.fitness))
        pop_size = args.pop_size
        strategy_elite_ratio = float(
            getattr(strategy, "elite_ratio", strategy_params.get("elite_ratio", 0.5))
        )
        k_elite = int(strategy_elite_ratio * pop_size)
        p_mask = jnp.arange(pop_size) < k_elite
        _, key_ask = jr.split(jr.PRNGKey(seed_i + 3), 2)
        k_cross, k_mut, k1, k2 = jax.random.split(key_ask, 4)
        # per-individual crossover keys
        cross_keys = jax.random.split(k_cross, pop_size)
        parents_1 = jax.random.choice(k1, population_init, (pop_size,), p=p_mask)
        parents_2 = jax.random.choice(k2, population_init, (pop_size,), p=p_mask)
        # Evosax crossover primitive (vectorized)
        ev_offspring = jax.vmap(evosax_crossover_fn, in_axes=(0, 0, 0, None))(
            cross_keys, parents_1, parents_2, args.crossover_rate
        )
        ev_off_pop = RealPopulation.from_array(np.asarray(ev_offspring), genome_config, axis=0)
        ev_off_eval = evaluator.evaluate_population(ev_off_pop)

        # MalthusJAX crossover: prepare p1_pop / p2_pop like engine does
        init_state = engine.init_state(jr.PRNGKey(seed_i))
        evaluated_population = evaluator.evaluate_population(
            RealPopulation.from_array(population_init, genome_config, axis=0)
        )
        # need parent indices as in selection sampling to extract parents
        # reuse evosax sampled indices mapping to sorted order
        sel1 = np.asarray(jax.random.choice(k1, jnp.arange(pop_size), (pop_size,), p=p_mask))
        sel2 = np.asarray(jax.random.choice(k2, jnp.arange(pop_size), (pop_size,), p=p_mask))
        p1_idx = np.asarray(sorted_idx)[sel1]
        p2_idx = np.asarray(sorted_idx)[sel2]
        p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], evaluated_population.genes)
        p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], evaluated_population.genes)
        dummy_fitness = jnp.zeros(pop_size)
        p1_pop = evaluated_population.spawn_offspring(p1_genes, fitness=dummy_fitness)
        p2_pop = evaluated_population.spawn_offspring(p2_genes, fitness=dummy_fitness)

        k_sel, k_cross_keys, k_mut_keys, _, k_next = engine._allocate_entropy(init_state)
        # call the baked crossover operator — use the originating `k_cross` so the
        # wrapper's internal split matches Evosax's per-pair keys (parity test)
        mj_cross_off = init_state.operators.crossover(
            jnp.expand_dims(k_cross, 0), p1_pop, p2_pop, engine.genome_config, generation=0
        )
        mj_off_eval = evaluator.evaluate_population(mj_cross_off)

        # compute stats
        ev_f = np.asarray(ev_off_eval.fitness)
        mj_f = np.asarray(mj_off_eval.fitness)

        ev_max = float(np.max(ev_f))
        ev_med = float(np.median(ev_f))
        ev_mean = float(np.mean(ev_f))

        ev_off = np.asarray(ev_offspring)
        ev_dist_1 = _mean_euclidean_distance(ev_off, np.asarray(parents_1))
        ev_dist_2 = _mean_euclidean_distance(ev_off, np.asarray(parents_2))
        ev_dist = 0.5 * (ev_dist_1 + ev_dist_2)

        mj_max = float(np.max(mj_f))
        mj_med = float(np.median(mj_f))
        mj_mean = float(np.mean(mj_f))

        mj_off = np.asarray(jax.tree_util.tree_leaves(mj_cross_off.genes)[0])
        mj_dist_1 = _mean_euclidean_distance(mj_off, np.asarray(p1_pop.genes))
        mj_dist_2 = _mean_euclidean_distance(mj_off, np.asarray(p2_pop.genes))
        mj_dist = 0.5 * (mj_dist_1 + mj_dist_2)

        ev_diffs_max.append(ev_max - mj_max)
        ev_diffs_med.append(ev_med - mj_med)
        ev_diffs_mean.append(ev_mean - mj_mean)
        ev_diffs_dist.append(ev_dist - mj_dist)
        ev_all.extend(ev_f.tolist())
        mj_all.extend(mj_f.tolist())
        ev_dist_parent1.append(ev_dist_1)
        ev_dist_parent2.append(ev_dist_2)
        mj_dist_parent1.append(mj_dist_1)
        mj_dist_parent2.append(mj_dist_2)

    ev_diffs_max = np.array(ev_diffs_max)
    ev_diffs_med = np.array(ev_diffs_med)
    ev_diffs_mean = np.array(ev_diffs_mean)
    ev_diffs_dist = np.array(ev_diffs_dist)

    def summarize(d: np.ndarray, name: str) -> None:
        mean = d.mean()
        std = d.std(ddof=1)
        cohens_d = mean / std if std > 0 else float("nan")
        print(f"\nMetric: {name}")
        print(f" mean difference (evosax - malthusjax): {mean:.6e}")
        print(f" std of differences: {std:.6e}")
        print(f" Cohen's d: {cohens_d:.4f}")
        if have_scipy:
            tstat, pval = stats.ttest_1samp(d, 0.0)
            try:
                wstat, w_p = stats.wilcoxon(d)
            except Exception:
                wstat, w_p = float("nan"), float("nan")
            print(f" paired t-test (one-sample vs 0): t={tstat:.4f}, p={pval:.4e}")
            print(f" wilcoxon: stat={wstat:.4f}, p={w_p:.4e}")
        else:
            rng = np.random.default_rng(0)
            obs = mean
            perm_means = []
            for _ in range(10000):
                signs = rng.choice([-1, 1], size=d.shape[0])
                perm_means.append((d * signs).mean())
            perm_means = np.array(perm_means)
            pval = (np.abs(perm_means) >= abs(obs)).mean()
            print(f" permutation sign-flip p-value ~ {pval:.4e} (10000 samples)")

    summarize(ev_diffs_max, "max")
    summarize(ev_diffs_med, "median")
    summarize(ev_diffs_mean, "mean")
    summarize(ev_diffs_dist, "avg_parent_distance")

    ev_all = np.asarray(ev_all)
    mj_all = np.asarray(mj_all)
    print("\nPooled crossover sample sizes:", ev_all.shape[0], mj_all.shape[0])
    print(
        "Evosax parent distances (p1/p2 mean):",
        float(np.mean(ev_dist_parent1)),
        float(np.mean(ev_dist_parent2)),
    )
    print(
        "MalthusJAX parent distances (p1/p2 mean):",
        float(np.mean(mj_dist_parent1)),
        float(np.mean(mj_dist_parent2)),
    )
    if have_scipy:
        ks_stat, ks_p = stats.ks_2samp(ev_all, mj_all)
        print(f"KS test (two-sample): stat={ks_stat:.4f}, p={ks_p:.4e}")
    else:

        def ks_naive(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
            a = np.sort(a)
            b = np.sort(b)
            n, m = a.size, b.size
            data = np.concatenate([a, b])
            cdf_a = np.searchsorted(a, data, side="right") / n
            cdf_b = np.searchsorted(b, data, side="right") / m
            d = np.max(np.abs(cdf_a - cdf_b))
            en = np.sqrt(n * m / (n + m))
            p = np.exp(-2 * (en * d) ** 2)
            return d, p

        ks_stat, ks_p = ks_naive(ev_all, mj_all)
        print(f"KS test (naive two-sample): stat={ks_stat:.4f}, approx_p={ks_p:.4e}")


def analyze_and_capture_crossover(
    args: argparse.Namespace, trials: int = 100, top_k: int = 3
) -> None:
    """Screen trials to find the top-k largest mean-difference crossover trials, then capture detailed traces.

    Saves per-trial trace files to `results/crossover_trace_{seed}.npz` and a human-readable
    summary `results/crossover_trace_{seed}.txt` containing keys, masks and small array samples.
    """
    print(f"\n== crossover capture: screening {trials} trials, capturing top {top_k} ==")
    evaluator = BBOBEvaluator.create(
        BBOBConfig(fn_name=args.function, num_dims=args.dimensions, seed=args.seed, maximize=False)
    )

    seeds = [args.seed + 4000 + i for i in range(trials)]
    diffs = []
    # Quick screening: compute mean offspring fitness difference per trial
    for seed_i in seeds:
        key = jr.PRNGKey(seed_i)
        keys = jr.split(key, args.pop_size)
        population_init = jax.vmap(evaluator.evosax_problem.sample)(keys)
        key_eval = jr.PRNGKey(seed_i + 1)
        fitness_init, _, _ = evaluator.evosax_problem.eval(
            key_eval, population_init, evaluator.problem_state
        )

        # Evosax crossover (one-shot like in compare_crossover)
        from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover_fn

        # sample parents same as SimpleGA
        k_ask = jr.PRNGKey(seed_i + 3)
        k_cross, k_mut, k1, k2 = jax.random.split(k_ask, 4)
        strategy = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evaluator,
            pop_size=args.pop_size,
            generations=1,
            bounds=(-5.0, 5.0),
            maximize=False,
            seed=args.seed,
            strategy_params=_shared_strategy_params(args, max(1, args.pop_size // 2)),
            initial_population=None,
        ).strategy
        state = strategy.init(
            jr.PRNGKey(seed_i + 2), population_init, fitness_init, strategy.default_params
        )
        sorted_idx = np.asarray(jnp.argsort(state.fitness))
        pop_size = args.pop_size
        p_mask = jnp.arange(pop_size) < int(getattr(strategy, "elite_ratio", 0.5) * pop_size)
        parents_1 = jax.random.choice(k1, population_init, (pop_size,), p=p_mask)
        parents_2 = jax.random.choice(k2, population_init, (pop_size,), p=p_mask)
        cross_keys = jax.random.split(k_cross, pop_size)
        ev_offspring = jax.vmap(evosax_crossover_fn, in_axes=(0, 0, 0, None))(
            cross_keys, parents_1, parents_2, args.crossover_rate
        )
        ev_off_eval = evaluator.evaluate_population(
            RealPopulation.from_array(
                np.asarray(ev_offspring),
                RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0)),
                axis=0,
            )
        )
        ev_mean = float(np.mean(np.asarray(ev_off_eval.fitness)))

        # MalthusJAX crossover
        genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
        engine = GeneticEngine(
            engine_params=GeneticEngineParams(pop_size=args.pop_size, elitism=0, num_generations=1),
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(
                num_selections=args.pop_size, elite_k=max(1, args.pop_size // 2)
            ),
            crossover=EvosaxUniformCrossoverWrapper(
                num_offspring=1, crossover_rate=args.crossover_rate
            ),
            mutation=EvosaxGaussianWrapper(
                num_offspring=1, mutation_strength=args.mutation_strength
            ),
            enable_progress_bar=False,
        )
        init_state = engine.init_state(jr.PRNGKey(seed_i))
        evaluated_population = evaluator.evaluate_population(
            RealPopulation.from_array(population_init, genome_config, axis=0)
        )
        # reuse parent sampling indices as above
        sel1 = np.asarray(jax.random.choice(k1, jnp.arange(pop_size), (pop_size,), p=p_mask))
        sel2 = np.asarray(jax.random.choice(k2, jnp.arange(pop_size), (pop_size,), p=p_mask))
        p1_idx = np.asarray(sorted_idx)[sel1]
        p2_idx = np.asarray(sorted_idx)[sel2]
        p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], evaluated_population.genes)
        p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], evaluated_population.genes)
        dummy_fitness = jnp.zeros(pop_size)
        p1_pop = evaluated_population.spawn_offspring(p1_genes, fitness=dummy_fitness)
        p2_pop = evaluated_population.spawn_offspring(p2_genes, fitness=dummy_fitness)
        k_sel, k_cross_keys, k_mut_keys, _, k_next = engine._allocate_entropy(init_state)
        mj_cross_off = init_state.operators.crossover(
            jnp.expand_dims(k_cross, 0), p1_pop, p2_pop, engine.genome_config, generation=0
        )
        mj_off_eval = evaluator.evaluate_population(mj_cross_off)
        mj_mean = float(np.mean(np.asarray(mj_off_eval.fitness)))

        diffs.append((abs(ev_mean - mj_mean), seed_i))

    # pick top-k seeds
    diffs.sort(reverse=True)
    top = diffs[:top_k]
    print("Top differing trials (abs mean diff, seed):", top)

    # capture detailed traces for each top seed
    Path("results").mkdir(parents=True, exist_ok=True)
    for diff_val, seed_i in top:
        print(f"\nCapturing trace for seed {seed_i} (abs mean diff={diff_val:.6e})")
        # re-run trial with detailed capture
        key = jr.PRNGKey(seed_i)
        keys = jr.split(key, args.pop_size)
        population_init = jax.vmap(evaluator.evosax_problem.sample)(keys)
        key_eval = jr.PRNGKey(seed_i + 1)
        fitness_init, _, _ = evaluator.evosax_problem.eval(
            key_eval, population_init, evaluator.problem_state
        )

        # prepare evosax pieces
        k_ask = jr.PRNGKey(seed_i + 3)
        k_cross, k_mut, k1, k2 = jax.random.split(k_ask, 4)
        strategy = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evaluator,
            pop_size=args.pop_size,
            generations=1,
            bounds=(-5.0, 5.0),
            maximize=False,
            seed=args.seed,
            strategy_params=_shared_strategy_params(args, max(1, args.pop_size // 2)),
            initial_population=None,
        ).strategy
        state = strategy.init(
            jr.PRNGKey(seed_i + 2), population_init, fitness_init, strategy.default_params
        )
        sorted_idx = np.asarray(jnp.argsort(state.fitness))
        pop_size = args.pop_size
        p_mask = jnp.arange(pop_size) < int(getattr(strategy, "elite_ratio", 0.5) * pop_size)
        parents_1 = jax.random.choice(k1, population_init, (pop_size,), p=p_mask)
        parents_2 = jax.random.choice(k2, population_init, (pop_size,), p=p_mask)
        cross_keys = jax.random.split(k_cross, pop_size)
        from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover_fn

        ev_offspring = jax.vmap(evosax_crossover_fn, in_axes=(0, 0, 0, None))(
            cross_keys, parents_1, parents_2, args.crossover_rate
        )

        # MalthusJAX pieces
        genome_config = RealGenomeConfig(shape=(args.dimensions,), bounds=(-5.0, 5.0))
        engine = GeneticEngine(
            engine_params=GeneticEngineParams(pop_size=args.pop_size, elitism=0, num_generations=1),
            genome_config=genome_config,
            evaluator=evaluator,
            selection=ElitePoolSelection(
                num_selections=args.pop_size, elite_k=max(1, args.pop_size // 2)
            ),
            crossover=EvosaxUniformCrossoverWrapper(
                num_offspring=1, crossover_rate=args.crossover_rate
            ),
            mutation=EvosaxGaussianWrapper(
                num_offspring=1, mutation_strength=args.mutation_strength
            ),
            enable_progress_bar=False,
        )
        init_state = engine.init_state(jr.PRNGKey(seed_i))
        evaluated_population = evaluator.evaluate_population(
            RealPopulation.from_array(population_init, genome_config, axis=0)
        )
        sel1 = np.asarray(jax.random.choice(k1, jnp.arange(pop_size), (pop_size,), p=p_mask))
        sel2 = np.asarray(jax.random.choice(k2, jnp.arange(pop_size), (pop_size,), p=p_mask))
        p1_idx = np.asarray(sorted_idx)[sel1]
        p2_idx = np.asarray(sorted_idx)[sel2]
        p1_genes = jax.tree_util.tree_map(lambda x: x[p1_idx], evaluated_population.genes)
        p2_genes = jax.tree_util.tree_map(lambda x: x[p2_idx], evaluated_population.genes)
        dummy_fitness = jnp.zeros(pop_size)
        p1_pop = evaluated_population.spawn_offspring(p1_genes, fitness=dummy_fitness)
        p2_pop = evaluated_population.spawn_offspring(p2_genes, fitness=dummy_fitness)
        k_sel, k_cross_keys, k_mut_keys, _, k_next = engine._allocate_entropy(init_state)
        mj_cross_off = init_state.operators.crossover(
            jnp.expand_dims(k_cross, 0), p1_pop, p2_pop, engine.genome_config, generation=0
        )

        # Save traces
        out_npz = Path(f"results/crossover_trace_{seed_i}.npz")
        np.savez_compressed(
            str(out_npz),
            population_init=np.asarray(population_init),
            fitness_init=np.asarray(fitness_init),
            parents_1=np.asarray(parents_1),
            parents_2=np.asarray(parents_2),
            cross_keys=np.asarray(cross_keys),
            ev_offspring=np.asarray(ev_offspring),
            p1_idx=p1_idx,
            p2_idx=p2_idx,
            p1_pop=np.asarray(p1_pop.genes),
            p2_pop=np.asarray(p2_pop.genes),
            k_cross_keys=np.asarray(k_cross_keys),
            # Save the two independent parent-draw keys used by Evosax capture
            # so replays can inject the exact same RNG streams.
            k_parent_keys=np.asarray([k1, k2]),
            mj_offspring=np.asarray(jax.tree_util.tree_leaves(mj_cross_off.genes)[0]),
        )

        # Write human-readable summary
        out_txt = Path(f"results/crossover_trace_{seed_i}.txt")
        with out_txt.open("w") as f:
            f.write(f"seed: {seed_i}\n")
            f.write(f"abs mean diff: {diff_val:.6e}\n")
            f.write(f"population_init shape: {np.asarray(population_init).shape}\n")
            f.write(f"fitness_init[0:5]: {np.asarray(fitness_init)[:5].tolist()}\n")
            f.write(f"parents_1[0]: {np.asarray(parents_1)[0].tolist()}\n")
            f.write(f"parents_2[0]: {np.asarray(parents_2)[0].tolist()}\n")
            f.write(
                f"cross_keys[0]: {np.asarray(cross_keys)[0].tolist() if np.asarray(cross_keys).ndim > 1 else np.asarray(cross_keys)[0]}\n"
            )
            f.write(f"ev_offspring[0]: {np.asarray(ev_offspring)[0].tolist()}\n")
            f.write(f"p1_idx[:10]: {p1_idx[:10].tolist()}\n")
            f.write(f"p2_idx[:10]: {p2_idx[:10].tolist()}\n")
            f.write(f"k_cross_keys shape: {np.asarray(k_cross_keys).shape}\n")
            f.write(
                f"mj_offspring[0]: {np.asarray(jax.tree_util.tree_leaves(mj_cross_off.genes)[0])[0].tolist()}\n"
            )
        print(f"Saved trace files: {out_npz}, {out_txt}")


def main() -> None:
    args = parse_args()
    if args.replay_selection_parity:
        replay_selection_parity(args)
        return
    if args.replay_crossover_parity:
        replay_crossover_parity(args)
        return
    # Compare operators side-by-side when requested (default for "both")
    if args.backend == "both":
        if args.capture_top_k > 0:
            # screening + capture workflow for crossover diffs
            analyze_and_capture_crossover(
                args, trials=args.capture_trials, top_k=args.capture_top_k
            )
            return

        if getattr(args, "param_sweep", False):
            # default ratios to test
            ratios = [0.125, 0.25, 0.5, 1.0]
            print("\n== parameter sweep: elite_ratio / elite_k values ==")
            for r in ratios:
                m_k = max(1, int(r * args.pop_size))
                print(f"\n-- testing elite_ratio={r:.3f} (elite_k={m_k}) --")
                compare_selection(
                    args, trials=args.compare_trials, elite_ratio_override=r, elite_k_override=m_k
                )
                compare_mutation(
                    args, trials=args.compare_trials, elite_ratio_override=r, elite_k_override=m_k
                )
                # compare crossover as well
                compare_crossover(
                    args, trials=args.compare_trials, elite_ratio_override=r, elite_k_override=m_k
                )
        else:
            compare_selection(args, trials=args.compare_trials)
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
        ax.set_title(f"Toy debug run: gap to optimum over time ({args.function})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
