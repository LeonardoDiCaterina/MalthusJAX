"""
MalthusJAX vs Evosax — pytest-benchmark Snapshot Suite
=======================================================

Phase 0 baseline benchmarks for tracking performance across the fix plan.
Run with:
    pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-only
    pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-save=baseline
    pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-compare=0001_baseline

Compares:
    - Single-step latency (warm)
    - Multi-generation throughput (scan loop)
    - JIT compilation time
    - Operator-level microbenchmarks (selection, crossover, mutation, fitness)

    **Note:** the engines now record the starting and ending best fitness
    values (and the delta between them) in the benchmark summaries so that the
    results show whether evolution is making real improvements instead of
    merely churning random numbers.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from evosax.algorithms.population_based import SimpleGA
from evosax.problems import BBOBProblem

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.base import _get_evolution_kernel
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
)
from malthusjax.engine.schedules import TrackBest
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation
# wrappers around native evosax implementations; used to isolate
# architectural overhead by keeping the same operators as evosax itself
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.tournament import TournamentSelection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
"""fn_names_short_dict = {
    # Part 1: Separable functions
    "sphere": "Sphere",
    "ellipsoidal": "Ellipsoidal",
    "rastrigin": "Rastrigin",
    "bueche_rastrigin": "Büche-Rastrigin",
    "linear_slope": "Linear Slope",
    # Part 2: Functions with low or moderate conditioning
    "attractive_sector": "Attractive Sector",
    "step_ellipsoidal": "Step Ellipsoidal",
    "rosenbrock": "Rosenbrock",
    "rosenbrock_rotated": "Rosenbrock Rotated",
    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_rotated": "Ellipsoidal Rotated",
    "discus": "Discus",
    "bent_cigar": "Bent Cigar",
    "sharp_ridge": "Sharp Ridge",
    "different_powers": "Different Powers",
    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_rotated": "Rastrigin Rotated",
    "weierstrass": "Weierstrass",
    "schaffers_f7": "Schaffers F7",
    "schaffers_f7_ill_cond": "Schaffers F7 Ill-cond.",
    "griewank_rosenbrock": "Griewank-Rosenbrock",
    # Part 5: Multi-modal functions with weak global structure
    "schwefel": "Schwefel",
    "gallagher_101_me": "Gallagher 101-me",
    "gallagher_21_hi": "Gallagher 21-hi",
    "katsuura": "Katsuura",
    "lunacek": "Lunacek",
}

"""

SEED = 42
PROBLEMS = ["sphere", "rastrigin", "ellipsoidal_rotated", "schwefel", "griewank_rosenbrock"]  # subset of BBOB functions for benchmarking
DIMENSIONS = [10, 50]
POP_SIZES = [100, 500, 1024, 1025]  # powers of two and just above to test vectorization effects
NUM_GENERATIONS_SHORT = 50
NUM_GENERATIONS_LONG = 500
UNROLL_FACTORS = [1, 5, 10, 25]  # lax.scan unroll sweep


# ---------------------------------------------------------------------------
# Helpers — MalthusJAX
# ---------------------------------------------------------------------------


def _build_malthusjax_engine(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    num_generations: int = 1,
    elite_ratio: float = 0.5,
    selection_type: str = "elite_pool",
    unroll_num: int = 1,
    track_best: TrackBest = TrackBest.LIGHT,
    use_evosax_ops: bool = False,
) -> GeneticEngine:
    """Build a ready-to-use MalthusJAX GeneticEngine.

    When ``use_evosax_ops`` is True the engine will employ the
    ``EvosaxUniformCrossoverWrapper`` and ``EvosaxGaussianWrapper``
    rather than the native implementations.  This lets us measure the
    costs of the MalthusJAX engine architecture itself while holding the
    genetic operators constant with evosax's own operators.
    """
    genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))

    bbob_config = BBOBConfig(fn_name=problem, num_dims=dims, seed=SEED, maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)

    elite_count = max(1, int(pop_size * elite_ratio))

    if selection_type == "tournament":
        selection = TournamentSelection(num_selections=pop_size, tournament_size=3)
    else:
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=elite_count)

    if use_evosax_ops:
        crossover = EvosaxUniformCrossoverWrapper(num_offspring=1, crossover_rate=0.5)
        # Evosax wrapper only accepts mutation_strength; drop mutation_rate
        mutation = EvosaxGaussianWrapper(num_offspring=1, mutation_strength=0.1)
    else:
        crossover = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1)

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=num_generations,
        elitism=elite_count,
        unroll_num=unroll_num,
        track_best=track_best,
    )

    return GeneticEngine(
        evaluator=evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params,
        enable_progress_bar=False,
    )


def _malthusjax_init_and_warmup(
    engine: GeneticEngine,
) -> Tuple[GeneticEvolutionState, Callable]:
    """Init state + JIT-compile and warm up the step function, returning (state, jit_step)."""
    key = jr.PRNGKey(SEED)
    state = engine.init_state(key)

    jit_step = jax.jit(engine.step)
    # Warm-up: compile once
    _warmup_state, _ = jit_step(state)
    _warmup_state.best_fitness.block_until_ready()

    return state, jit_step


# ---------------------------------------------------------------------------
# Helpers — Evosax
# ---------------------------------------------------------------------------


def _build_evosax_ga(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    elite_ratio: float = 0.5,
) -> Tuple[SimpleGA, Any, BBOBProblem, Any]:
    """Build evosax SimpleGA + BBOB problem, return (strategy, params, problem, init_carry)."""
    rng = jr.PRNGKey(SEED)

    es_problem = BBOBProblem(problem, num_dims=dims, seed=SEED)
    init_solution = es_problem.sample(rng)

    strategy = SimpleGA(population_size=pop_size, solution=init_solution)
    strategy.elite_ratio = elite_ratio
    es_params = strategy.default_params.replace(crossover_rate=0.5)

    # Build initial carry = (es_state, problem_state, rng)
    r_init, r_start = jax.random.split(rng)
    p_state = es_problem.init(r_init)

    init_x = jax.random.uniform(r_init, (pop_size, dims), minval=-5.0, maxval=5.0)
    init_fit = jnp.full((pop_size,), jnp.inf)
    es_state = strategy.init(r_init, init_x, init_fit, es_params)

    carry = (es_state, p_state, r_start)
    return strategy, es_params, es_problem, carry


def _evosax_step_fn(
    strategy: SimpleGA, params: Any, problem: BBOBProblem
) -> Callable:
    """Return a scan-compatible step function for evosax."""

    def step(carry, _=None):
        state, p_state, rng = carry
        rng, rng_step = jax.random.split(rng)
        x, state = strategy.ask(rng_step, state, params)
        fitness, p_state, _ = problem.eval(rng_step, x, p_state)
        state, _ = strategy.tell(rng_step, x, fitness, state, params)
        return (state, p_state, rng), None

    return step


def _evosax_init_and_warmup(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
) -> Tuple[Any, Callable]:
    """Init + JIT-compile and warm up evosax step, returning (carry, jit_step)."""
    strategy, params, es_problem, carry = _build_evosax_ga(pop_size, dims, problem)
    step = _evosax_step_fn(strategy, params, es_problem)
    jit_step = jax.jit(step)

    # Warm-up
    _warmup_carry, _ = jit_step(carry)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), _warmup_carry)

    return carry, jit_step


# ---------------------------------------------------------------------------
# Engine Protocol Adapters — for BenchmarkRunner integration
# ---------------------------------------------------------------------------


@dataclass
class MalthusJAXBenchEngine:
    """Wraps a MalthusJAX GeneticEngine to satisfy the ``Engine`` protocol.

    ``run_once(key)`` returns the standard dict expected by
    :class:`BenchmarkRunner`: ``{history, summary, timings}``.

    The `use_evosax_ops` flag controls whether the engine is constructed
    with native MalthusJAX operators or the evosax wrappers.  This allows
    us to measure the overhead of the engine architecture itself.
    """

    pop_size: int
    dims: int
    problem: str = "sphere"
    num_generations: int = NUM_GENERATIONS_LONG
    elite_ratio: float = 0.5
    unroll_num: int = 1
    use_evosax_ops: bool = False

    def run_once(self, key: jax.Array) -> Dict[str, Any]:
        engine = _build_malthusjax_engine(
            self.pop_size,
            self.dims,
            problem=self.problem,
            num_generations=self.num_generations,
            elite_ratio=self.elite_ratio,
            unroll_num=self.unroll_num,
            use_evosax_ops=self.use_evosax_ops,
        )
        t0 = time.time()
        state = engine.init_state(key)
        t_init = time.time() - t0

        t0 = time.time()
        final_state, scan_history, _ = engine.run(state, compile=True)
        final_state.best_fitness.block_until_ready()
        t_evo = time.time() - t0

        # Unpack stacked scan history arrays → list of dicts
        n_gens = int(scan_history.generation.shape[0])
        history: List[Dict[str, Any]] = []
        for g in range(n_gens):
            history.append(
                {
                    "generation": int(scan_history.generation[g]),
                    "best_fitness": float(scan_history.best_fitness[g]),
                    "mean_fitness": float(scan_history.mean_fitness[g]),
                }
            )

        # compute fitness deltas for reporting
        start_best = history[0]["best_fitness"]
        end_best = history[-1]["best_fitness"]
        delta_best = start_best - end_best  # positive improvement for minimisation

        summary = {
            "best_fitness": float(final_state.best_fitness),
            "start_best_fitness": start_best,
            "end_best_fitness": end_best,
            "delta_best": delta_best,
            "final_generation": n_gens - 1,
            "total_evaluations": n_gens * self.pop_size,
        }
        timings = {"initialization": t_init, "evolution": t_evo}
        return {"history": history, "summary": summary, "timings": timings}


@dataclass
class EvosaxBenchEngine:
    """Wraps evosax SimpleGA to satisfy the ``Engine`` protocol.

    ``run_once(key)`` returns the standard dict expected by
    :class:`BenchmarkRunner`: ``{history, summary, timings}``.
    """

    pop_size: int
    dims: int
    problem: str = "sphere"
    num_generations: int = NUM_GENERATIONS_LONG
    elite_ratio: float = 0.5

    def run_once(self, key: jax.Array) -> Dict[str, Any]:
        strategy, params, es_problem, carry = _build_evosax_ga(
            self.pop_size, self.dims, self.problem, self.elite_ratio
        )
        step = _evosax_step_fn(strategy, params, es_problem)

        # Record per-generation history via a modified scan that outputs fitness
        def step_with_output(carry, _):
            state, p_state, rng = carry
            rng, rng_step = jax.random.split(rng)
            x, state = strategy.ask(rng_step, state, params)
            fitness, p_state, _ = es_problem.eval(rng_step, x, p_state)
            state, _ = strategy.tell(rng_step, x, fitness, state, params)
            new_carry = (state, p_state, rng)
            output = {
                "best_fitness": state.best_fitness,
                "mean_fitness": jnp.mean(fitness),
            }
            return new_carry, output

        jit_scan = jax.jit(
            lambda c: jax.lax.scan(step_with_output, c, None, length=self.num_generations)
        )

        t0 = time.time()
        final_carry, gen_outputs = jit_scan(carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_carry)
        t_evo = time.time() - t0

        es_state, _, _ = final_carry
        n_gens = self.num_generations

        history: List[Dict[str, Any]] = []
        for g in range(n_gens):
            history.append(
                {
                    "generation": g,
                    "best_fitness": float(gen_outputs["best_fitness"][g]),
                    "mean_fitness": float(gen_outputs["mean_fitness"][g]),
                }
            )

        # fitness deltas
        start_best = history[0]["best_fitness"]
        end_best = history[-1]["best_fitness"]
        delta_best = start_best - end_best

        summary = {
            "best_fitness": float(es_state.best_fitness),
            "start_best_fitness": start_best,
            "end_best_fitness": end_best,
            "delta_best": delta_best,
            "final_generation": n_gens - 1,
            "total_evaluations": n_gens * self.pop_size,
        }
        timings = {"evolution": t_evo}
        return {"history": history, "summary": summary, "timings": timings}


def _run_comparison(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    num_generations: int = NUM_GENERATIONS_LONG,
    seeds: Tuple[int, ...] = (42, 123, 7, 99, 0, 1, 2021, 2022, 2023, 2024),
    use_evosax_ops: bool = False,
) -> ComparisonResult:
    """Run both frameworks via BenchmarkRunner and return a ComparisonResult.

    This is the reusable core for Groups 2 & 5.  MalthusJAX fitness values
    are **not** negated here because ``BBOBConfig(maximize=False)`` already
    produces raw minimisation values — no sign-flip needed.
    """
    mjx_engine = MalthusJAXBenchEngine(
        pop_size=pop_size,
        dims=dims,
        problem=problem,
        num_generations=num_generations,
        use_evosax_ops=use_evosax_ops,
    )
    esx_engine = EvosaxBenchEngine(
        pop_size=pop_size,
        dims=dims,
        problem=problem,
        num_generations=num_generations,
    )

    mjx_runner = BenchmarkRunner(
        engine=mjx_engine,
        experiment_name=f"malthusjax_{problem}_p{pop_size}_d{dims}",
        write_artifacts=False,
    )
    esx_runner = BenchmarkRunner(
        engine=esx_engine,
        experiment_name=f"evosax_{problem}_p{pop_size}_d{dims}",
        write_artifacts=False,
    )

    mjx_result = mjx_runner.run(seeds=seeds)
    esx_result = esx_runner.run(seeds=seeds)

    return ComparisonResult(
        pipelines={"malthusjax": mjx_result, "evosax": esx_result},
        shared_config={
            "pop_size": pop_size,
            "dims": dims,
            "problem": problem,
            "num_generations": num_generations,
            "seeds": list(seeds),
        },
        # MalthusJAX with maximize=False already returns raw minimisation
        # values, so no sign flip is needed for this benchmark.
        negate_map={"malthusjax": False, "evosax": False},
    )


# ============================================================================
# BENCHMARK GROUP 1 — Single-Step Latency (warm dispatch)
# ============================================================================


class TestSingleStepLatency:
    """Warm single-step latency for both frameworks at different scales."""

    # --- MalthusJAX ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    @pytest.mark.parametrize("use_evosax_ops", [False, True])
    def test_malthusjax_step(
        self, benchmark, pop_size: int, dims: int, use_evosax_ops: bool
    ):
        """MalthusJAX: single jit-compiled step (warm)."""
        engine = _build_malthusjax_engine(pop_size, dims, use_evosax_ops=use_evosax_ops)
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = (
            "malthusjax_evosaxops" if use_evosax_ops else "malthusjax"
        )
        benchmark(_run)

    # --- Evosax ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_step(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: single jit-compiled step (warm)."""
        carry, jit_step = _evosax_init_and_warmup(pop_size, dims)

        def _run():
            c, _ = jit_step(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark(_run)


# ============================================================================
# BENCHMARK GROUP 2 — Multi-Generation Throughput (scan loop)
# ============================================================================


class TestMultiGenThroughput:
    """Full evolution loop: N generations via jax.lax.scan.

    Uses the ``Engine``-protocol adapters
    (:class:`MalthusJAXBenchEngine` / :class:`EvosaxBenchEngine`) so the
    benchmarked code path is identical to what :class:`BenchmarkRunner`
    executes.  pytest-benchmark still handles the wall-clock timing.
    """

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    @pytest.mark.parametrize("use_evosax_ops", [False, True])
    def test_malthusjax_scan(
        self, benchmark, pop_size: int, dims: int, use_evosax_ops: bool
    ):
        """MalthusJAX: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=NUM_GENERATIONS_SHORT,
            use_evosax_ops=use_evosax_ops,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{NUM_GENERATIONS_SHORT}gen/pop{pop_size}_d{dims}"
        benchmark.name = (
            "malthusjax_evosaxops" if use_evosax_ops else "malthusjax"
        )
        benchmark(_run)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_scan(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        bench_engine = EvosaxBenchEngine(
            pop_size=pop_size,
            dims=dims,
            num_generations=NUM_GENERATIONS_SHORT,
        )
        # Warm-up (compile)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"scan_{NUM_GENERATIONS_SHORT}gen/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark(_run)


# ============================================================================
# BENCHMARK GROUP 3 — JIT Compilation Time
# ============================================================================


class TestCompilationTime:
    """Measures cold JIT compilation time (first call overhead)."""

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_compile(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: time to JIT-compile the step function (cold)."""

        def _compile():
            # Clear JAX caches by creating a fresh engine each time
            engine = _build_malthusjax_engine(pop_size, dims)
            key = jr.PRNGKey(SEED)
            state = engine.init_state(key)
            jit_step = jax.jit(engine.step)
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"compile/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax"
        # Use 1 round for compilation benchmarks — they're expensive
        benchmark.pedantic(_compile, rounds=3, warmup_rounds=0)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_compile(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: time to JIT-compile the step function (cold)."""

        def _compile():
            strategy, params, es_problem, carry = _build_evosax_ga(pop_size, dims)
            step = _evosax_step_fn(strategy, params, es_problem)
            jit_step = jax.jit(step)
            c, _ = jit_step(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

        benchmark.group = f"compile/pop{pop_size}_d{dims}"
        benchmark.name = "evosax"
        benchmark.pedantic(_compile, rounds=3, warmup_rounds=0)


# ============================================================================
# BENCHMARK GROUP 4 — Operator-Level Microbenchmarks
# ============================================================================


class TestOperatorMicrobenchmarks:
    """Isolated operator-level benchmarks for MalthusJAX components.

    These have NO evosax counterpart — they serve as regression baselines
    for the operator-level fixes in Phases 3–5.
    """

    # --- Selection ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    def test_elite_pool_selection(self, benchmark, pop_size: int):
        """ElitePoolSelection warm dispatch."""
        elite_k = max(1, pop_size // 2)
        sel = ElitePoolSelection(num_selections=pop_size, elite_k=elite_k)

        fitness = jax.random.uniform(jr.PRNGKey(0), (pop_size,))
        key = jr.PRNGKey(1)

        jit_sel = jax.jit(sel)
        # Warm up — __call__ returns (parent_idx, elite_idx) tuple
        _parent, _elite = jit_sel(key, fitness)
        _parent.block_until_ready()

        def _run():
            parent_idx, elite_idx = jit_sel(key, fitness)
            parent_idx.block_until_ready()

        benchmark.group = f"operator_selection/pop{pop_size}"
        benchmark.name = "elite_pool"
        benchmark(_run)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    def test_tournament_selection(self, benchmark, pop_size: int):
        """TournamentSelection warm dispatch."""
        sel = TournamentSelection(num_selections=pop_size, tournament_size=3)

        fitness = jax.random.uniform(jr.PRNGKey(0), (pop_size,))
        key = jr.PRNGKey(1)

        jit_sel = jax.jit(sel)
        _parent, _elite = jit_sel(key, fitness)
        _parent.block_until_ready()

        def _run():
            parent_idx, elite_idx = jit_sel(key, fitness)
            parent_idx.block_until_ready()

        benchmark.group = f"operator_selection/pop{pop_size}"
        benchmark.name = "tournament"
        benchmark(_run)

    # --- Crossover ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_uniform_crossover(self, benchmark, pop_size: int, dims: int):
        """UniformCrossover: full __call__ including vmap + transpose."""
        genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))
        num_pairs = pop_size // 2

        crossover = UniformCrossover(
            num_offspring=2, crossover_rate=0.5, input_length=num_pairs
        )

        key = jr.PRNGKey(SEED)
        p1_pop = RealPopulation.init_random(jr.PRNGKey(0), genome_config, size=num_pairs)
        p2_pop = RealPopulation.init_random(jr.PRNGKey(1), genome_config, size=num_pairs)

        num_keys = crossover.num_keys(input_shape=(num_pairs,))
        keys = jax.random.split(key, num_keys)

        jit_cross = jax.jit(crossover)
        # Warm up
        _out = jit_cross(keys, p1_pop, p2_pop, genome_config)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), _out)

        def _run():
            out = jit_cross(keys, p1_pop, p2_pop, genome_config)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

        benchmark.group = f"operator_crossover/pop{pop_size}_d{dims}"
        benchmark.name = "uniform_crossover"
        benchmark(_run)

    # --- Mutation ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_gaussian_mutation(self, benchmark, pop_size: int, dims: int):
        """GaussianMutation: full __call__ including vmap + flatten."""
        genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))

        mutation = GaussianMutation(
            num_offspring=1,
            mutation_rate=0.1,
            mutation_strength=0.1,
            input_length=pop_size,
        )

        key = jr.PRNGKey(SEED)
        pop = RealPopulation.init_random(jr.PRNGKey(0), genome_config, size=pop_size)

        num_keys = mutation.num_keys(input_shape=(pop_size,))
        keys = jax.random.split(key, num_keys)

        jit_mut = jax.jit(mutation)
        _out = jit_mut(keys, pop, genome_config)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), _out)

        def _run():
            out = jit_mut(keys, pop, genome_config)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

        benchmark.group = f"operator_mutation/pop{pop_size}_d{dims}"
        benchmark.name = "gaussian"
        benchmark(_run)

    # --- Fitness Evaluation ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_bbob_fitness_eval(self, benchmark, pop_size: int, dims: int):
        """BBOB fitness evaluation via MalthusJAX evaluator."""
        bbob_config = BBOBConfig(fn_name="sphere", num_dims=dims, seed=SEED, maximize=False)
        evaluator = BBOBEvaluator.create(bbob_config)
        genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))
        pop = RealPopulation.init_random(jr.PRNGKey(0), genome_config, size=pop_size)

        jit_eval = jax.jit(evaluator.evaluate_population)
        _out = jit_eval(pop)
        _out.fitness.block_until_ready()

        def _run():
            out = jit_eval(pop)
            out.fitness.block_until_ready()

        benchmark.group = f"operator_fitness/pop{pop_size}_d{dims}"
        benchmark.name = "bbob_sphere"
        benchmark(_run)


# ============================================================================
# BENCHMARK GROUP 5 — Fitness Quality (Convergence Parity)
# ============================================================================


class TestConvergenceParity:
    """Verify both frameworks converge to comparable fitness on BBOB problems.

    Uses :class:`BenchmarkRunner` and :class:`ComparisonResult` from the
    ``malthusjax.benchmarking`` infrastructure for structured multi-seed
    execution and sign-normalised comparison.

    This is NOT a speed benchmark — it's a correctness-adjacent snapshot
    ensuring MalthusJAX produces finite, reasonable fitness values and
    that the reporting pipeline works end-to-end.
    """

    @pytest.mark.parametrize("problem", PROBLEMS)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_fitness_parity(self, problem: str, dims: int):
        """Compare final best fitness after fixed generation budget (10 seeds)."""
        pop_size = 200
        num_gens = NUM_GENERATIONS_LONG
        seeds = (42, 123, 7, 99, 0, 1, 2021, 2022, 2023, 2024)

        comparison = _run_comparison(
            pop_size=pop_size,
            dims=dims,
            problem=problem,
            num_generations=num_gens,
            seeds=seeds,
        )

        # ---- Validate ComparisonResult structure ----
        assert set(comparison.names) == {"malthusjax", "evosax"}

        for name in comparison.names:
            exp = comparison.pipelines[name]
            assert len(exp.runs) == len(seeds), (
                f"{name}: expected {len(seeds)} runs, got {len(exp.runs)}"
            )
            for run in exp.runs:
                assert run.status == "success", (
                    f"{name} seed={run.seed} failed: {run.error}"
                )
                # each run summary should now include start/end/delta metrics
                # metrics dictionary holds the numeric summary fields
                assert "start_best_fitness" in run.metrics
                assert "end_best_fitness" in run.metrics
                assert "delta_best" in run.metrics
                # minimization problems should show improvement (start>end)
                # delta may be zero when the initial population already contains
                # the best value; we only fail if the metric is negative.
                assert run.metrics["delta_best"] >= 0, (
                    f"non‑improving run {run.seed}: delta={run.metrics['delta_best']}"
                )

        # ---- Aggregated summary via ComparisonResult ----
        table = comparison.summary_table()
        mjx_best = table["malthusjax"]["best_fitness"]
        esx_best = table["evosax"]["best_fitness"]
        mjx_start = table["malthusjax"].get("start_best_fitness")
        mjx_end = table["malthusjax"].get("end_best_fitness")
        mjx_delta = table["malthusjax"].get("delta_best")
        esx_start = table["evosax"].get("start_best_fitness")
        esx_end = table["evosax"].get("end_best_fitness")
        esx_delta = table["evosax"].get("delta_best")

        print(
            f"\n  [{problem} d={dims}]  (mean over {len(seeds)} seeds)"
            f"\n    MalthusJAX best_fitness = {mjx_best:.6f}, start={mjx_start}, end={mjx_end}, Δ={mjx_delta}"
            f"\n    Evosax     best_fitness = {esx_best:.6f}, start={esx_start}, end={esx_end}, Δ={esx_delta}"
        )

        # also log metrics from each run for debugging
        for name in comparison.names:
            print(f"\n  {name} per-run metrics:")
            for run in comparison.pipelines[name].runs:
                print(f"    seed={run.seed} metrics={run.metrics}")

        # ---- Convergence history from first seed ----
        conv = comparison.convergence_data(seed_index=0)
        for name in comparison.names:
            assert len(conv[name]) > 0, f"No history for {name}"
            assert "best_fitness" in conv[name][0], f"Missing best_fitness in {name} history"

        # ---- Phase 0: sanity-only assertions (finite values) ----
        assert jnp.isfinite(mjx_best), (
            f"MalthusJAX returned non-finite mean best_fitness: {mjx_best}"
        )
        assert jnp.isfinite(esx_best), (
            f"Evosax returned non-finite mean best_fitness: {esx_best}"
        )


# ============================================================================
# BENCHMARK GROUP 6 — Unroll Factor Sweep
# ============================================================================


class TestUnrollSweep:
    """Measure how lax.scan unroll_num affects per-generation throughput.

    Higher unroll values allow XLA to fuse more steps into one HLO program,
    reducing dispatch overhead at the cost of compile time and peak memory.
    Run at fixed pop=100, d=10 to isolate the unroll effect cleanly.

    Benchmark: N-generation scan wall-clock time at each unroll factor.
    """

    # Fixed config so results are directly comparable
    _POP = 100
    _DIMS = 10
    _GENS = 50  # Short enough to keep memory per unroll level reasonable

    @pytest.mark.parametrize("unroll", UNROLL_FACTORS)
    def test_malthusjax_unroll_scan(self, benchmark, unroll: int):
        """MalthusJAX: {_GENS}-gen scan at unroll={unroll}."""
        bench_engine = MalthusJAXBenchEngine(
            pop_size=self._POP,
            dims=self._DIMS,
            num_generations=self._GENS,
            unroll_num=unroll,
        )
        # Warm-up (compile with this unroll factor)
        bench_engine.run_once(jr.PRNGKey(0))

        def _run():
            result = bench_engine.run_once(jr.PRNGKey(SEED))
            assert result["summary"]["best_fitness"] is not None

        benchmark.group = f"unroll_scan_{self._GENS}gen/pop{self._POP}_d{self._DIMS}"
        benchmark.name = f"unroll_{unroll}"
        benchmark(_run)


# ============================================================================
# BENCHMARK GROUP 7 — Step Phase Breakdown (MalthusJAX only)
# ============================================================================


class TestStepPhaseBreakdown:
    """Isolate and benchmark each phase of GeneticEngine.step() independently.

    Explains the single-step latency gap vs evosax by attributing cost to:
      Phase 0 — PRNG allocation (4-way jr.split)
      Phase 1 — Selection  (top_k for elites + selection operator)
      Phase 2 — Reproduction (crossover vmap + mutation vmap)
      Phase 3a — Merge (dynamic_update_slice buffer reuse)
      Phase 3b — Evaluate (BBOB fitness vmap)

    All phases are JIT-compiled individually over a warm state so XLA
    cost models are representative.  pop=500, d=10 to match the config
    where the single-step gap is most visible.
    """

    _POP = 500
    _DIMS = 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        engine = _build_malthusjax_engine(self._POP, self._DIMS)
        key = jr.PRNGKey(SEED)
        state, jit_step = _malthusjax_init_and_warmup(engine)
        self.engine = engine
        self.state = state
        self.jit_step = jit_step

    def test_full_step(self, benchmark):
        """Full engine step — baseline for phase sum."""
        state = self.state

        def _run():
            s, _ = self.jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "00_full_step"
        benchmark(_run)

    def test_phase0_entropy(self, benchmark):
        """Phase 0: 4-way PRNG split (_allocate_entropy)."""
        state = self.state
        jit_entropy = jax.jit(self.engine._allocate_entropy)
        # Warm up
        out = jit_entropy(state)
        out[0].block_until_ready()

        def _run():
            keys = jit_entropy(state)
            keys[0].block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "01_entropy"
        benchmark(_run)

    def test_phase1_selection(self, benchmark):
        """Phase 1: elite top_k + selection operator."""
        state = self.state
        engine = self.engine

        k_sel, _, _, _ = engine._allocate_entropy(state)

        jit_sel = jax.jit(
            lambda k, pop, ops, params: engine._selection_phase(k, pop, ops, params)
        )
        # Warm up
        _, idx = jit_sel(k_sel, state.population, state.operators, engine.engine_params)
        idx.block_until_ready()

        def _run():
            _, idx = jit_sel(k_sel, state.population, state.operators, engine.engine_params)
            idx.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "02_selection"
        benchmark(_run)

    def test_phase2_reproduction(self, benchmark):
        """Phase 2: crossover vmap + mutation vmap."""
        state = self.state
        engine = self.engine

        k_sel, k_cross, k_mut, _ = engine._allocate_entropy(state)
        _, parent_indices = engine._selection_phase(
            k_sel, state.population, state.operators, engine.engine_params
        )

        jit_repro = jax.jit(
            lambda kc, km, pidx, pop, ops, rmap: engine._reproduction_phase(
                kc, km, pidx, pop, ops, rmap
            )
        )
        out_pop = jit_repro(
            k_cross, k_mut, parent_indices,
            state.population, state.operators, state.resource_map,
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out_pop)

        def _run():
            pop = jit_repro(
                k_cross, k_mut, parent_indices,
                state.population, state.operators, state.resource_map,
            )
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), pop)

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "03_reproduction"
        benchmark(_run)

    def test_phase3b_evaluate(self, benchmark):
        """Phase 3b: fitness evaluation only."""
        state = self.state
        engine = self.engine

        jit_eval = jax.jit(
            lambda genes, s: engine._evaluate(genes, s)
        )
        out = jit_eval(state.population.genes, state)
        out.fitness.block_until_ready()

        def _run():
            pop = jit_eval(state.population.genes, state)
            pop.fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "04_evaluate"
        benchmark(_run)

    def test_track_best_full_vs_light(self, benchmark):
        """Compare step latency: TrackBest.LIGHT (default) vs FULL (argmax+gather+where)."""
        engine_full = _build_malthusjax_engine(
            self._POP, self._DIMS, track_best=TrackBest.FULL
        )
        state_full, jit_full = _malthusjax_init_and_warmup(engine_full)

        engine_light = _build_malthusjax_engine(
            self._POP, self._DIMS, track_best=TrackBest.LIGHT
        )
        state_light, jit_light = _malthusjax_init_and_warmup(engine_light)

        # Benchmark FULL
        def _run_full():
            s, _ = jit_full(state_full)
            s.best_fitness.block_until_ready()

        # Benchmark LIGHT
        def _run_light():
            s, _ = jit_light(state_light)
            s.best_fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "05_trackbest_light"
        benchmark(_run_light)

    def test_elite_topk_vs_argpartition(self, benchmark):
        """Direct comparison: jax.lax.top_k vs jnp.argpartition for elite selection.

        Isolates just the index-extraction kernel at pop=500, elite_k=250.
        Run this to confirm argpartition wins over top_k for elite extraction.
        """
        import jax.lax as lax

        fitness = jax.random.uniform(jr.PRNGKey(0), (self._POP,))
        elite_k = self._POP // 2

        jit_argpart = jax.jit(lambda f: jnp.argpartition(-f, elite_k)[:elite_k])
        jit_argpart(fitness).block_until_ready()  # warm up

        def _run():
            idx = jit_argpart(fitness)
            idx.block_until_ready()

        benchmark.group = f"elite_extraction/pop{self._POP}"
        benchmark.name = "argpartition"
        benchmark(_run)

    def test_elite_topk(self, benchmark):
        """Elite extraction baseline: jax.lax.top_k (O(N log N) full sort)."""
        import jax.lax as lax

        fitness = jax.random.uniform(jr.PRNGKey(0), (self._POP,))
        elite_k = self._POP // 2

        jit_topk = jax.jit(lambda f: lax.top_k(f, elite_k)[1])
        jit_topk(fitness).block_until_ready()  # warm up

        def _run():
            idx = jit_topk(fitness)
            idx.block_until_ready()

        benchmark.group = f"elite_extraction/pop{self._POP}"
        benchmark.name = "top_k"
        benchmark(_run)


# ============================================================================
# BENCHMARK GROUP 8 — Scaling Sweep
# ============================================================================

size_sweep_pop_sizes = [50, 100, 200, 500, 1000]

class TestScalingSweep:
    """Measure how throughput scales with population size.

    Useful for detecting O(N²) regressions introduced by fixes.
    """

    @pytest.mark.parametrize("pop_size", size_sweep_pop_sizes)
    def test_malthusjax_scaling(self, benchmark, pop_size: int):
        """MalthusJAX step latency scaling with population size (d=10)."""
        dims = 10
        engine = _build_malthusjax_engine(pop_size, dims)
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = "scaling_d10"
        benchmark.name = f"malthusjax_pop{pop_size}"
        benchmark(_run)

    @pytest.mark.parametrize("pop_size", size_sweep_pop_sizes)
    def test_evosax_scaling(self, benchmark, pop_size: int):
        """Evosax step latency scaling with population size (d=10)."""
        dims = 10
        carry, jit_step = _evosax_init_and_warmup(pop_size, dims)

        def _run():
            c, _ = jit_step(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

        benchmark.group = "scaling_d10"
        benchmark.name = f"evosax_pop{pop_size}"
        benchmark(_run)
        