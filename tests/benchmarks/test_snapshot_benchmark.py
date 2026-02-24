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
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from evosax.algorithms.population_based import SimpleGA
from evosax.problems import BBOBProblem

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.engine.base import _get_evolution_kernel
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
)
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.tournament import TournamentSelection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
PROBLEMS = ["sphere", "rastrigin"]
DIMENSIONS = [10, 50]
POP_SIZES = [100, 500]
NUM_GENERATIONS_SHORT = 50
NUM_GENERATIONS_LONG = 500


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
) -> GeneticEngine:
    """Build a ready-to-use MalthusJAX GeneticEngine."""
    genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))

    bbob_config = BBOBConfig(fn_name=problem, num_dims=dims, seed=SEED, maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)

    elite_count = max(1, int(pop_size * elite_ratio))

    if selection_type == "tournament":
        selection = TournamentSelection(num_selections=pop_size, tournament_size=3)
    else:
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=elite_count)

    crossover = UniformCrossover(num_offspring=2, crossover_rate=0.5)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1)

    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=num_generations,
        elitism=elite_count,
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


# ============================================================================
# BENCHMARK GROUP 1 — Single-Step Latency (warm dispatch)
# ============================================================================


class TestSingleStepLatency:
    """Warm single-step latency for both frameworks at different scales."""

    # --- MalthusJAX ---

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_step(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: single jit-compiled step (warm)."""
        engine = _build_malthusjax_engine(pop_size, dims)
        state, jit_step = _malthusjax_init_and_warmup(engine)

        def _run():
            s, _ = jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"single_step/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax"
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
    """Full evolution loop: N generations via jax.lax.scan."""

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_malthusjax_scan(self, benchmark, pop_size: int, dims: int):
        """MalthusJAX: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        engine = _build_malthusjax_engine(
            pop_size, dims, num_generations=NUM_GENERATIONS_SHORT
        )
        key = jr.PRNGKey(SEED)
        state = engine.init_state(key)

        # Warm up the scan kernel (compile once)
        final, _, _ = engine.run(state, compile=True)
        final.best_fitness.block_until_ready()

        def _run():
            # Re-init state each iteration (engine.run consumes/donates buffers)
            s = engine.init_state(key)
            f, _, _ = engine.run(s, compile=True)
            f.best_fitness.block_until_ready()

        benchmark.group = f"scan_{NUM_GENERATIONS_SHORT}gen/pop{pop_size}_d{dims}"
        benchmark.name = "malthusjax"
        benchmark(_run)

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_evosax_scan(self, benchmark, pop_size: int, dims: int):
        """Evosax SimpleGA: full scan loop for {NUM_GENERATIONS_SHORT} gens."""
        strategy, params, es_problem, carry = _build_evosax_ga(pop_size, dims)
        step = _evosax_step_fn(strategy, params, es_problem)

        def scan_loop(carry):
            return jax.lax.scan(step, carry, None, length=NUM_GENERATIONS_SHORT)

        jit_scan = jax.jit(scan_loop)

        # Warm up
        _wc, _ = jit_scan(carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), _wc)

        def _run():
            c, _ = jit_scan(carry)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), c)

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
        # Warm up
        _idx = jit_sel(key, fitness)
        _idx.block_until_ready()

        def _run():
            idx = jit_sel(key, fitness)
            idx.block_until_ready()

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
        _idx = jit_sel(key, fitness)
        _idx.block_until_ready()

        def _run():
            idx = jit_sel(key, fitness)
            idx.block_until_ready()

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

    This is NOT a speed benchmark — it's a correctness-adjacent snapshot ensuring
    that MalthusJAX achieves competitive final fitness vs evosax on the same problem.
    Runs once per (problem, dims) pair with a fixed budget of generations.
    """

    @pytest.mark.parametrize("problem", PROBLEMS)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_fitness_parity(self, problem: str, dims: int):
        """Compare final best fitness after fixed generation budget."""
        pop_size = 200
        num_gens = NUM_GENERATIONS_LONG

        # --- MalthusJAX ---
        engine = _build_malthusjax_engine(
            pop_size, dims, problem=problem, num_generations=num_gens
        )
        key = jr.PRNGKey(SEED)
        state = engine.init_state(key)
        final_mjx, _, _ = engine.run(state, compile=True)
        mjx_fitness = float(final_mjx.best_fitness)
        final_mjx.best_fitness.block_until_ready()

        # --- Evosax ---
        strategy, params, es_problem, carry = _build_evosax_ga(pop_size, dims, problem)
        step = _evosax_step_fn(strategy, params, es_problem)
        jit_scan = jax.jit(lambda c: jax.lax.scan(step, c, None, length=num_gens))
        final_carry, _ = jit_scan(carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_carry)
        es_state, _, _ = final_carry
        esx_fitness = float(es_state.best_fitness)

        # MalthusJAX uses maximize=False for BBOB, so best_fitness is the raw
        # minimization value (lower is better). Evosax also minimizes.
        print(
            f"\n  [{problem} d={dims}] "
            f"MalthusJAX={mjx_fitness:.6f}  Evosax={esx_fitness:.6f}"
        )

        # Phase 0 baseline: record values, sanity-check only (not NaN/Inf).
        # Strict convergence assertions will be added in later phases after
        # the engine fixes are applied.
        assert jnp.isfinite(mjx_fitness), f"MalthusJAX returned non-finite fitness: {mjx_fitness}"
        assert jnp.isfinite(esx_fitness), f"Evosax returned non-finite fitness: {esx_fitness}"


# ============================================================================
# BENCHMARK GROUP 6 — Scaling Sweep
# ============================================================================


class TestScalingSweep:
    """Measure how throughput scales with population size.

    Useful for detecting O(N²) regressions introduced by fixes.
    """

    @pytest.mark.parametrize("pop_size", [50, 100, 200, 500, 1000])
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

    @pytest.mark.parametrize("pop_size", [50, 100, 200, 500, 1000])
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
