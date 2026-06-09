"""
Shared fixtures, constants, and helper utilities for benchmark groups.
This module is imported by each individual benchmark file that replaces the
monolithic ``test_snapshot_benchmark.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from evosax.algorithms import SimpleGA
from evosax.problems import BBOBProblem

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.core.fitness.bbob_evaluator import (
    BBOBConfig,
    BBOBEvaluator,
)
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
)
from malthusjax.engine.resource_mapper import KeyDerivationStrategy
from malthusjax.engine.schedules import TrackBest

# wrappers around native evosax implementations; used to isolate
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.crossover.real import (
    BinomialCrossover,
    BinomialCrossover_injection,
    BlendCrossover,
    BlendCrossover_injection,
    SimulatedBinaryCrossover,
    SimulatedBinaryCrossover_injection,
    UniformCrossover,
    UniformCrossover_injection,
)
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.operators.mutation.real import (
    BallMutation,
    BallMutation_injection,
    GaussianMutation,
    GaussianMutation_injection,
    PolynomialMutation,
    PolynomialMutation_injection,
)
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.roulette import RouletteSelection
from malthusjax.operators.selection.tournament import TournamentSelection

# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------

SEED = 42
# subset of BBOB functions for benchmarking
PROBLEMS = [
    "sphere",
    "rastrigin",
    "ellipsoidal_rotated",
    "schwefel",
    "griewank_rosenbrock",
]
DIMENSIONS = [10, 50]
POP_SIZES = [100, 500, 1024, 1025]  # powers of two and just above
NUM_GENERATIONS_SHORT = 50
NUM_GENERATIONS_LONG = 500
UNROLL_FACTORS = [1, 5, 10, 25]  # lax.scan unroll sweep
_INJECTION_CROSSOVER_TYPES = ["uniform", "blend", "sbx", "binomial"]
_INJECTION_MUTATION_TYPES = ["gaussian", "ball", "polynomial"]
size_sweep_pop_sizes = [50, 100, 200, 500, 1000]

# Default root for fitness-parity artifacts.
# Override via the ``MALTHUSJAX_PARITY_RESULTS`` env-var.
_PARITY_RESULTS_DIR = Path(
    __import__("os").environ.get(
        "MALTHUSJAX_PARITY_RESULTS",
        str(Path(__file__).resolve().parents[2] / "results" / "fitness_parity"),
    )
)

# ---------------------------------------------------------------------------
# Helpers — MalthusJAX
# ---------------------------------------------------------------------------


@pytest.fixture(params=size_sweep_pop_sizes)
def pop_size(request: Any) -> int:
    """Parameterized fixture for population size sweeps."""
    return request.param


def _build_crossover(crossover_type: str, use_injection: bool):
    """Factory for crossover operators, including injection-mode variants.

    Args:
        crossover_type: One of "uniform", "blend", "sbx", "binomial".
        use_injection: When True, returns the injection-mode (_injection) variant
            that materialises the full noise tensor from a single PRNG key.
    """
    if crossover_type == "uniform":
        cls_std = UniformCrossover
        cls_inj = UniformCrossover_injection
        kwargs = dict(num_offspring=1, crossover_rate=0.5)
    elif crossover_type == "blend":
        cls_std = BlendCrossover
        cls_inj = BlendCrossover_injection
        kwargs = dict(num_offspring=1, crossover_rate=0.9, alpha=0.5)
    elif crossover_type == "sbx":
        cls_std = SimulatedBinaryCrossover
        cls_inj = SimulatedBinaryCrossover_injection
        kwargs = dict(num_offspring=2, crossover_rate=0.9, eta=20.0)
    elif crossover_type == "binomial":
        cls_std = BinomialCrossover
        cls_inj = BinomialCrossover_injection
        kwargs = dict(num_offspring=1, crossover_rate=0.9)
    else:
        raise ValueError(f"Unknown crossover_type: {crossover_type!r}")
    return (cls_inj if use_injection else cls_std)(**kwargs)


def _build_mutation(mutation_type: str, use_injection: bool):
    """Factory for mutation operators, including injection-mode variants.

    Args:
        mutation_type: One of "gaussian", "ball", "polynomial".
        use_injection: When True, returns the injection-mode (_injection) variant.
    """
    if mutation_type == "gaussian":
        cls_std = GaussianMutation
        cls_inj = GaussianMutation_injection
        kwargs = dict(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1)
    elif mutation_type == "ball":
        cls_std = BallMutation
        cls_inj = BallMutation_injection
        kwargs = dict(num_offspring=1, radius=0.1, mutation_rate=1.0)
    elif mutation_type == "polynomial":
        cls_std = PolynomialMutation
        cls_inj = PolynomialMutation_injection
        kwargs = dict(num_offspring=1, mutation_rate=0.1, eta=20.0)
    else:
        raise ValueError(f"Unknown mutation_type: {mutation_type!r}")
    return (cls_inj if use_injection else cls_std)(**kwargs)


def _build_malthusjax_engine(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    num_generations: int = 1,
    elite_ratio: float = 0.5,
    selection_type: str = "elite_pool",
    unroll_num: int = 1,
    track_best: TrackBest = TrackBest.NONE,
    use_evosax_ops: bool = False,
    crossover_type: str = "uniform",
    mutation_type: str = "gaussian",
    use_injection_ops: bool = False,
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT,
) -> GeneticEngine:
    """Build a ready-to-use MalthusJAX GeneticEngine.

    Parameters
    ----------
    use_evosax_ops:
        When True the engine employs ``EvosaxUniformCrossoverWrapper`` and
        ``EvosaxGaussianWrapper`` — overrides ``crossover_type`` /
        ``mutation_type`` / ``use_injection_ops``.
    crossover_type:
        One of ``"uniform"``, ``"blend"``, ``"sbx"``, ``"binomial"``.
    mutation_type:
        One of ``"gaussian"``, ``"ball"``, ``"polynomial"``.
    use_injection_ops:
        When True, use injection-mode operator variants that materialise the
        full noise tensor from a single PRNG key before applying it pair-wise.
    key_derivation:
        ``KeyDerivationStrategy.SPLIT`` (default, uncorrelated sub-keys via
        ``jax.random.split``) or ``KeyDerivationStrategy.FOLD`` (parallel
        sub-keys via ``jax.random.fold_in``).
    """
    genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))

    bbob_config = BBOBConfig(fn_name=problem, num_dims=dims, seed=SEED, maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)

    elite_count = max(1, int(pop_size * elite_ratio))

    if selection_type == "tournament":
        selection = TournamentSelection(num_selections=pop_size, tournament_size=3)
    elif selection_type == "roulette":
        selection = RouletteSelection(num_selections=pop_size)
    else:
        # Use ElitePoolSelection for structurally consistent hardware parity
        # testing against Evosax, saving us the JAX argpartition cost!
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=elite_count)

    if use_evosax_ops:
        crossover = EvosaxUniformCrossoverWrapper(num_offspring=1, crossover_rate=0.5)
        mutation = EvosaxGaussianWrapper(num_offspring=1, mutation_strength=0.1)
    else:
        crossover = _build_crossover(crossover_type, use_injection_ops)
        mutation = _build_mutation(mutation_type, use_injection_ops)

    # Fair Benchmark Hardware Parity:
    # Evosax's SimpleGA selects parents from the elite pool, but mutates and
    # evaluates *all* generated offspring. MalthusJAX's EngineParams normally
    # preserves `elitism` exact copies, skipping those crossover/mutation ops.
    # We must enforce `elitism=0` here so that both engines execute the exact
    # same tensor manipulation loads per generation.
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=num_generations,
        elitism=0,
        unroll_num=unroll_num,
        track_best=track_best,
        key_derivation=key_derivation,
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
# Helpers — Canonical (shared) population initialisation
# ---------------------------------------------------------------------------


def _canonical_population(
    key: jax.Array,
    pop_size: int,
    dims: int,
    bounds: Tuple[float, float] = (-5.0, 5.0),
) -> jax.Array:
    """Generate a deterministic starting population from a PRNG key.

    Both MalthusJAX and evosax engines call this same function so that,
    for a given seed, every configuration begins from *identical* initial
    genes.  The output is a plain ``(pop_size, dims)`` float32 array.
    """
    return jax.random.uniform(
        key,
        (pop_size, dims),
        minval=bounds[0],
        maxval=bounds[1],
    )


# ---------------------------------------------------------------------------
# Helpers — Evosax
# ---------------------------------------------------------------------------


def _build_evosax_ga(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    elite_ratio: float = 0.5,
    rng: Optional[jax.Array] = None,
    init_x: Optional[jax.Array] = None,
) -> Tuple[SimpleGA, Any, Any, Any]:
    """Build evosax SimpleGA + BBOB problem, return (strategy, params, problem, init_carry).

    Uses evosax 0.2.0 with ask/tell interface.

    Parameters
    ----------
    rng : optional
        Root PRNG key.  Defaults to ``jr.PRNGKey(SEED)`` for backward
        compatibility with the speed-benchmark groups.
    init_x : optional
        Pre-generated initial population ``(pop_size, dims)``.
        When given, this array is used instead of sampling a new one.
    """
    if rng is None:
        rng = jr.PRNGKey(SEED)

    # evosax 0.2.0 uses lowercase function names
    es_problem = BBOBProblem(problem.lower(), num_dims=dims)

    # Initialize strategy with sample solution
    init_solution = es_problem.sample(rng)
    strategy = SimpleGA(population_size=pop_size, solution=init_solution)
    es_params = strategy.default_params

    # Build initial carry = (es_state, problem_state, rng)
    r_init, r_start = jax.random.split(rng)
    p_state = es_problem.init(r_init)

    if init_x is None:
        init_x = jax.random.uniform(r_init, (pop_size, dims), minval=-5.0, maxval=5.0)
    init_fit = jnp.full((pop_size,), jnp.inf)
    es_state = strategy.init(r_init, init_x, init_fit, es_params)

    carry = (es_state, p_state, r_start)
    return strategy, es_params, es_problem, carry


def _evosax_step_fn(strategy: SimpleGA, params: Any, problem: Any) -> Callable:
    """Return a scan-compatible step function for evosax ask/tell.

    Uses evosax 0.2.0 ask/tell interface.
    """

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
    def _get_engine(self) -> GeneticEngine:
        if not hasattr(self, "_engine") or self._engine is None:
            self._engine = _build_malthusjax_engine(
                self.pop_size,
                self.dims,
                problem=self.problem,
                num_generations=self.num_generations,
                elite_ratio=self.elite_ratio,
                selection_type=self.selection_type,
                unroll_num=self.unroll_num,
                use_evosax_ops=self.use_evosax_ops,
                crossover_type=self.crossover_type,
                mutation_type=self.mutation_type,
                use_injection_ops=self.use_injection_ops,
                key_derivation=self.key_derivation,
            )
        return self._engine

    def init_state(self, key: jax.Array):
        """Expose GeneticEngine API for benchmark warmup utilities."""
        return self._get_engine().init_state(key)

    def step(self, state: GeneticEvolutionState):
        return self._get_engine().step(state)

    """Wraps a MalthusJAX GeneticEngine to satisfy the ``Engine`` protocol.

    ``run_once(key)`` returns the standard dict expected by
    :class:`BenchmarkRunner`: ``{history, summary, timings}``.

    The ``use_evosax_ops`` flag controls whether the engine is constructed
    with native MalthusJAX operators or the evosax wrappers.  This allows
    us to measure the overhead of the engine architecture itself.

    The ``crossover_type`` / ``mutation_type`` / ``use_injection_ops`` fields
    control which operator variant (standard or injection-mode) is used.
    ``key_derivation`` selects between ``SPLIT`` (sequential, uncorrelated) and
    ``FOLD`` (parallel, deterministic) entropy strategies.

    When ``canonical_init`` is True, the engine derives a *canonical* starting
    population from the per-seed key via :func:`_canonical_population` so that
    every configuration — including evosax — begins from identical gene
    values.  This makes fitness comparisons across operator variants and
    frameworks meaningful.
    """

    pop_size: int
    dims: int
    problem: str = "sphere"
    num_generations: int = NUM_GENERATIONS_LONG
    elite_ratio: float = 0.5
    selection_type: str = "elite_pool"
    unroll_num: int = 1
    use_evosax_ops: bool = False
    crossover_type: str = "uniform"
    mutation_type: str = "gaussian"
    use_injection_ops: bool = False
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT
    canonical_init: bool = False
    canonical_bounds: Tuple[float, float] = (-5.0, 5.0)

    # Internal cached engine for warmup/benchmark hooking
    _engine: Optional[GeneticEngine] = None
    _jit_scan: Optional[Callable] = None  # NEW: Cache the pure JIT scan

    def _setup(self):
        """Pre-compile and warm up the pure JAX scan loop."""
        if self._jit_scan is not None:
            return

        engine = self._get_engine()

        # 1. Create a pure JIT-compiled scan loop
        def scan_fn(state):
            def scan_body(c, _):
                new_c, full_hist = engine.step(c)
                # --- MANUALLY STRIP THE HISTORY ---
                # We extract ONLY the scalars the benchmark runner needs.
                # XLA's Dead Code Elimination will now optimize away the rest
                # of the heavy MalthusJAX history PyTree allocations!
                light_hist = {
                    "generation": full_hist.generation,
                    "best_fitness": full_hist.best_fitness,
                    "mean_fitness": full_hist.mean_fitness,
                }
                return new_c, light_hist
            return jax.lax.scan(scan_body, state, None, length=self.num_generations)

        self._jit_scan = jax.jit(scan_fn)

        # 2. WARMUP: Force XLA compilation before any timing starts
        dummy_state = engine.init_state(jr.PRNGKey(0))
        warmup_out = self._jit_scan(dummy_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_out)

    def run_once(self, key: jax.Array) -> Dict[str, Any]:
        self._setup()  # Ensure warmup happened
        engine = self._get_engine()

        if self.canonical_init:
            pop_key, evo_key = jr.split(key)
            shared_genes = _canonical_population(
                pop_key,
                self.pop_size,
                self.dims,
                self.canonical_bounds,
            )
            t0 = time.time()
            state = engine.init_state(evo_key)
            population_genes_cls = type(state.population.genes)
            state = state.replace(
                population=state.population.replace(
                    genes=population_genes_cls.from_tensor(shared_genes)
                )
            )
            t_init = time.time() - t0
        else:
            t0 = time.time()
            state = engine.init_state(key)
            t_init = time.time() - t0

        # --- THE FAIR TIMING BLOCK ---
        t0 = time.time()
        final_state, stripped_history = self._jit_scan(state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_state)
        # Ensure history is also ready before stopping the clock
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), stripped_history)
        t_evo = time.time() - t0

        # --- THE FIX: BULK TRANSFER TO CPU ---
        # Pull the entire dictionary of arrays across the PCIe bus exactly once
        local_history = jax.device_get(stripped_history)

        # Unpack the local CPU dictionary
        n_gens = int(local_history["generation"].shape[0])
        history: List[Dict[str, Any]] = []
        for g in range(n_gens):
            history.append(
                {
                    "generation": int(local_history["generation"][g]),
                    "best_fitness": float(local_history["best_fitness"][g]),
                    "mean_fitness": float(local_history["mean_fitness"][g]),
                }
            )

        start_best = history[0]["best_fitness"]
        end_best = history[-1]["best_fitness"]
        # Repository convention: lower is better (minimization).
        # Compute positive improvement as start - end so that improving runs
        # yield a non-negative ``delta_best`` value.
        delta_best = start_best - end_best

        summary = {
            "best_fitness": float(final_state.best_fitness),
            "start_best_fitness": start_best,
            "end_best_fitness": end_best,
            "delta_best": delta_best,
            "final_generation": n_gens - 1,
            "total_evaluations": n_gens * self.pop_size,
        }
        timings = {"warmup": t_init, "execution": t_evo, "total": t_init + t_evo}

        return {"history": history, "summary": summary, "timings": timings}


@dataclass
class EvosaxBenchEngine:
    """Wraps evosax SimpleGA to satisfy the ``Engine`` protocol.

    ``run_once(key)`` returns the standard dict expected by
    :class:`BenchmarkRunner`: ``{history, summary, timings}``.

    When ``canonical_init`` is True the engine derives the starting
    population via :func:`_canonical_population` using the same key-split
    convention as :class:`MalthusJAXBenchEngine`, guaranteeing identical
    initial genes across frameworks for each seed.
    """

    pop_size: int
    dims: int
    problem: str = "sphere"
    num_generations: int = NUM_GENERATIONS_LONG
    elite_ratio: float = 0.5
    canonical_init: bool = False
    canonical_bounds: Tuple[float, float] = (-5.0, 5.0)

    _strategy: Optional[Any] = None
    _params: Optional[Any] = None
    _es_problem: Optional[Any] = None
    _jit_scan: Optional[Callable] = None  # NEW: Cache the pure JIT scan

    def _setup(self):
        if self._strategy is not None:
            return

        strategy, params, es_problem, _ = _build_evosax_ga(
            self.pop_size,
            self.dims,
            self.problem,
            self.elite_ratio,
        )

        self._strategy = strategy
        self._params = params
        self._es_problem = es_problem

        def step_with_output(carry, _):
            es_state, p_state, rng = carry
            rng, rng_step = jax.random.split(rng)
            x, es_state = strategy.ask(rng_step, es_state, params)
            fitness, p_state, _ = es_problem.eval(rng_step, x, p_state)
            es_state, _ = strategy.tell(rng_step, x, fitness, es_state, params)

            # Evosax BBOB returns raw minimization scores (lower is better).
            # Keep the raw sign so all benchmark paths follow the repository
            # minimization convention consistently.
            metrics = {
                "generation": es_state.generation_counter,
                "best_fitness": es_state.best_fitness,
                "mean_fitness": jnp.mean(fitness),
            }
            return (es_state, p_state, rng), metrics

        self._jit_scan = jax.jit(
            lambda c: jax.lax.scan(step_with_output, c, None, length=self.num_generations)
        )

        # 2. WARMUP: Force XLA compilation before timing starts
        r_init, r_start = jax.random.split(jr.PRNGKey(0))
        p_state = self._es_problem.init(r_init)
        init_x = jax.random.uniform(r_init, (self.pop_size, self.dims), minval=-5.0, maxval=5.0)
        init_fit = jnp.full((self.pop_size,), jnp.inf)
        es_state = self._strategy.init(r_init, init_x, init_fit, self._params)

        dummy_carry = (es_state, p_state, r_start)
        warmup_out = self._jit_scan(dummy_carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_out)

    def run_once(self, key: jax.Array) -> Dict[str, Any]:
        self._setup()  # Ensure warmup happened

        t0 = time.time()
        r_init, r_start = jax.random.split(key)
        p_state = self._es_problem.init(r_init)

        if self.canonical_init:
            init_x = _canonical_population(
                r_init,
                self.pop_size,
                self.dims,
                self.canonical_bounds,
            )
        else:
            init_x = jax.random.uniform(
                r_init, (self.pop_size, self.dims), minval=-5.0, maxval=5.0
            )

        init_fit = jnp.full((self.pop_size,), jnp.inf)
        es_state = self._strategy.init(r_init, init_x, init_fit, self._params)
        carry = (es_state, p_state, r_start)
        t_init = time.time() - t0

        # --- THE FAIR TIMING BLOCK ---
        t0 = time.time()
        final_carry, stripped_history = self._jit_scan(carry)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_carry)
        # Ensure history is also ready before stopping the clock
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), stripped_history)
        t_evo = time.time() - t0

        final_es_state, _, _ = final_carry

        # --- THE FIX: BULK TRANSFER TO CPU ---
        # Pull the entire dictionary of arrays across the PCIe bus exactly once
        local_history = jax.device_get(stripped_history)

        n_gens = int(local_history["generation"].shape[0])
        history: List[Dict[str, Any]] = []
        for g in range(n_gens):
            history.append(
                {
                    "generation": int(local_history["generation"][g]),
                    "best_fitness": float(local_history["best_fitness"][g]),
                    "mean_fitness": float(local_history["mean_fitness"][g]),
                }
            )

        start_best = history[0]["best_fitness"]
        end_best = history[-1]["best_fitness"]
        # Repository convention: lower is better (minimization).
        # Compute positive improvement as start - end so that improving runs
        # yield a non-negative ``delta_best`` value.
        delta_best = start_best - end_best

        summary = {
            "best_fitness": float(final_es_state.best_fitness),
            "start_best_fitness": start_best,
            "end_best_fitness": end_best,
            "delta_best": delta_best,
            "final_generation": n_gens - 1,
            "total_evaluations": n_gens * self.pop_size,
        }
        timings = {"warmup": t_init, "execution": t_evo, "total": t_init + t_evo}

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
        negate_map={"malthusjax": False, "evosax": False},
    )


def _run_injection_experiment(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    num_generations: int = NUM_GENERATIONS_LONG,
    seeds: Tuple[int, ...] = (42, 123, 7),
    crossover_type: str = "uniform",
    mutation_type: str = "gaussian",
    use_injection_ops: bool = False,
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT,
    output_dir: Optional[Path] = None,
    canonical_init: bool = False,
) -> "ExperimentResult":
    """Run a single MalthusJAX configuration via BenchmarkRunner.

    Reusable core for Group 11 (injection + key-derivation parity tests).
    Returns an :class:`ExperimentResult` rather than a
    :class:`ComparisonResult` because these tests have no evosax counterpart.

    If *output_dir* is given the runner writes ``summary.json`` and
    ``histories_combined.csv`` to a sub-directory named after the
    experiment.

    When *canonical_init* is True, every seed starts from a canonical
    population generated by :func:`_canonical_population` so that
    cross-configuration comparisons are apples-to-apples.
    """
    suffix = (
        f"{'inj' if use_injection_ops else 'std'}"
        f"_{crossover_type}x_{mutation_type}m"
        f"_{key_derivation.value}"
    )
    experiment_name = f"mjx_{problem}_p{pop_size}_d{dims}_{suffix}"
    engine = MalthusJAXBenchEngine(
        pop_size=pop_size,
        dims=dims,
        problem=problem,
        num_generations=num_generations,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        use_injection_ops=use_injection_ops,
        key_derivation=key_derivation,
        canonical_init=canonical_init,
    )

    write = output_dir is not None
    exp_output_dir = (output_dir / experiment_name) if write else None

    runner = BenchmarkRunner(
        engine=engine,
        experiment_name=experiment_name,
        output_dir=exp_output_dir,
        write_artifacts=write,
    )
    return runner.run(seeds=seeds)


def _run_parity_comparison(
    pop_size: int,
    dims: int,
    problem: str = "sphere",
    num_generations: int = NUM_GENERATIONS_LONG,
    seeds: Tuple[int, ...] = (42, 123, 7),
    crossover_type: str = "uniform",
    mutation_type: str = "gaussian",
    use_injection_ops: bool = False,
    use_evosax_ops: bool = False,
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT,
    output_dir: Optional[Path] = None,
) -> ComparisonResult:
    """Run a MalthusJAX configuration **and** evosax from the same canonical
    starting population, returning a :class:`ComparisonResult`.

    Evosax acts as the *golden-standard* baseline.  Both engines derive an
    identical initial population from each seed key via
    :func:`_canonical_population`, ensuring that any fitness difference is
    attributable to the evolutionary operators — not random initialisation.

    If ``use_evosax_ops=True``, the MalthusJAX engine is constructed with the
    Evosax compatibility wrappers ``EvosaxUniformCrossoverWrapper`` and
    ``EvosaxGaussianWrapper`` for a closer operator-level parity comparison.

    If *output_dir* is given, artifacts for **both** pipelines are written.
    """
    suffix = (
        f"{'inj' if use_injection_ops else 'std'}"
        f"_{crossover_type}x_{mutation_type}m"
        f"_{key_derivation.value}"
    )
    mjx_name = f"mjx_{problem}_p{pop_size}_d{dims}_{suffix}"
    if use_evosax_ops:
        mjx_name += "_evosaxops"
    esx_name = f"evosax_{problem}_p{pop_size}_d{dims}"

    bounds = (-5.0, 5.0)  # match RealGenomeConfig default for BBOB benchmarks

    mjx_engine = MalthusJAXBenchEngine(
        pop_size=pop_size,
        dims=dims,
        problem=problem,
        num_generations=num_generations,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        use_injection_ops=use_injection_ops,
        use_evosax_ops=use_evosax_ops,
        key_derivation=key_derivation,
        canonical_init=True,
    )
    esx_engine = EvosaxBenchEngine(
        pop_size=pop_size,
        dims=dims,
        problem=problem,
        num_generations=num_generations,
        canonical_init=True,
        canonical_bounds=bounds,
    )

    write = output_dir is not None

    mjx_runner = BenchmarkRunner(
        engine=mjx_engine,
        experiment_name=mjx_name,
        output_dir=(output_dir / mjx_name) if write else None,
        write_artifacts=write,
    )
    esx_runner = BenchmarkRunner(
        engine=esx_engine,
        experiment_name=esx_name,
        output_dir=(output_dir / esx_name) if write else None,
        write_artifacts=write,
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
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "use_injection_ops": use_injection_ops,
            "use_evosax_ops": use_evosax_ops,
            "key_derivation": key_derivation.value,
            "seeds": list(seeds),
            "canonical_init": True,
        },
        negate_map={"malthusjax": False, "evosax": False},
    )
