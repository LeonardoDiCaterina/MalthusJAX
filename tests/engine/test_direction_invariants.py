"""Comprehensive test suite for optimization direction invariants across MalthusJAX.

Validates the canonical single-source-of-truth convention:
- All evaluators MUST return fitness in LOWER-IS-BETTER form.
- Evaluators with maximize=True internally negate (-score).
- Engines uniformly minimize (jnp.min, jnp.minimum, jnp.argmin) and never branch on maximize.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.core.fitness.binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackData,
    KnapsackEvaluator,
)
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import (
    BaseOptimizationEnvironment,
    BaseSupervisedEnvironment,
)
from malthusjax.core.fitness.composable.evaluators import (
    OptimizationEvaluator,
    SupervisedEvaluator,
    TensorNeatEvaluator,
)
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    TrackBest,
)
from malthusjax.operators.crossover.binary import UniformCrossover as BinaryUniformCrossover
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.tournament import TournamentSelection


# =============================================================================
# Helper Environments & Dummy Problems for Independent Oracles
# =============================================================================


@struct.dataclass
class SimpleSphereEnv(BaseOptimizationEnvironment):
    """Simple 2D sphere environment with an independent evaluate method."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        # Raw objective: f(x) = sum(x^2). Global minimum at 0.0.
        return jnp.sum(solution**2)


@struct.dataclass
class DummySupervisedEnv(BaseSupervisedEnvironment):
    """Dummy static supervised dataset with known linear relation."""

    X: chex.Array = struct.field(pytree_node=False, default=None)
    y: chex.Array = struct.field(pytree_node=False, default=None)


class DummyLinearInterpreter(IdentityInterpreter):
    def apply(self, genome: Any, inputs: Any = None) -> chex.Array:
        # Predict y_hat = inputs @ genome
        return inputs @ genome.values


class DummyTensorNeatProblem:
    """Mock problem for testing TensorNeatEvaluator direction and NaN sentinels."""

    def evaluate(self, state: Any, key: Any, forward_fn: Any, pop_member: Any) -> Tuple[chex.Array, chex.Array]:
        # pop_member is an array whose first element indicates score
        score = pop_member[0]
        # Return (score, empty_descriptors)
        return score, jnp.zeros(0)


class DummyTensorNeatTransform:
    def transform_population(self, genes: Any, state: Any = None) -> Any:
        # Pass through values directly
        return getattr(genes, "values", genes)


# =============================================================================
# Test 1: PROPERTY: Direction Invariance Across TrackBest Modes
# =============================================================================


@pytest.mark.parametrize(
    "eval_type",
    ["sphere", "binary_sum", "knapsack", "bbobax"],
)
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("track_best", [TrackBest.NONE, TrackBest.LIGHT])
def test_direction_invariance_across_track_best_modes(eval_type: str, maximize: bool, track_best: TrackBest):
    """Assert that final_state.best_genome corresponds to the individual with the

    best RAW objective according to declared maximize intent.
    Independent oracle: directly evaluates the final population using raw objective.
    """
    key = jax.random.PRNGKey(42)
    pop_size = 10
    num_generations = 3

    if eval_type == "sphere":
        genome_cfg = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0))
        env = SimpleSphereEnv()
        evaluator = OptimizationEvaluator(
            env=env,
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=maximize),
        )
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=1)
        crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
        mutation = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.1)

        def raw_oracle(genes: RealGenome) -> np.ndarray:
            # Independent raw objective evaluation (sum of squares)
            vals = np.asarray(genes.values)
            return np.sum(vals**2, axis=-1)

    elif eval_type == "binary_sum":
        genome_cfg = BinaryGenomeConfig(shape=(10,), p=0.5)
        evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=maximize))
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=1)
        crossover = BinaryUniformCrossover(num_offspring=2, crossover_rate=0.5)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        def raw_oracle(genes: BinaryGenome) -> np.ndarray:
            # Raw objective: count of ones
            return np.sum(np.asarray(genes.values), axis=-1)

    elif eval_type == "knapsack":
        weights = jnp.array([1.0, 1.0, 1.0, 1.0])
        values = jnp.array([2.0, 5.0, 10.0, 20.0])
        genome_cfg = BinaryGenomeConfig(shape=(4,), p=0.5)
        data = KnapsackData(weights=weights, values=values)
        cfg = KnapsackConfig(n_items=4, capacity=100.0, maximize=maximize)  # Feasible-only
        evaluator = KnapsackEvaluator(config=cfg, data=data)
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=1)
        crossover = BinaryUniformCrossover(num_offspring=2, crossover_rate=0.5)
        mutation = BitFlipMutation(num_offspring=1, mutation_rate=0.1)

        def raw_oracle(genes: BinaryGenome) -> np.ndarray:
            # Raw knapsack value (penalty is 0 due to large capacity)
            v = np.array([2.0, 5.0, 10.0, 20.0])
            return np.sum(np.asarray(genes.values) * v, axis=-1)

    elif eval_type == "bbobax":
        genome_cfg = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0))
        evaluator = BBOBAXEvaluator.create(
            BBOBAXConfig(fn_name="sphere", num_dims=2, maximize=maximize, seed=0)
        )
        selection = ElitePoolSelection(num_selections=pop_size, elite_k=1)
        crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
        mutation = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.1)

        def raw_oracle(genes: RealGenome) -> np.ndarray:
            # Independent raw BBOBAX task evaluation directly from the underlying BBOB task
            rng = jax.random.PRNGKey(0)

            def eval_single(x):
                if evaluator.state is not None:
                    _, res = evaluator.task.evaluate(rng, x, evaluator.state, evaluator.params)
                else:
                    res = evaluator.task.evaluate(rng, x, evaluator.params)
                return res.fitness

            return np.asarray(jax.vmap(eval_single)(genes.values))

    import numpy as np

    engine = GeneticEngine(
        evaluator=evaluator,
        genome_config=genome_cfg,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=GeneticEngineParams(
            pop_size=pop_size,
            num_generations=num_generations,
            track_best=track_best,
        ),
        enable_progress_bar=False,
    )

    init_state = engine.init_state(key)
    final_state, _, _ = engine.run(init_state, compile=False)

    # 1. Evaluate every individual in final population via independent raw oracle
    raw_scores = raw_oracle(final_state.population.genes)

    # 2. Determine best individual according to user's declared maximize intent
    if maximize:
        expected_best_idx = int(np.argmax(raw_scores))
    else:
        expected_best_idx = int(np.argmin(raw_scores))

    expected_best_genome_values = np.asarray(final_state.population.genes.values)[expected_best_idx]
    actual_best_genome_values = np.asarray(final_state.best_genome.values)

    # 3. Assert best_genome matches the optimal member according to raw intent
    assert np.allclose(actual_best_genome_values, expected_best_genome_values)

    # 4. Assert best_fitness matches the minimum internal fitness in final population (TrackBest.NONE)
    if track_best == TrackBest.NONE:
        expected_best_fitness = float(np.min(np.asarray(final_state.population.fitness)))
        assert np.isclose(float(final_state.best_fitness), expected_best_fitness)


# =============================================================================
# Test 2: UNIT: run() Post-Processing Matches step() / debug_step() Bookkeeping
# =============================================================================


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("track_best", [TrackBest.NONE, TrackBest.LIGHT, TrackBest.FULL])
def test_run_post_processing_matches_manual_step_loop(maximize: bool, track_best: TrackBest):
    """Verify that run()'s post-processing produces the exact same best_genome and

    best_fitness as a manual step() loop for any track_best mode.
    """
    key = jax.random.PRNGKey(123)
    pop_size = 20
    num_gens = 4

    genome_cfg = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0))
    evaluator = OptimizationEvaluator(
        env=SimpleSphereEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize),
    )
    engine = GeneticEngine(
        evaluator=evaluator,
        genome_config=genome_cfg,
        selection=ElitePoolSelection(num_selections=pop_size, elite_k=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.1),
        engine_params=GeneticEngineParams(
            pop_size=pop_size,
            num_generations=num_gens,
            track_best=track_best,
        ),
        enable_progress_bar=False,
    )

    # 1. Manual step loop
    init_state_step = engine.init_state(key)
    state = init_state_step
    for _ in range(num_gens):
        state, _ = engine.step(state)

    # If track_best == TrackBest.NONE or LIGHT, run() post-processes the final population
    expected_best_idx = int(jnp.argmin(state.population.fitness))
    if track_best in (TrackBest.NONE, TrackBest.LIGHT):
        expected_genome = jax.tree_util.tree_map(lambda x: x[expected_best_idx], state.population.genes)
    else:
        expected_genome = state.best_genome

    if track_best == TrackBest.NONE:
        expected_fitness = state.population.fitness[expected_best_idx]
    else:
        expected_fitness = state.best_fitness

    # 2. Call engine.run()
    init_state_run = engine.init_state(key)
    final_state_run, _, _ = engine.run(init_state_run, compile=False)

    # 3. Assert run() matches manual step loop exactly
    assert jnp.allclose(final_state_run.best_genome.values, expected_genome.values)
    assert jnp.isclose(final_state_run.best_fitness, expected_fitness)


# =============================================================================
# Test 3: UNIT: AST Static Check: Zero Engine-Level Maximize Branching
# =============================================================================


def test_no_engine_level_maximize_branching_ast():
    """Static AST audit asserting that no engine class branches on maximize or calls

    argmax on fitness arrays.
    """
    repo_root = Path(__file__).resolve().parents[2]
    engine_dir = repo_root / "src" / "malthusjax" / "engine"

    python_files = list(engine_dir.rglob("*.py"))
    assert len(python_files) > 0, "Could not locate engine files"

    forbidden_patterns = ["argmax"]

    for py_file in python_files:
        # Ignore non-code or test files
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for node in ast.walk(tree):
            # 1. Check for 'def maximize(self)' in any engine class
            if isinstance(node, ast.ClassDef):
                if node.name in ("GeneticEngine", "BaseEngine", "MOEngine", "BaseIslandModel"):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            assert item.name != "maximize", (
                                f"Forbidden maximize property found in {node.name} ({py_file.name})"
                            )

            # 2. Check for argmax calls in GeneticEngine or IslandEngineAdapter
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name in forbidden_patterns:
                    # Check arguments for fitness
                    args_text = ast.unparse(node)
                    # Exclude QD repertoires where QDAX explicitly maximizes
                    if "fitness" in args_text and not ("repertoire" in args_text or "rep_" in args_text):
                        pytest.fail(
                            f"Forbidden argmax on fitness array found in {py_file.name}:{node.lineno}: {args_text}"
                        )


# =============================================================================
# Test 4: UNIT: Evaluator Sign-Convention Compliance (A vs B)
# =============================================================================


@pytest.mark.parametrize("maximize", [True, False])
def test_evaluator_sign_convention_scalar_output(maximize: bool):
    """ScalarOutput / OptimizationEvaluator: superior individual has lower fitness."""
    env = SimpleSphereEnv()
    evaluator = OptimizationEvaluator(
        env=env,
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize),
    )
    # Sphere cost: opt_A = [0.1, 0.1] (cost 0.02), opt_B = [2.0, 2.0] (cost 8.0)
    # If maximize=True, user wants higher sum of squares (B is better, A is worse)
    # If maximize=False, user wants lower sum of squares (A is better, B is worse)
    if maximize:
        better_genome = RealGenome(values=jnp.array([2.0, 2.0]))
        worse_genome = RealGenome(values=jnp.array([0.1, 0.1]))
    else:
        better_genome = RealGenome(values=jnp.array([0.1, 0.1]))
        worse_genome = RealGenome(values=jnp.array([2.0, 2.0]))

    pop = BasePopulation(
        genes=RealGenome(values=jnp.stack([better_genome.values, worse_genome.values])),
        fitness=jnp.zeros(2),
    )
    eval_pop = evaluator.evaluate_population(pop)
    fit_better, fit_worse = eval_pop.fitness[0], eval_pop.fitness[1]

    # Lower fitness is ALWAYS better
    assert fit_better < fit_worse


@pytest.mark.parametrize("maximize", [True, False])
def test_evaluator_sign_convention_binary_sum(maximize: bool):
    """BinarySumEvaluator: superior individual has lower fitness."""
    evaluator = BinarySumEvaluator(config=BinarySumConfig(maximize=maximize))
    # A has 8 ones, B has 2 ones
    g_many_ones = BinaryGenome(values=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]))
    g_few_ones = BinaryGenome(values=jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

    better = g_many_ones if maximize else g_few_ones
    worse = g_few_ones if maximize else g_many_ones

    fit_better = evaluator.evaluate(better)
    fit_worse = evaluator.evaluate(worse)

    # Lower fitness is ALWAYS better
    assert fit_better < fit_worse


@pytest.mark.parametrize("maximize", [True, False])
def test_evaluator_sign_convention_knapsack(maximize: bool):
    """KnapsackEvaluator: superior individual has lower fitness."""
    weights = jnp.array([1.0, 1.0, 1.0, 1.0])
    values = jnp.array([2.0, 5.0, 10.0, 20.0])
    cfg = KnapsackConfig(n_items=4, capacity=100.0, maximize=maximize)
    evaluator = KnapsackEvaluator(config=cfg, data=KnapsackData(weights=weights, values=values))

    g_high_val = BinaryGenome(values=jnp.array([1, 1, 1, 1]))  # value = 37.0
    g_low_val = BinaryGenome(values=jnp.array([1, 0, 0, 0]))   # value = 2.0

    better = g_high_val if maximize else g_low_val
    worse = g_low_val if maximize else g_high_val

    fit_better = evaluator.evaluate(better)
    fit_worse = evaluator.evaluate(worse)

    # Lower fitness is ALWAYS better
    assert fit_better < fit_worse


@pytest.mark.parametrize("maximize", [True, False])
def test_evaluator_sign_convention_bbobax(maximize: bool):
    """BBOBAXEvaluator: superior individual has lower fitness."""
    evaluator = BBOBAXEvaluator.create(
        BBOBAXConfig(fn_name="sphere", num_dims=2, maximize=maximize, seed=0)
    )
    # Sphere cost: [0.0, 0.0] -> 0.0, [3.0, 3.0] -> 18.0
    if maximize:
        better = RealGenome(values=jnp.array([3.0, 3.0]))
        worse = RealGenome(values=jnp.array([0.0, 0.0]))
    else:
        better = RealGenome(values=jnp.array([0.0, 0.0]))
        worse = RealGenome(values=jnp.array([3.0, 3.0]))

    fit_better = evaluator.evaluate(better)
    fit_worse = evaluator.evaluate(worse)

    # Lower fitness is ALWAYS better
    assert fit_better < fit_worse


# =============================================================================
# Test 5: UNIT: NaN / Invalid Evaluation Sentinel Handling
# =============================================================================


@pytest.mark.parametrize("maximize", [True, False])
def test_tensorneat_nan_sentinel_is_worst_value(maximize: bool):
    """Assert that NaN/failed evaluation in TensorNeat produces +inf, which is strictly

    worse than any valid finite fitness under lower-is-better convention.
    """
    evaluator = TensorNeatEvaluator(
        env=DummyTensorNeatProblem(),
        transform=DummyTensorNeatTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(),
        forward_fn=lambda x: x,
        maximize=maximize,
    )

    # In TensorNeat, values is (nodes, conns)
    nodes = jnp.array([[10.0], [jnp.nan]])
    conns = jnp.zeros((2, 1))
    genes = RealGenome(values=(nodes, conns))
    pop = BasePopulation(genes=genes, fitness=jnp.zeros(2))

    eval_pop = evaluator.evaluate_population(pop)

    fit_valid = eval_pop.fitness[0]
    fit_nan = eval_pop.fitness[1]

    # In lower-is-better, valid must be strictly better (lower) than the NaN sentinel
    assert jnp.isfinite(fit_valid)
    assert jnp.isposinf(fit_nan)  # +inf is worst under minimization
    assert fit_valid < fit_nan


# =============================================================================
# Test 6: REGRESSION: forward_presplit_keys=True
# =============================================================================


@pytest.mark.xfail(
    strict=True,
    reason="E7: TournamentSelection key shape mismatch with forward_presplit_keys=True handled in separate PR",
)
def test_forward_presplit_keys_regression():
    """Verify that forward_presplit_keys=True executes without crashing or silently

    swallowing exceptions.
    """
    engine = GeneticEngine(
        evaluator=OptimizationEvaluator(
            env=SimpleSphereEnv(),
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=False),
        ),
        genome_config=RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0)),
        selection=TournamentSelection(num_selections=10, tournament_size=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1),
        engine_params=GeneticEngineParams(
            pop_size=10,
            num_generations=2,
            forward_presplit_keys=True,
        ),
        enable_progress_bar=False,
    )

    init_state = engine.init_state(jax.random.PRNGKey(42))
    k_sel = jax.random.split(jax.random.PRNGKey(10), 1)

    # In E7, this silently caught ValueError and returned un-presplit parent_idx
    _, parent_idx = engine._selection_phase(
        k_sel, init_state.population, init_state.operators, engine.engine_params
    )
    assert parent_idx.shape == (20,)
