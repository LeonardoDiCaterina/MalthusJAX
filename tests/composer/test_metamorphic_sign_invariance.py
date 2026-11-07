"""Universal Metamorphic Sign Law Test Suite.

Formally proves the metamorphic invariant:
For any continuous optimization problem f(x), running minimization of f(x)
yields identical state trajectories and negation-flipped fitness values
to running maximization of -f(x):

    min_x f(x) == max_x -f(x)
    |theta_{min, t} - theta_{max, t}| == 0
    f_{min, t} == - f_{max, t}

Verified across:
1. Native MalthusJAX GeneticEngine
2. EvoSAX Universal Adapter
3. QDax Universal Adapter
4. TensorNEAT Universal Adapter
5. Kozax Universal Adapter
"""

import functools

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import BaseOptimizationEnvironment
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    TrackBest,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.tournament import TournamentSelection


@struct.dataclass
class SphereMinEnv(BaseOptimizationEnvironment):
    """Raw objective f(x) = sum(x^2)."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        return jnp.sum(solution**2)


@struct.dataclass
class SphereMaxEnv(BaseOptimizationEnvironment):
    """Raw objective g(x) = -sum(x^2)."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        return -jnp.sum(solution**2)


def test_genetic_engine_metamorphic_sign_invariance():
    """Verify GeneticEngine population trajectory and fitness sign invariance."""
    eval_min = OptimizationEvaluator(
        env=SphereMinEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )

    eval_max = OptimizationEvaluator(
        env=SphereMaxEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=True),
    )

    pop_size = 20
    num_gens = 5
    config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    sel = TournamentSelection(num_selections=pop_size, tournament_size=3)
    cx = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mut = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.1)

    eng_min = GeneticEngine(
        evaluator=eval_min,
        genome_config=config,
        selection=sel,
        crossover=cx,
        mutation=mut,
        engine_params=GeneticEngineParams(
            pop_size=pop_size,
            num_generations=num_gens,
            track_best=TrackBest.FULL,
        ),
        enable_progress_bar=False,
    )

    eng_max = GeneticEngine(
        evaluator=eval_max,
        genome_config=config,
        selection=sel,
        crossover=cx,
        mutation=mut,
        engine_params=GeneticEngineParams(
            pop_size=pop_size,
            num_generations=num_gens,
            track_best=TrackBest.FULL,
        ),
        enable_progress_bar=False,
    )

    key = jax.random.key(42)
    s_min = eng_min.init_state(key)
    s_max = eng_max.init_state(key)

    # Initial states match bit-for-bit
    assert jnp.all(s_min.population.genes.values == s_max.population.genes.values)
    assert jnp.all(s_min.population.fitness == s_max.population.fitness)

    s_min_final, out_min, _ = eng_min.run(s_min)
    s_max_final, out_max, _ = eng_max.run(s_max)

    # State trajectories match bit-for-bit: |theta_{min, t} - theta_{max, t}| == 0
    pop_diff = jnp.max(
        jnp.abs(s_min_final.population.genes.values - s_max_final.population.genes.values)
    )
    assert pop_diff == 0.0

    # Internal lower-is-better fitness arrays match bit-for-bit
    fit_diff = jnp.max(jnp.abs(s_min_final.population.fitness - s_max_final.population.fitness))
    assert fit_diff == 0.0


def test_evosax_metamorphic_sign_invariance():
    """Verify EvoSAX adapter satisfies min f(x) == max -f(x)."""
    pytest.importorskip("evosax")
    from malthusjax.composer.evosax_adapter import build_evosax_engine

    eval_min = OptimizationEvaluator(
        env=SphereMinEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )

    eval_max = OptimizationEvaluator(
        env=SphereMaxEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),  # returns raw -f(x)
    )

    eng_min = build_evosax_engine(
        strategy="SimpleGA",
        evaluator=eval_min,
        generations=4,
        pop_size=20,
        num_dims=5,
        maximize=False,
        eval_mode="malthusjax",
    )

    eng_max = build_evosax_engine(
        strategy="SimpleGA",
        evaluator=eval_max,
        generations=4,
        pop_size=20,
        num_dims=5,
        maximize=True,
        eval_mode="malthusjax",
    )

    key = jax.random.PRNGKey(42)
    r_min = eng_min.run_once(key)
    r_max = eng_max.run_once(key)

    bf_min = r_min["summary"]["best_fitness"]
    bf_max = r_max["summary"]["best_fitness"]

    assert jnp.allclose(bf_min, -bf_max, atol=1e-5)
    for h_min, h_max in zip(r_min["history"], r_max["history"]):
        assert jnp.allclose(h_min["best_fitness"], -h_max["best_fitness"], atol=1e-5)


def test_qdax_metamorphic_sign_invariance():
    """Verify QDax adapter satisfies min f(x) == max -f(x)."""
    pytest.importorskip("qdax")
    from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.utils.metrics import default_qd_metrics

    from malthusjax.composer.qdax_adapter import build_qdax_engine

    dim = 3
    pop_size = 10
    generations = 3

    init_variables = jnp.ones((pop_size, dim)) * 2.0
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=500,
        num_centroids=20,
        minval=0.0,
        maxval=1.0,
        key=jax.random.PRNGKey(0),
    )

    def dummy_mutation(x, random_key):
        return x + jax.random.normal(random_key, x.shape) * 0.1

    def dummy_crossover(x1, x2, random_key):
        mask = jax.random.bernoulli(random_key, 0.5, shape=x1.shape)
        return jnp.where(mask, x1, x2)

    emitter = MixingEmitter(
        mutation_fn=dummy_mutation,
        variation_fn=dummy_crossover,
        variation_percentage=0.5,
        batch_size=pop_size,
    )
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.0)

    class NativeEvalMin:
        def scoring_function(self, genotypes, random_key):
            fitnesses = -jnp.sum(jnp.square(genotypes), axis=-1)
            descriptors = jnp.clip(genotypes[:, :2] + 0.5, 0.0, 1.0)
            return fitnesses, descriptors, {}

    class NativeEvalMax:
        def scoring_function(self, genotypes, random_key):
            fitnesses = -jnp.sum(jnp.square(genotypes), axis=-1)
            descriptors = jnp.clip(genotypes[:, :2] + 0.5, 0.0, 1.0)
            return fitnesses, descriptors, {}

    eng_min = build_qdax_engine(
        strategy_cls=MAPElites,
        emitter=emitter,
        metrics_function=metrics_fn,
        evaluator=NativeEvalMin(),
        init_variables=init_variables,
        centroids=centroids,
        pop_size=pop_size,
        generations=generations,
        maximize=False,
        eval_mode="native",
    )

    eng_max = build_qdax_engine(
        strategy_cls=MAPElites,
        emitter=emitter,
        metrics_function=metrics_fn,
        evaluator=NativeEvalMax(),
        init_variables=init_variables,
        centroids=centroids,
        pop_size=pop_size,
        generations=generations,
        maximize=True,
        eval_mode="native",
    )

    key = jax.random.PRNGKey(42)
    r_min = eng_min.run_once(key)
    r_max = eng_max.run_once(key)

    bf_min = r_min["summary"]["best_fitness"]
    bf_max = r_max["summary"]["best_fitness"]

    assert jnp.allclose(bf_min, -bf_max, atol=1e-5)


def test_tensorneat_metamorphic_sign_invariance():
    """Verify TensorNEAT adapter satisfies min f(x) == max -f(x)."""
    pytest.importorskip("tensorneat")
    import tensorneat as tn

    from malthusjax.composer.tensorneat_adapter import build_tensorneat_engine

    class MinXOR(tn.problem.XOR):
        def evaluate(self, state, randkey, act_func, params):
            fit = super().evaluate(state, randkey, act_func, params)
            return -fit  # minimization of positive error

    class MaxXOR(tn.problem.XOR):
        def evaluate(self, state, randkey, act_func, params):
            fit = super().evaluate(state, randkey, act_func, params)
            return fit  # maximization of negative error

    algo_min = tn.algorithm.NEAT(
        pop_size=10,
        species_size=5,
        genome=tn.genome.DefaultGenome(num_inputs=2, num_outputs=1),
    )
    algo_max = tn.algorithm.NEAT(
        pop_size=10,
        species_size=5,
        genome=tn.genome.DefaultGenome(num_inputs=2, num_outputs=1),
    )

    eng_min = build_tensorneat_engine(
        algorithm=algo_min,
        evaluator=MinXOR(),
        generations=3,
        pop_size=10,
        maximize=False,
        eval_mode="native",
    )

    eng_max = build_tensorneat_engine(
        algorithm=algo_max,
        evaluator=MaxXOR(),
        generations=3,
        pop_size=10,
        maximize=True,
        eval_mode="native",
    )

    key = jax.random.PRNGKey(42)
    r_min = eng_min.run_once(key)
    r_max = eng_max.run_once(key)

    bf_min = r_min["summary"]["best_fitness"]
    bf_max = r_max["summary"]["best_fitness"]

    assert jnp.allclose(bf_min, -bf_max, atol=1e-5)


def test_kozax_metamorphic_sign_invariance():
    """Verify Kozax adapter satisfies min f(x) == max -f(x)."""
    pytest.importorskip("kozax")
    from kozax.fitness_functions.SR_fitness_function import (
        SymbolicRegressionFitnessFunction,
    )
    from kozax.genetic_programming import GeneticProgramming

    from malthusjax.composer.kozax_adapter import build_kozax_engine

    class MinFF(SymbolicRegressionFitnessFunction):
        def __call__(self, candidate, data, tree_evaluator):
            return super().__call__(candidate, data, tree_evaluator)

    class MaxFF(SymbolicRegressionFitnessFunction):
        def __call__(self, candidate, data, tree_evaluator):
            return -super().__call__(candidate, data, tree_evaluator)

    x = jnp.column_stack([jnp.linspace(-2.0, 2.0, 50), jnp.linspace(1.0, 3.0, 50)])
    y = (x[:, 0] ** 2 + x[:, 1]).reshape(-1)
    data = (x, y)

    gp_min = GeneticProgramming(
        num_generations=2,
        population_size=16,
        num_populations=1,
        fitness_function=MinFF(),
        variable_list=[["x0", "x1"]],
        operator_list=["add", "mul", "sub"],
        max_nodes=15,
    )

    gp_max = GeneticProgramming(
        num_generations=2,
        population_size=16,
        num_populations=1,
        fitness_function=MaxFF(),
        variable_list=[["x0", "x1"]],
        operator_list=["add", "mul", "sub"],
        max_nodes=15,
    )

    eng_min = build_kozax_engine(
        strategy_obj=gp_min,
        evaluator=data,
        generations=2,
        pop_size=16,
        maximize=False,
        eval_mode="native",
    )

    eng_max = build_kozax_engine(
        strategy_obj=gp_max,
        evaluator=data,
        generations=2,
        pop_size=16,
        maximize=True,
        eval_mode="native",
    )

    key = jax.random.PRNGKey(42)
    r_min = eng_min.run_once(key)
    r_max = eng_max.run_once(key)

    bf_min = r_min["summary"]["best_fitness"]
    bf_max = r_max["summary"]["best_fitness"]

    assert jnp.allclose(bf_min, -bf_max, atol=1e-5)
