"""
BENCHMARK GROUP 4 — Operator-Level Microbenchmarks
==================================================

Isolated operator-level benchmarks for MalthusJAX components. No evosax
counterpart.
"""

import jax
import jax.random as jr
import pytest

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.selection.tournament import TournamentSelection
from tests.benchmarks.conftest_benchmarks import (
    DIMENSIONS,
    POP_SIZES,
    SEED,
)


class TestOperatorMicrobenchmarks:
    """Isolated operator-level benchmarks for MalthusJAX components."""

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    def test_elite_pool_selection(self, benchmark, pop_size: int):
        """ElitePoolSelection warm dispatch."""
        elite_k = max(1, pop_size // 2)
        sel = ElitePoolSelection(num_selections=pop_size, elite_k=elite_k)

        fitness = jax.random.uniform(jr.PRNGKey(0), (pop_size,))
        key = jr.PRNGKey(1)

        jit_sel = jax.jit(sel)
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

    @pytest.mark.parametrize("pop_size", POP_SIZES)
    @pytest.mark.parametrize("dims", DIMENSIONS)
    def test_uniform_crossover(self, benchmark, pop_size: int, dims: int):
        """UniformCrossover: full __call__ including vmap + transpose."""
        genome_config = RealGenomeConfig(shape=(dims,), bounds=(-5.0, 5.0))
        num_pairs = pop_size // 2

        crossover = UniformCrossover(num_offspring=2, crossover_rate=0.5, input_length=num_pairs)

        key = jr.PRNGKey(SEED)
        p1_pop = RealPopulation.init_random(jr.PRNGKey(0), genome_config, size=num_pairs)
        p2_pop = RealPopulation.init_random(jr.PRNGKey(1), genome_config, size=num_pairs)

        num_keys = crossover.num_keys(input_shape=(num_pairs,))
        keys = jax.random.split(key, num_keys)

        jit_cross = jax.jit(crossover)
        _out = jit_cross(keys, p1_pop, p2_pop, genome_config)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), _out)

        def _run():
            out = jit_cross(keys, p1_pop, p2_pop, genome_config)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

        benchmark.group = f"operator_crossover/pop{pop_size}_d{dims}"
        benchmark.name = "uniform_crossover"
        benchmark(_run)

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
