import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

pytest.importorskip("kozax")
from kozax.fitness_functions.SR_fitness_function import SymbolicRegressionFitnessFunction
from kozax.genetic_programming import GeneticProgramming

from malthusjax.composer.kozax_adapter import build_kozax_engine

from .base_adapter_suite import BaseAdapterTestSuite


class MockKozaxGP:
    def __init__(self, population_size, num_populations, max_nodes=10):
        self.population_size = population_size
        self.num_populations = num_populations
        self.max_nodes = max_nodes

    def initialize_population(self, key):
        # returns shape (num_populations * population_size, 1, max_nodes, 4)
        total_pop = self.population_size * self.num_populations
        return jax.random.uniform(key, (total_pop, 1, self.max_nodes, 4))

    def evaluate_population(self, pop, data, key):
        # Mock evaluation: just sum all elements along axes to produce scalar fitness per individual
        fitness = jnp.sum(pop, axis=(1, 2, 3))
        # Return fitness, and unmutated pop
        return fitness, pop

    def evolve_population(self, pop, fitness, key):
        # Mock evolve: add noise and scale by fitness to make it dependent on fitness
        noise = jax.random.normal(key, pop.shape) * 0.1
        fitness_broadcast = fitness.reshape(-1, 1, 1, 1)
        return pop + noise * fitness_broadcast


class TestKozaxAdapterMocked(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        if eval_mode != "native":
            return None

        strategy = MockKozaxGP(population_size=10, num_populations=2)
        evaluator = {"mock_data": True}

        return build_kozax_engine(
            strategy_obj=strategy,
            evaluator=evaluator,
            generations=3,
            pop_size=20,
            maximize=maximize,
            eval_mode=eval_mode,
        )


class TestKozaxAdapterRealIntegration(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        if eval_mode != "native":
            return None

        x = jnp.column_stack([jnp.linspace(-2.0, 2.0, 50), jnp.linspace(1.0, 3.0, 50)])
        y = (x[:, 0] ** 2 + x[:, 1]).reshape(-1)
        data = (x, y)
        ff = SymbolicRegressionFitnessFunction()

        gp = GeneticProgramming(
            num_generations=2,
            population_size=16,
            num_populations=1,
            fitness_function=ff,
            variable_list=[["x0", "x1"]],
            operator_list=["add", "mul", "sub"],
            max_nodes=15,
        )

        return build_kozax_engine(
            strategy_obj=gp,
            evaluator=data,
            generations=2,
            pop_size=16,
            maximize=maximize,
            eval_mode=eval_mode,
        )

    def test_different_keys_different_results(self, small_adapter):
        """Verify different PRNG keys yield distinct trajectories."""
        r1 = small_adapter.run_once(jr.PRNGKey(0))
        r2 = small_adapter.run_once(jr.PRNGKey(12345))

        # In symbolic regression, multiple seeds may discover the same discrete expression
        # as best_fitness, but the evolutionary trajectory (mean fitness across population)
        # must be distinct.
        trajectory_differs = (
            r1["summary"]["best_fitness"] != r2["summary"]["best_fitness"]
            or r1["summary"]["mean_fitness"] != r2["summary"]["mean_fitness"]
            or any(
                h1["mean_fitness"] != h2["mean_fitness"]
                for h1, h2 in zip(r1["history"], r2["history"])
            )
        )
        assert trajectory_differs
