import jax
import jax.numpy as jnp
import pytest
pytest.importorskip('kozax')

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
        # Return fitness, and unmutated pop (Kozax allows optimization inside eval, mock doesn't)
        return fitness, pop

    def evolve_population(self, pop, fitness, key):
        # Mock evolve: add noise and scale by fitness to make it dependent on fitness
        noise = jax.random.normal(key, pop.shape) * 0.1
        # reshape fitness to broadcast
        fitness_broadcast = fitness.reshape(-1, 1, 1, 1)
        return pop + noise * fitness_broadcast


class TestKozaxAdapter(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = False, eval_mode: str = "native", seed: int = 0):
        # We don't support MalthusJAX mode yet for Kozax, so we bypass this test if asked
        if eval_mode != "native":
            pytest.skip("Kozax adapter does not support EvalMode.MALTHUSJAX yet.")

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

    def test_maximize_flag_changes_outcome(self):
        pytest.skip(
            "Kozax native mode uses native evaluator which always minimizes. Maximize flag only affects metrics."
        )

    def test_maximize_history_changes(self):
        pytest.skip("Skipping because Kozax native mode mock doesn't support generic maximization.")
