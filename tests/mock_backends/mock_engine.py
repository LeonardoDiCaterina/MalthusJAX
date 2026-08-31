from typing import Any, Dict, NamedTuple, Tuple

import chex
import jax.numpy as jnp

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation


class DummyRepertoire(NamedTuple):
    fitnesses: chex.Array


@chex.dataclass
class DummyState:
    generation: int

    def update(self, **kwargs):
        return self


class MockEvaluator:
    """A dummy evaluator that performs virtually zero work and returns deterministic fitness arrays."""

    def __init__(self, pop_size: int, genome_length: int, num_descriptors: int = 2):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.num_descriptors = num_descriptors
        self.config = type(
            "DummyConfig",
            (),
            {"genome_config": RealGenomeConfig(shape=(genome_length,), bounds=(-5.0, 5.0))},
        )()
        # For MalthusJAX
        self.evosax_problem = self

    def evaluate_population(self, population: RealPopulation) -> RealPopulation:
        # Return fitness based on the first gene value (which is set by ask)
        fitness = population.genes.values[:, 0]
        descriptors = jnp.zeros((self.pop_size, self.num_descriptors))

        pop = RealPopulation(
            genes=population.genes,
            fitness=fitness,
            config=population.config,
            info={"descriptors": descriptors},
        )
        return pop

    def eval(self, key, population, state):
        fitness = jnp.arange(self.pop_size, dtype=jnp.float32)
        return fitness, state, {}

    def sample(self, key):
        return jnp.zeros(self.genome_length)

    def evaluate(self, state, keys, forward, transformed_pop):
        return transformed_pop[0]

    def scoring_function(self, genotypes, random_key):
        fitness = genotypes[:, 0]
        descriptors = jnp.zeros((self.pop_size, self.num_descriptors))
        return fitness, descriptors, {}


class MockUniversalEngine:
    """A unified object exposing signatures for all 4 backend adapters to simulate zero-overhead workflows."""

    def __init__(
        self, pop_size: int = 100, genome_length: int = 2, maximize: bool = True, **kwargs
    ):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.maximize = maximize

        # MalthusJAX specific
        self.engine_params = type("DummyParams", (), {"maximize": maximize})()

        # EvoSAX specific
        self.num_dims = genome_length

    # --- EvoSAX / QDAX API ---
    def init(self, *args, **kwargs) -> Any:
        # EvoSAX: init(key, pop, fit, params) -> state
        # QDAX: init(init_variables, centroids, key) -> repertoire, emitter_state, random_key

        # If the first argument is a PRNGKey (usually an array of size 2 or shaped), it might be EvoSAX.
        # But an easier way: EvoSAX passes 4 arguments usually (key, pop, fit, params).
        # QDAX passes 3 (init_variables, centroids, key).
        if len(args) == 3:
            # QDAX
            key = args[2]
            dummy_repertoire = DummyRepertoire(
                fitnesses=jnp.arange(self.pop_size, dtype=jnp.float32)
                if self.maximize
                else -jnp.arange(self.pop_size, dtype=jnp.float32)
            )
            emitter_state = jnp.zeros(1, dtype=jnp.int32)
            return dummy_repertoire, emitter_state, key
        else:
            # EvoSAX
            return jnp.zeros(1, dtype=jnp.int32)  # Simple state is a generation counter

    def ask(self, *args, **kwargs) -> Any:
        # Create a population where the first element of each genome is its index
        pop = jnp.zeros((self.pop_size, self.genome_length))
        indices = jnp.arange(self.pop_size, dtype=jnp.float32)
        pop = pop.at[:, 0].set(indices)

        if len(args) == 1:
            # TensorNEAT: ask(state)
            state = args[0]
            return pop
        else:
            # EvoSAX: ask(key, state, params)
            state = args[1]
            return pop, state

    def tell(self, *args, **kwargs) -> Any:
        if len(args) == 2:
            # TensorNEAT: tell(state, fitness)
            state = args[0]
            return state.update(generation=state.generation + 1)
        else:
            # EvoSAX: tell(key, pop, fit, state, params)
            state = args[3]
            tell_fitness = args[2]
            return state + 1, {"best_fitness": jnp.min(tell_fitness)}

    # --- QDAX API ---
    def update(
        self, repertoire: Any, emitter_state: Any, key: chex.PRNGKey
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        dummy_repertoire = DummyRepertoire(
            fitnesses=jnp.arange(self.pop_size, dtype=jnp.float32)
            if self.maximize
            else -jnp.arange(self.pop_size, dtype=jnp.float32)
        )
        metrics = {
            "max_fitness": jnp.max(dummy_repertoire.fitnesses),
            "qd_score": jnp.sum(jnp.arange(self.pop_size, dtype=jnp.float32)),
            "coverage": 1.0,
        }
        return dummy_repertoire, emitter_state, metrics

    # --- TensorNEAT API ---
    def setup(self, state: Any = None) -> Any:
        return DummyState(generation=0)

    def transform(self, state: Any, pop: chex.Array) -> chex.Array:
        return pop

    def forward(self, state: Any, genes: chex.Array) -> chex.Array:
        return genes

    # --- MalthusJAX API ---
    def step(self, key: chex.PRNGKey, state: Any) -> Tuple[Any, Dict[str, Any]]:
        dummy_repertoire = DummyRepertoire(fitnesses=jnp.arange(self.pop_size, dtype=jnp.float32))

        new_state = (
            (state[0], state[1], state[2] + 1)
            if isinstance(state, tuple) and len(state) > 2
            else state
        )

        metrics = {
            "best_fitness": jnp.max(dummy_repertoire.fitnesses),
            "qd_score": jnp.sum(dummy_repertoire.fitnesses),
            "coverage": 1.0,
            "mean_fitness": jnp.mean(dummy_repertoire.fitnesses),
            "std_fitness": jnp.std(dummy_repertoire.fitnesses),
        }
        return new_state, metrics
