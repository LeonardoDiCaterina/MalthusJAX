import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.engine.base import AbstractEvolutionState


# --- Mock Classes for Testing ---
@struct.dataclass
class MockGenome(BaseGenome):
    values: jnp.ndarray
    static_val: int = struct.field(pytree_node=False, default=42)

    @classmethod
    def random_init(cls, key, config):
        return cls(values=jax.random.normal(key, (10,)))

    def distance(self, other, metric):
        return 0.0

    def autocorrect(self, config):
        return self

    @property
    def size(self):
        return 10

    @property
    def shape(self):
        return (10,)

    @classmethod
    def from_tensor(cls, arr, config=None):
        return cls(values=arr)


@struct.dataclass
class MockPopulation(BasePopulation[MockGenome]):
    pass


@struct.dataclass
class MockState(AbstractEvolutionState[MockGenome, MockPopulation]):
    pass


# --- Fixtures ---
@pytest.fixture
def sample_state():
    key = jax.random.PRNGKey(0)
    genome = MockGenome(values=jnp.ones((5, 10)))
    population = MockPopulation(
        genes=genome,
        fitness=jnp.zeros(5),
        info={"meta": jnp.array([1, 2, 3])}
    )
    return MockState(
        population=population,
        best_genome=MockGenome(values=jnp.ones(10)),
        generation=1,
        best_fitness=jnp.array(0.0),
        rng_key=key
    )


# --- Tests ---
def test_copy_value_equivalence(sample_state):
    """Test that all arrays are deeply copied and equivalent in value."""
    copied_state = sample_state.copy()

    # Verify standard values
    assert copied_state.generation == sample_state.generation
    assert jnp.array_equal(copied_state.best_fitness, sample_state.best_fitness)
    assert jnp.array_equal(copied_state.population.fitness, sample_state.population.fitness)
    assert jnp.array_equal(copied_state.population.genes.values, sample_state.population.genes.values)
    assert jnp.array_equal(copied_state.population.info["meta"], sample_state.population.info["meta"])


def test_copy_buffer_donation_isolation(sample_state):
    """
    Test that the copied state survives the original state being donated
    to a JIT compiled function.
    """
    copied_state = sample_state.copy()

    # Define a function that aggressively donates its input state
    @jax.jit(donate_argnums=0)
    def consume_state(s):
        # Mutate the arrays to ensure XLA actually consumes the buffer
        new_pop = s.population.replace(fitness=s.population.fitness + 1.0)
        return s.replace(population=new_pop, generation=s.generation + 1)

    # Consume the original state
    _ = consume_state(sample_state)

    # Verify the copied state is completely unaffected by the donation/mutation
    assert copied_state.generation == 1
    assert jnp.array_equal(copied_state.population.fitness, jnp.zeros(5))
    assert jnp.array_equal(copied_state.population.genes.values, jnp.ones((5, 10)))


def test_copy_immutability_preservation(sample_state):
    """Test that static (non-pytree) nodes are preserved as-is."""
    copied_state = sample_state.copy()

    # The static_val shouldn't be cast to a JAX array
    assert isinstance(copied_state.population.genes.static_val, int)
    assert copied_state.population.genes.static_val == 42
