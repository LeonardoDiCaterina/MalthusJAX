import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig


@pytest.fixture
def config():
    return LinearGenomeConfig(
        length=5,
        max_arity=2,
        num_inputs=2,
        num_ops=3
    )

def test_linear_genome_random_init(config):
    key = jax.random.PRNGKey(0)
    genome = LinearGenome.random_init(key, config)
    assert genome.ops.shape == (5,)
    assert genome.args.shape == (5, 2)


def test_linear_genome_decode_tree(config):
    # create a simple genome
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2]),
        args=jnp.array([[0, 1], [0, 1], [0, 1]])
    )
    assert genome.ops.shape == (3,)

def test_linear_genome_init_population(config):
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, 10)
    assert pop.genes.ops.shape == (10, 5)
    assert pop.genes.args.shape == (10, 5, 2)
    assert pop.fitness.shape == (10,)

def test_linear_genome_values():
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2]),
        args=jnp.array([[0, 1], [0, 1], [0, 1]])
    )
    ops, args = genome.values
    assert ops.shape == (3,)
    assert args.shape == (3, 2)

def test_linear_genome_autocorrect(config):
    genome = LinearGenome(
        ops=jnp.array([10, -1, 0, 1, 2]),
        args=jnp.array([[10, 10], [-1, -1], [0, 1], [0, 1], [0, 1]])
    )
    corrected = genome.autocorrect(config)
    assert jnp.all(corrected.ops >= 0)
    assert jnp.all(corrected.ops < config.num_ops)
    assert corrected.args.shape == (5, 2)

def test_linear_genome_distance():
    g1 = LinearGenome(
        ops=jnp.array([0, 1, 2]),
        args=jnp.array([[0, 1], [0, 1], [0, 1]])
    )
    g2 = LinearGenome(
        ops=jnp.array([0, 1, 0]),
        args=jnp.array([[0, 1], [0, 1], [1, 1]])
    )
    dist = g1.distance(g2)
    assert dist == 2 # 1 ops diff, 1 args diff
    
    dist_euclid = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)
    assert dist_euclid > 0

def test_linear_genome_properties():
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2]),
        args=jnp.array([[0, 1], [0, 1], [0, 1]])
    )
    assert genome.size == 3
    assert genome.shape == (3, 2)

def test_linear_genome_from_tensor():
    ops = jnp.array([0, 1, 2])
    args = jnp.array([[0, 1], [0, 1], [0, 1]])
    genome = LinearGenome.from_tensor((ops, args))
    assert genome.ops.shape == (3,)
    assert genome.args.shape == (3, 2)

def test_linear_genome_render(config):
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2, 0, 1]),
        args=jnp.zeros((5, 2), dtype=jnp.int32)
    )
    s = genome.render(config)
    assert isinstance(s, str)
    assert len(s) > 0
    assert "v_0" in s


def test_linear_genome_render_with_op_names(config):
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2, 0, 1]),
        args=jnp.zeros((5, 2), dtype=jnp.int32)
    )
    s = genome.render(config, op_names=["add", "mul", "sin"])
    assert "add" in s or "mul" in s or "sin" in s


def test_linear_genome_repr():
    genome = LinearGenome(
        ops=jnp.array([0, 1, 2]),
        args=jnp.array([[0, 1], [0, 1], [0, 1]])
    )
    r = repr(genome)
    assert "LinearGenome" in r


def test_linear_population_init_random(config):
    from malthusjax.core.genome.linear_genome import LinearPopulation
    key = jax.random.PRNGKey(42)
    pop = LinearPopulation.init_random(key, config, size=4)
    assert pop.genes.ops.shape == (4, 5)
    assert pop.genes.args.shape == (4, 5, 2)
    assert pop.fitness.shape == (4,)
