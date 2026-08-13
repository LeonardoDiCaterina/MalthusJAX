import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation
from malthusjax.operators.emitters.tensorneat_variants import (
    TensorNeatCrossoverEmitter,
    TensorNeatMutationEmitter,
)

try:
    from tensorneat.common import State
    from tensorneat.genome import DefaultGenome

    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False


class MockTNRepertoire:
    def __init__(self, key, batch_size, nodes, conns):
        self.batch_size = batch_size
        self.genotypes = (jnp.tile(nodes, (batch_size, 1, 1)), jnp.tile(conns, (batch_size, 1, 1)))

    def select(self, key, batch_size):
        # Return exact slice matching requested batch size
        return type(
            "MockSelected",
            (object,),
            {"genotypes": (self.genotypes[0][:batch_size], self.genotypes[1][:batch_size])},
        )()


@pytest.mark.skipif(not TENSORNEAT_AVAILABLE, reason="TensorNEAT not installed")
def test_tensorneat_mutation_emitter():
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)

    emitter = TensorNeatMutationEmitter(_batch_size=8, genome=genome)

    # 1 sampling key + 8 * 1 atomic key = 9 keys
    assert emitter.num_keys() == 9

    total_keys = emitter.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(42), total_keys)

    # Init empty genome
    state = State(randkey=jax.random.PRNGKey(0), generation=0.0)
    state = genome.setup(state)

    init_keys = jax.random.split(jax.random.PRNGKey(1), 8)
    nodes, conns = jax.vmap(genome.initialize, in_axes=(None, 0))(state, init_keys)

    pop = TensorNeatPopulation(
        genes=TensorNeatGenome(values=(jnp.expand_dims(nodes, 0), jnp.expand_dims(conns, 0))),
        fitness=jnp.array([0.0]),
        config=None,
    )

    repertoire = MockTNRepertoire(jax.random.PRNGKey(0), 100, nodes, conns)

    emitter_state = emitter.init(jax.random.PRNGKey(0), pop)

    @jax.jit
    def jit_ask(st, k):
        return emitter.ask(st, repertoire, k)

    out_pop, new_state = jit_ask(emitter_state, keys)

    assert out_pop.genes.values[0].shape == (8, 10, genome.node_gene.length)
    assert out_pop.genes.values[1].shape == (8, 20, genome.conn_gene.length)


@pytest.mark.skipif(not TENSORNEAT_AVAILABLE, reason="TensorNEAT not installed")
def test_tensorneat_crossover_emitter():
    genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=10, max_conns=20)

    emitter = TensorNeatCrossoverEmitter(_batch_size=8, genome=genome)

    # 2 sampling keys + 8 * 1 atomic key = 10 keys
    assert emitter.num_keys() == 10

    total_keys = emitter.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(42), total_keys)

    state = State(randkey=jax.random.PRNGKey(0), generation=0.0)
    state = genome.setup(state)

    init_keys = jax.random.split(jax.random.PRNGKey(1), 8)
    nodes, conns = jax.vmap(genome.initialize, in_axes=(None, 0))(state, init_keys)

    pop = TensorNeatPopulation(
        genes=TensorNeatGenome(values=(jnp.expand_dims(nodes, 0), jnp.expand_dims(conns, 0))),
        fitness=jnp.array([0.0]),
        config=None,
    )

    repertoire = MockTNRepertoire(jax.random.PRNGKey(0), 100, nodes, conns)

    emitter_state = emitter.init(jax.random.PRNGKey(0), pop)

    @jax.jit
    def jit_ask(st, k):
        return emitter.ask(st, repertoire, k)

    out_pop, new_state = jit_ask(emitter_state, keys)

    assert out_pop.genes.values[0].shape == (8, 10, genome.node_gene.length)
    assert out_pop.genes.values[1].shape == (8, 20, genome.conn_gene.length)
