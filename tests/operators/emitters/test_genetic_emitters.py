import jax

from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.emitters.genetic import GeneticCrossoverEmitter, GeneticMutationEmitter
from malthusjax.operators.mutation.real import GaussianMutation


class MockRepertoire:
    def __init__(self, key, batch_size, genome_size):
        self.batch_size = batch_size
        self.genotypes = jax.random.normal(key, (batch_size, genome_size))

    def select(self, key, batch_size):
        # Mocking selection by returning random genotypes for now
        return type(
            "MockSelected",
            (object,),
            {"genotypes": jax.random.normal(key, (batch_size, self.genotypes.shape[1]))},
        )()


def test_genetic_mutation_emitter():
    genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    mutation = GaussianMutation(mutation_strength=1.0, mutation_rate=0.5)

    emitter = GeneticMutationEmitter(_batch_size=32, mutation=mutation, genome_config=genome_config)

    # 1 sampling key + 32 * 2 atomic keys (GaussianMutation uses 2 keys)
    assert emitter.num_keys() == 1 + 32 * 2

    total_keys = emitter.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(42), total_keys)

    repertoire = MockRepertoire(jax.random.PRNGKey(0), 100, 10)

    @jax.jit
    def jit_ask(k):
        return emitter.ask(None, repertoire, k)

    pop, state = jit_ask(keys)

    assert pop.genes.values.shape == (32, 10)


def test_genetic_crossover_emitter():
    genome_config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
    crossover = UniformCrossover(crossover_rate=1.0)

    emitter = GeneticCrossoverEmitter(
        _batch_size=32, crossover=crossover, genome_config=genome_config
    )

    # 2 sampling keys + 32 * 1 atomic keys (UniformCrossover uses 1 key)
    assert emitter.num_keys() == 2 + 32 * 1

    total_keys = emitter.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(42), total_keys)

    repertoire = MockRepertoire(jax.random.PRNGKey(0), 100, 10)

    @jax.jit
    def jit_ask(k):
        return emitter.ask(None, repertoire, k)

    pop, state = jit_ask(keys)

    assert pop.genes.values.shape == (32, 10)
