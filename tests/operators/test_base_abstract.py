import jax
import jax.numpy as jnp

from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection
from malthusjax.operators.base_injection import BaseCrossover_injection, BaseMutation_injection


class DummyGenome:
    pass


class DummyPopulation:
    def spawn_offspring(self, genes):
        return genes


class DummyMutation(BaseMutation):
    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _mutate_one(self, genome, noise, config, **kwargs):
        return genome

    def _generate_noise(self, keys, config, generation=0):
        return keys


class DummyMutation_injection(BaseMutation_injection):
    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _mutate_one(self, genome, noise, config, **kwargs):
        return genome

    def _generate_noise(self, keys, config, generation=0):
        return keys


class DummyCrossover(BaseCrossover):
    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _generate_noise(self, keys, config, generation=0):
        return keys

    def _recombine_one(self, p1, p2, noise, config, **kwargs):
        return p1


class DummyCrossover_injection(BaseCrossover_injection):
    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _generate_noise(self, keys, config, generation=0):
        return keys

    def _recombine_one(self, p1, p2, noise, config, **kwargs):
        return p1


class DummySelection(BaseSelection):
    @property
    def num_keys_per_atomic_operation(self):
        return 1

    def _select(self, keys, fitness, config=None, **kwargs):
        return jnp.array([0, 1])


def test_base_mutation_methods():
    mutations = [
        (
            DummyMutation(num_offspring=2)
            .set_input_length(10)
            .set_typed_keys(True)
            .set_max_generations(100),
            20,
        ),
        (
            DummyMutation_injection(num_offspring=2)
            .set_input_length(10)
            .set_typed_keys(True)
            .set_max_generations(100),
            1,
        ),
    ]
    for m, exp in mutations:
        assert m.num_keys((10,)) == exp


def test_base_crossover_methods():
    crossovers = [
        (
            DummyCrossover(num_offspring=2)
            .set_input_length(10)
            .set_typed_keys(True)
            .set_max_generations(100),
            20,
        ),
        (
            DummyCrossover_injection(num_offspring=2)
            .set_input_length(10)
            .set_typed_keys(True)
            .set_max_generations(100),
            1,
        ),
    ]
    for c, exp in crossovers:
        assert c.num_keys((10,)) == exp


def test_base_selection_methods():
    sel = DummySelection(num_selections=2).set_input_length(10).set_typed_keys(True).set_n_elites(5)
    assert sel.num_keys((10,)) == 1

    fit = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
    elites = sel.get_elite_indices(fit)
    assert len(elites) == 5

    sel2 = DummySelection(num_selections=2).set_n_elites(0)
    assert len(sel2.get_elite_indices(fit)) == 0

    sel3 = DummySelection(num_selections=2).set_n_elites(10)
    assert len(sel3.get_elite_indices(fit)) == 5

    # test __call__
    keys = jax.random.PRNGKey(0)
    parents, elites = sel(keys, fit)
    assert len(parents) == 2


def test_base_vmap_calls():
    # test __call__ for Mutation and Crossover to hit the JAX vmap branches
    class DummyPop(DummyPopulation):
        def __init__(self, genes):
            self.genes = genes

    pop = DummyPop(genes=jnp.zeros((10, 2)))
    config = None
    keys = jnp.zeros((10, 1, 2), dtype=jnp.uint32)

    mut = DummyMutation(num_offspring=1).set_input_length(10).set_typed_keys(False)
    out_mut = mut(keys, pop, config)
    assert out_mut.shape == (10, 2)

    cross = DummyCrossover(num_offspring=1).set_input_length(10).set_typed_keys(False)
    out_cross = cross(keys, pop, pop, config)
    assert out_cross.shape == (10, 2)

    # test multiple offspring and new typed keys
    keys_typed = jnp.zeros(20, dtype=jnp.uint32)
    mut2 = DummyMutation(num_offspring=2).set_input_length(10).set_typed_keys(True)
    out_mut2 = mut2(keys_typed, pop, config)
    assert out_mut2.shape == (20, 2)

    cross2 = DummyCrossover(num_offspring=2).set_input_length(10).set_typed_keys(True)
    out_cross2 = cross2(keys_typed, pop, pop, config)
    assert out_cross2.shape == (20, 2)

    # test cross_single_pair
    out_single = cross.cross_single_pair(jax.random.PRNGKey(0), jnp.zeros(2), jnp.zeros(2), config)
    assert out_single.shape == (1, 2)
