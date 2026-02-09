"""Reproducibility tests across PRNG implementations and key derivation strategies."""

import jax
import jax.random as jr
import pytest

from malthusjax.core.random import create_key, PRNGImpl
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation


def make_engine(prng_impl=PRNGImpl.THREEFRY, key_derivation=None):
    params = GeneticEngineParams(pop_size=32, elitism=1, num_generations=3)
    if key_derivation is not None:
        params = params.replace(key_derivation=key_derivation)
    # Default prng_impl set at dataclass level; tests will pass keys explicitly

    genome_config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    bbob = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=5, maximize=False))

    engine = GeneticEngine(
        engine_params=params,
        genome_config=genome_config,
        evaluator=bbob,
        selection=ElitePoolSelection(num_selections=32, elite_k=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1),
    )

    return engine


@pytest.mark.parametrize("impl", list(PRNGImpl))
def test_same_seed_same_impl_identical_run(impl):
    try:
        key = create_key(1234, impl=impl)
    except ValueError:
        pytest.skip("impl not supported by this JAX build")

    engine = make_engine()
    s1 = engine.init_state(key)
    final1, _, _ = engine.run(s1, compile=False)

    key2 = create_key(1234, impl=impl)
    s2 = engine.init_state(key2)
    final2, _, _ = engine.run(s2, compile=False)

    assert jax.numpy.allclose(final1.population.genes.values, final2.population.genes.values)


@pytest.mark.parametrize("impl_pair", [(PRNGImpl.THREEFRY, PRNGImpl.PHILOX)])
def test_same_seed_different_impl_different_run(impl_pair):
    impl1, impl2 = impl_pair
    try:
        k1 = create_key(42, impl=impl1)
        k2 = create_key(42, impl=impl2)
    except ValueError:
        pytest.skip("impl pair not available in this JAX build")

    engine = make_engine()
    f1, _, _ = engine.run(engine.init_state(k1), compile=False)
    f2, _, _ = engine.run(engine.init_state(k2), compile=False)

    # Likely differ; assert they are not identical
    assert not jax.numpy.allclose(f1.population.genes.values, f2.population.genes.values)


def test_reproducibility_across_key_derivation():
    impl = PRNGImpl.THREEFRY
    try:
        k = create_key(999, impl=impl)
    except ValueError:
        pytest.skip("impl not supported")

    engine_split = make_engine(key_derivation=jax.random.split)  # note: we pass enum earlier normally
    # Use enum types properly: KeyDerivationStrategy is set via params.replace in make_engine above

    # Instead, directly create two engines with different KeyDerivationStrategy
    from malthusjax.engine.resource_mapper import KeyDerivationStrategy

    e_split = make_engine()
    e_split = e_split.replace(engine_params=e_split.engine_params.replace(key_derivation=KeyDerivationStrategy.SPLIT))
    e_fold = e_split.replace(engine_params=e_split.engine_params.replace(key_derivation=KeyDerivationStrategy.FOLD))

    f1, _, _ = e_split.run(e_split.init_state(k), compile=False)
    f1b, _, _ = e_split.run(e_split.init_state(k), compile=False)
    assert jax.numpy.allclose(f1.population.genes.values, f1b.population.genes.values)

    f2, _, _ = e_fold.run(e_fold.init_state(k), compile=False)
    f2b, _, _ = e_fold.run(e_fold.init_state(k), compile=False)
    assert jax.numpy.allclose(f2.population.genes.values, f2b.population.genes.values)

    # The split and fold results may differ but are reproducible individually


def test_jit_does_not_break_reproducibility():
    impl = PRNGImpl.THREEFRY
    try:
        k = create_key(2023, impl=impl)
    except ValueError:
        pytest.skip("impl not supported")

    engine = make_engine()
    state = engine.init_state(k)

    # Compare eager step vs jitted step for one generation
    state1, _ = engine.step(state)
    jit_step = jax.jit(engine.step)
    state2, _ = jit_step(state)

    assert jax.numpy.allclose(state1.population.genes.values, state2.population.genes.values)
