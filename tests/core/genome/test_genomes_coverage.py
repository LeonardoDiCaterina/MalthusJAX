import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
)
from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig, LinearPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.core.genome.series_genome import (
    FourierBasis,
    SeriesGenome,
    SeriesGenomeConfig,
)
from malthusjax.core.genome.tensorneat_genome import (
    TensorNeatGenome,
    TensorNeatGenomeConfig,
)


def test_real_genome():
    config = RealGenomeConfig(shape=(4,), bounds=(-1.0, 1.0))
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, size=5)

    # Test random_init
    gen = RealGenome.random_init(key, config)
    assert gen.size == 4
    assert gen.shape == (4,)

    # Test autocorrect
    gen_oob = RealGenome(values=jnp.array([-2.0, 0.0, 2.0, 0.0]))
    gen_clamped = gen_oob.autocorrect(config)
    assert jnp.allclose(gen_clamped.values, jnp.array([-1.0, 0.0, 1.0, 0.0]))

    # Test distance
    gen2 = RealGenome(values=jnp.array([1.0, 0.0, -1.0, 0.0]))
    d_euclidean = gen_clamped.distance(gen2, metric="euclidean")
    d_manhattan = gen_clamped.distance(gen2, metric="manhattan")
    d_hamming = gen_clamped.distance(gen2, metric="hamming")

    with pytest.raises(ValueError):
        gen_clamped.distance(gen2, metric="unknown")

    # Test magnitude and normalize
    mag = gen_clamped.magnitude()
    norm_gen = gen_clamped.normalize()

    # Test add_noise
    noisy_gen = gen_clamped.add_noise(key, noise_std=0.1)

    # Test from_tensor
    gen_from_tensor = RealGenome.from_tensor(jnp.ones(4))
    assert gen_from_tensor.values.shape == (4,)


def test_binary_genome():
    config = BinaryGenomeConfig(shape=(4,))
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, size=5)

    gen = BinaryGenome.random_init(key, config)
    assert gen.size == 4
    assert gen.shape == (4,)

    gen_bool = BinaryGenome(values=jnp.array([1, 0, 1, 0]))

    gen2 = BinaryGenome(values=jnp.array([0, 0, 1, 1]))
    d_hamming = gen_bool.distance(gen2, metric="hamming")

    with pytest.raises(ValueError):
        gen_bool.distance(gen2, metric="unknown")

    gen_from_tensor = BinaryGenome.from_tensor(jnp.zeros(4, dtype=jnp.int32))

    # Test legacy length
    config_legacy = BinaryGenomeConfig(length=10)
    assert config_legacy.resolved_shape == (10,)

    # Test autocorrect
    gen_oob = BinaryGenome(values=jnp.array([-1, 2, 0, 1]))
    gen_clamped = gen_oob.autocorrect(config)

    # Test helpers
    val_int = gen_bool.to_int(msb_first=True)
    val_int_lsb = gen_bool.to_int(msb_first=False)
    ones = gen_bool.count_ones()
    flipped = gen_bool.flip_bit(1)

    rep = repr(gen_bool)

    # Traced repr
    def test_traced(x):
        _ = repr(BinaryGenome(values=x))
        return x

    jax.jit(test_traced)(jnp.zeros(5))


def test_categorical_genome():
    config = CategoricalGenomeConfig(shape=(4,), num_categories=3)
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, size=5)

    gen = CategoricalGenome.random_init(key, config)
    assert gen.size == 4
    assert gen.shape == (4,)

    gen_oob = CategoricalGenome(values=jnp.array([-1, 0, 3, 2]))
    gen_clamped = gen_oob.autocorrect(config)
    assert jnp.all((gen_clamped.values >= 0) & (gen_clamped.values < 3))

    gen2 = CategoricalGenome(values=jnp.array([0, 0, 0, 0]))
    d_hamming = gen_clamped.distance(gen2, metric="hamming")

    with pytest.raises(ValueError):
        gen_clamped.distance(gen2, metric="unknown")

    gen_from_tensor = CategoricalGenome.from_tensor(jnp.zeros(4, dtype=jnp.int32))

    # Test helpers
    is_perm = gen_clamped.is_permutation()
    perm = gen_clamped.to_permutation(config)
    swapped = gen_clamped.swap_positions(0, 1)
    count = gen_clamped.count_category(0)

    rep = repr(gen_clamped)

    def test_traced_cat(x):
        _ = repr(CategoricalGenome(values=x))
        return x

    jax.jit(test_traced_cat)(jnp.zeros(5))


def test_linear_genome():
    config = LinearGenomeConfig(length=4, num_inputs=2, num_ops=10, max_arity=2)
    key = jax.random.PRNGKey(0)
    pop = LinearPopulation.init_random(key, config, size=5)

    gen = LinearGenome.random_init(key, config)
    assert gen.size == 4
    assert gen.shape == (4, 2)

    gen_oob = LinearGenome(
        ops=jnp.array([-1, 5, 12, 5]), args=jnp.array([[0, 0], [3, 3], [0, 0], [1, 1]])
    )
    gen_clamped = gen_oob.autocorrect(config)

    gen2 = LinearGenome(
        ops=jnp.array([0, 0, 0, 0]), args=jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    )
    d_hamming = gen_clamped.distance(gen2, metric="hamming")
    d_euc = gen_clamped.distance(gen2, metric="euclidean")

    with pytest.raises(ValueError):
        gen_clamped.distance(gen2, metric="unknown")

    rep = gen_clamped.render(config)

    gen_from_tensor = LinearGenome.from_tensor(
        (jnp.zeros(4, dtype=jnp.int32), jnp.zeros((4, 2), dtype=jnp.int32))
    )


def test_tensorneat_genome():
    config = TensorNeatGenomeConfig(max_nodes=5, max_conns=10)

    with pytest.raises(NotImplementedError):
        TensorNeatGenome.random_init(jax.random.PRNGKey(0), config)

    gen = TensorNeatGenome(values=(jnp.zeros((5,)), jnp.zeros((10,))))
    gen2 = TensorNeatGenome(values=(jnp.ones((5,)), jnp.ones((10,))))

    d = gen.distance(gen2, metric="any")
    gen_clamped = gen.autocorrect(config)

    gen_from_tensor = TensorNeatGenome.from_tensor((jnp.zeros((5,)), jnp.zeros((10,))))
    assert gen_from_tensor.size == 15
    assert gen_from_tensor.shape == ()


def test_series_genome():
    from malthusjax.core.genome.series_genome import (
        ChebyshevBasis,
        MonomialBasis,
    )

    config = SeriesGenomeConfig(n_dims=2, n_coeffs=3, basis=FourierBasis())
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, size=5)

    gen = SeriesGenome.random_init(key, config)
    assert gen.size == 6
    assert gen.shape == (2, 3)

    gen_oob = SeriesGenome(values=jnp.array([[-3.0, 0.0, 3.0], [0.0, -3.0, 3.0]]))
    gen_clamped = gen_oob.autocorrect(config)

    gen2 = SeriesGenome(values=jnp.zeros((2, 3)))
    d_frob = gen_clamped.distance(gen2, metric="frobenius")
    d_man = gen_clamped.distance(gen2, metric="manhattan")
    d_ham = gen_clamped.distance(gen2, metric="hamming")

    with pytest.raises(ValueError):
        gen_clamped.distance(gen2, metric="unknown")

    eval_val = gen_clamped.eval(0.5, config)
    eval_deriv_val = gen_clamped.eval_deriv(0.5, config)

    mag = gen_clamped.magnitude()
    norm_gen = gen_clamped.normalize()

    gen_from_tensor = SeriesGenome.from_tensor(jnp.zeros((2, 3)))

    # Test bases
    f_basis = FourierBasis()
    c_basis = ChebyshevBasis()
    m_basis = MonomialBasis()

    assert f_basis.name == "fourier"
    assert c_basis.name == "chebyshev"
    assert m_basis.name == "monomial"

    coeffs = jnp.array([1.0, 0.5, 0.2])

    f_basis.eval(0.5, coeffs)
    f_basis.eval_deriv(0.5, coeffs)
    f_basis.eval_deriv_nth(0.5, coeffs, 2)
    with pytest.raises(ValueError):
        f_basis.eval_deriv_nth(0.5, coeffs, 0)

    c_basis.eval(0.5, coeffs)
    c_basis.eval_deriv(0.5, coeffs)

    m_basis.eval(0.5, coeffs)
    m_basis.eval_deriv(0.5, coeffs)

    # Validation errors
    with pytest.raises(ValueError):
        SeriesGenomeConfig(n_dims=0).validate()
    with pytest.raises(ValueError):
        SeriesGenomeConfig(n_coeffs=0).validate()
    with pytest.raises(ValueError):
        SeriesGenomeConfig(bounds=(1.0, -1.0)).validate()
    with pytest.raises(ValueError):
        SeriesGenomeConfig(n_coeffs=2, basis=FourierBasis()).validate()
