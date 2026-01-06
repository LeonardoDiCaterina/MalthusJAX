"""
Minimal test suite for ablation operators.
Core tests: num_keys()=1, runtime-dynamic keys, output correctness.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation.ablation_mutation import (
    AblationGaussianMutation,
    AblationBallMutation,
    AblationPolynomialMutation,
)
from malthusjax.operators.crossover.ablation_crossover import (
    AblationUniformCrossover,
    AblationBlendCrossover,
    AblationSimulatedBinaryCrossover,
)
from malthusjax.operators.selection.ablation_selection import AblationElitePoolSelection


@pytest.fixture
def config_20d():
    return RealGenomeConfig(length=20, bounds=(-5.12, 5.12))


@pytest.fixture
def config_5d():
    return RealGenomeConfig(length=5, bounds=(-5.12, 5.12))


@pytest.fixture
def pop_100_20d(config_20d):
    key = jar.PRNGKey(42)
    genes = RealGenome.create_population(key, config_20d, 100)
    return RealPopulation(genes=genes, fitness=jnp.ones(100), config=config_20d)


@pytest.fixture
def pop_50_5d(config_5d):
    key = jar.PRNGKey(42)
    genes = RealGenome.create_population(key, config_5d, 50)
    return RealPopulation(genes=genes, fitness=jnp.ones(50), config=config_5d)


@pytest.fixture
def base_key():
    return jar.PRNGKey(123)


# ============================================================================
# MUTATION TESTS
# ============================================================================
class TestMutations:
    """Test all ablation mutation operators."""

    @pytest.mark.parametrize("OpClass", [
        AblationGaussianMutation,
        AblationBallMutation,
        AblationPolynomialMutation,
    ])
    def test_num_keys_returns_one(self, OpClass):
        """All ablation mutations should return num_keys()=1."""
        op = OpClass()
        assert op.num_keys((100, 20)) == 1

    def test_gaussian_mutation_basic(self, pop_100_20d, config_20d, base_key):
        """Gaussian mutation should produce valid RealPopulation."""
        op = AblationGaussianMutation(num_offspring=1, mutation_rate=0.1)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20
        # Check bounds are respected
        assert jnp.all(result.genes.values >= config_20d.bounds[0])
        assert jnp.all(result.genes.values <= config_20d.bounds[1])

    def test_ball_mutation_basic(self, pop_100_20d, config_20d, base_key):
        """Ball mutation should produce valid RealPopulation."""
        op = AblationBallMutation(num_offspring=1)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20

    def test_polynomial_mutation_basic(self, pop_100_20d, config_20d, base_key):
        """Polynomial mutation should produce valid RealPopulation."""
        op = AblationPolynomialMutation(num_offspring=1, eta=20.0)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20

    def test_mutation_jit_compatible(self, pop_50_5d, config_5d, base_key):
        """Mutations should be JIT-compilable."""
        op = AblationGaussianMutation(num_offspring=1)
        jitted = jax.jit(op)
        keys = jar.split(base_key, 1)
        result = jitted(keys, pop_50_5d, config_5d)
        
        assert isinstance(result, RealPopulation)

    def test_different_seeds_produce_different_mutations(self, pop_50_5d, config_5d, base_key):
        """Different seeds should produce different results."""
        op1 = AblationGaussianMutation(num_offspring=1, seed=42)
        op2 = AblationGaussianMutation(num_offspring=1, seed=99)
        keys = jar.split(base_key, 1)
        
        result1 = op1(keys, pop_50_5d, config_5d)
        result2 = op2(keys, pop_50_5d, config_5d)
        
        # Different seeds should produce at least somewhat different results
        assert not jnp.allclose(result1.genes.values, result2.genes.values)


# ============================================================================
# CROSSOVER TESTS
# ============================================================================
class TestCrossovers:
    """Test all ablation crossover operators."""

    @pytest.mark.parametrize("OpClass", [
        AblationUniformCrossover,
        AblationBlendCrossover,
        AblationSimulatedBinaryCrossover,
    ])
    def test_num_keys_returns_one(self, OpClass):
        """All ablation crossovers should return num_keys()=1."""
        op = OpClass()
        assert op.num_keys((100, 20)) == 1

    def test_uniform_crossover_basic(self, pop_100_20d, config_20d, base_key):
        """Uniform crossover should produce valid output."""
        op = AblationUniformCrossover(num_offspring=1, crossover_rate=0.5)
        keys = jar.split(base_key, 1)
        # Crossover takes two populations
        result = op(keys, pop_100_20d, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20

    def test_blend_crossover_basic(self, pop_100_20d, config_20d, base_key):
        """Blend crossover should produce valid output."""
        op = AblationBlendCrossover(num_offspring=1, alpha=0.5)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20

    def test_simulated_binary_crossover_basic(self, pop_100_20d, config_20d, base_key):
        """Simulated binary crossover should produce valid output."""
        op = AblationSimulatedBinaryCrossover(num_offspring=1, eta=15.0)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, pop_100_20d, config_20d)
        
        assert isinstance(result, RealPopulation)
        assert result.genes.values.shape[1] == 20

    def test_crossover_bounds_respected(self, pop_100_20d, config_20d, base_key):
        """Crossover should respect bounds."""
        op = AblationUniformCrossover(num_offspring=1, crossover_rate=1.0)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d, pop_100_20d, config_20d)
        
        assert jnp.all(result.genes.values >= config_20d.bounds[0])
        assert jnp.all(result.genes.values <= config_20d.bounds[1])

    def test_crossover_jit_compatible(self, pop_50_5d, config_5d, base_key):
        """Crossover should be JIT-compilable."""
        op = AblationUniformCrossover(num_offspring=1)
        jitted = jax.jit(op)
        keys = jar.split(base_key, 1)
        result = jitted(keys, pop_50_5d, pop_50_5d, config_5d)
        
        assert isinstance(result, RealPopulation)


# ============================================================================
# SELECTION TESTS
# ============================================================================
class TestSelection:
    """Test ablation selection operator."""

    def test_num_keys_returns_one(self):
        """Selection should request exactly 1 key."""
        op = AblationElitePoolSelection(num_selections=50, elite_k=10)
        assert op.num_keys((100,)) == 1

    def test_basic_selection(self, pop_100_20d, base_key):
        """Selection should return correct number of indices."""
        op = AblationElitePoolSelection(num_selections=50, elite_k=10)
        keys = jar.split(base_key, 1)
        result = op(keys, pop_100_20d)
        
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (50,)
        assert jnp.all(result >= 0)
        assert jnp.all(result < 100)

    def test_elite_bias_exists(self, config_20d, base_key):
        """Elite individuals should be selected more frequently."""
        # Create population with ranked fitness
        key = jar.PRNGKey(42)
        genes = RealGenome.create_population(key, config_20d, 100)
        fitness = jnp.arange(100.0)  # 0-99, worst to best
        pop = RealPopulation(genes=genes, fitness=fitness, config=config_20d)
        
        op = AblationElitePoolSelection(num_selections=1000, elite_k=10)
        keys = jar.split(base_key, 1)
        selected = op(keys, pop)
        
        # Top 10 (indices 90-99) should appear frequently
        top_k_count = jnp.sum(selected >= 90)
        assert top_k_count > 100  # More than random chance

    def test_selection_jit_compatible(self, pop_50_5d, base_key):
        """Selection should be JIT-compilable."""
        op = AblationElitePoolSelection(num_selections=25, elite_k=5)
        jitted = jax.jit(op)
        keys = jar.split(base_key, 1)
        result = jitted(keys, pop_50_5d)
        
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (25,)


# ============================================================================
# INTEGRATION TEST
# ============================================================================
class TestRuntimeDynamicKeys:
    """Verify ablation operators use runtime-dynamic keys (not constant-folded)."""

    def test_mutation_keys_runtime_dynamic(self, pop_50_5d, config_5d):
        """Different input keys should produce different mutations."""
        op = AblationGaussianMutation(num_offspring=1, mutation_rate=0.5, seed=42)
        
        key1 = jar.PRNGKey(0)
        key2 = jar.PRNGKey(1)
        keys1 = jar.split(key1, 1)
        keys2 = jar.split(key2, 1)
        
        result1 = op(keys1, pop_50_5d, config_5d)
        result2 = op(keys2, pop_50_5d, config_5d)
        
        # Different keys -> different results (not constant-folded)
        assert not jnp.allclose(result1.genes.values, result2.genes.values, rtol=1e-5)

    def test_all_ablation_operators_have_num_keys_one(self):
        """All ablation operators should have num_keys()=1."""
        ops = [
            AblationGaussianMutation(),
            AblationBallMutation(),
            AblationPolynomialMutation(),
            AblationUniformCrossover(),
            AblationBlendCrossover(),
            AblationSimulatedBinaryCrossover(),
            AblationElitePoolSelection(num_selections=50, elite_k=10),
        ]
        
        for op in ops:
            assert op.num_keys((100, 20)) == 1, f"{type(op).__name__} should have num_keys()=1"
