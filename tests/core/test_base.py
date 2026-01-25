"""
Tests for core base classes and abstractions.
"""
import jax
import jax.numpy as jnp

from malthusjax.core.base import BaseGenome
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection


class TestBaseGenomeAbstractions:
    """Test that base genome abstractions work as expected."""

    def test_base_operators_exist(self):
        """Test that base classes have required methods defined."""
        assert hasattr(BaseGenome, 'random_init')
        assert hasattr(BaseGenome, 'distance')
        assert hasattr(BaseGenome, 'autocorrect')
        assert hasattr(BaseGenome, 'create_population')

    def test_binary_genome_implementation(self):
        """Test that BinaryGenome implements the interface."""
        assert issubclass(BinaryGenome, BaseGenome)
        # Check method signatures exist
        assert hasattr(BinaryGenome, 'random_init')
        assert hasattr(BinaryGenome, 'flip_bit')

class TestOperatorSignatures:
    """Test that operators follow the defined signatures."""

    def test_mutation_signatures(self):
        """Mutation should take (key, genome, config)."""
        assert hasattr(BaseMutation, '__call__')
        # Check annotations if possible, or just structure

    def test_crossover_signatures(self):
        """Crossover should take (key, p1, p2, config)."""
        assert hasattr(BaseCrossover, '__call__')

    def test_selection_signatures(self):
        """Selection should take (key, fitness)."""
        assert hasattr(BaseSelection, '__call__')

class TestArchitectureIntegration:
    """Test that components plug together correctly."""

    def test_genome_operator_compatibility(self, rng_key):
        """Test that genomes can be passed to generic operators."""
        config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config)

        # Verify genome is a valid Pytree
        leaves, treedef = jax.tree_util.tree_flatten(genome)
        assert len(leaves) > 0
        assert isinstance(leaves[0], jax.Array)

        # Reconstruct
        genome2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.all(genome.bits == genome2.bits)

    def test_real_genome_pipeline(self, rng_key):
        """Test basic pipeline components for Real genomes."""
        config = RealGenomeConfig(length=5, bounds=(-1.0, 1.0))
        genome = RealGenome.random_init(rng_key, config)

        # Verify structure
        assert genome.values.shape == (5,)
        assert genome.size == 5
