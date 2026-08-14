import jax
import jax.numpy as jnp
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.core.base import BasePopulation
from malthusjax.operators.mutation.real import GaussianMutation, PolynomialMutation

def test_gaussian_mutation_clip():
    """Test Gaussian mutation clipping to bounds."""
    mutation = GaussianMutation(
        num_offspring=1, 
        input_length=1, 
        mutation_strength=10.0,  # Very high to force out of bounds
        clip=True,
    )
    
    config = RealGenomeConfig(shape=(1, 1), bounds=(-1.0, 1.0))
    genome = RealGenome(values=jnp.array([[0.0]]))
    pop = BasePopulation(genes=genome, fitness=jnp.array([[0.0]]), config=None)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, mutation.num_keys((1,)))
    

    
    new_pop = mutation(keys, pop, config=config)
    
    # Because clip=True, values should not exceed bounds
    assert jnp.all(new_pop.genes.values >= -1.0)
    assert jnp.all(new_pop.genes.values <= 1.0)

def test_polynomial_mutation_clip():
    """Test Polynomial mutation clipping."""
    mutation = PolynomialMutation(
        num_offspring=1,
        input_length=1,
        eta=20.0,
        mutation_rate=1.0, # Always mutate
        clip=True,
    )
    
    config = RealGenomeConfig(shape=(1, 1), bounds=(-1.0, 1.0))
    genome = RealGenome(values=jnp.array([[0.0]]))
    pop = BasePopulation(genes=genome, fitness=jnp.array([[0.0]]), config=None)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, mutation.num_keys((1,)))
    

    
    new_pop = mutation(keys, pop, config=config)
    
    # Because clip=True, values should not exceed bounds
    assert jnp.all(new_pop.genes.values >= -1.0)
    assert jnp.all(new_pop.genes.values <= 1.0)

def test_mutation_zero_rate():
    """Test mutation with zero rate."""
    mutation = GaussianMutation(
        num_offspring=1,
        input_length=1,
        mutation_rate=0.0,
    )
    config = RealGenomeConfig(shape=(1, 1), bounds=(-1.0, 1.0))
    genome = RealGenome(values=jnp.array([[0.0]]))
    pop = BasePopulation(genes=genome, fitness=jnp.array([[0.0]]), config=None)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, mutation.num_keys((1,)))
    
    new_pop = mutation(keys, pop, config=config)
    assert jnp.all(new_pop.genes.values == 0.0)

def test_mutation_no_clip():
    """Test mutation without clipping."""
    mutation = GaussianMutation(
        num_offspring=1,
        input_length=1,
        mutation_strength=10.0,
        mutation_rate=1.0,
        clip=False,
    )
    config = RealGenomeConfig(shape=(1, 1), bounds=(-1.0, 1.0))
    genome = RealGenome(values=jnp.array([[0.0]]))
    pop = BasePopulation(genes=genome, fitness=jnp.array([[0.0]]), config=None)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, mutation.num_keys((1,)))
    
    new_pop = mutation(keys, pop, config=config)
    assert not jnp.all(jnp.abs(new_pop.genes.values) <= 1.0)
