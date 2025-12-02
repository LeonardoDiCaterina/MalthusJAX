import pytest
import jax
import jax.numpy as jnp
import jax.random as jar
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.operators.mutation.categorical import CategoricalFlipMutation, RandomCategoryMutation

@pytest.fixture
def cat_genome():
    # 10 genes, categories 0-4
    return CategoricalGenome(categories=jnp.zeros(10, dtype=jnp.int32))

@pytest.fixture
def config():
    return CategoricalGenomeConfig(length=10, num_categories=5)

@pytest.fixture
def key():
    return jar.PRNGKey(42)

def test_categorical_flip(cat_genome, config, key):
    """Test simple flip mutation."""
    # Force mutation
    mut = CategoricalFlipMutation(mutation_rate=1.0)
    
    new_genome = mut._mutate_one(key, cat_genome, config)
    
    # Check that it changed
    assert not jnp.array_equal(new_genome.categories, cat_genome.categories)
    # Check bounds
    assert jnp.all(new_genome.categories >= 0)
    assert jnp.all(new_genome.categories < config.num_categories)

def test_random_category_mutation_distinct(cat_genome, config, key):
    """
    Test that RandomCategoryMutation forces a value change 
    (unlike flip which might randomly pick the same value).
    """
    mut = RandomCategoryMutation(mutation_rate=1.0)
    
    new_genome = mut._mutate_one(key, cat_genome, config)
    
    # With rate 1.0, EVERY gene should change to a different category
    # Because _mutate_one ensures new_val != old_val
    diffs = new_genome.categories != cat_genome.categories
    assert jnp.all(diffs)

def test_no_mutation(cat_genome, config, key):
    """Test rate=0.0."""
    mut = CategoricalFlipMutation(mutation_rate=0.0)
    new_genome = mut._mutate_one(key, cat_genome, config)
    assert jnp.array_equal(new_genome.categories, cat_genome.categories)