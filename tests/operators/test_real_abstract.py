import jax
import jax.numpy as jnp
import pytest
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

from malthusjax.operators.crossover.real import (
    UniformCrossover, UniformCrossover_injection,
    BlendCrossover, BlendCrossover_injection,
    SimulatedBinaryCrossover, SimulatedBinaryCrossover_injection,
    BinomialCrossover, BinomialCrossover_injection
)

from malthusjax.operators.mutation.real import (
    GaussianMutation, GaussianMutation_injection,
    BallMutation, BallMutation_injection,
    PolynomialMutation, PolynomialMutation_injection
)
from malthusjax.engine.schedules import ScheduleType

def test_crossover_coverage():
    config = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0), dtype=jnp.float32)
    p1 = RealGenome(values=jnp.array([1.0, 1.0]))
    p2 = RealGenome(values=jnp.array([-1.0, -1.0]))
    
    crossovers = [
        (UniformCrossover(), 1),
        (UniformCrossover_injection().set_input_length(1), 1),
        (BlendCrossover(), 2),
        (BlendCrossover_injection().set_input_length(1), 2),
        (SimulatedBinaryCrossover(), 3),
        (SimulatedBinaryCrossover_injection().set_input_length(1), 3),
        (BinomialCrossover(), 1),
        (BinomialCrossover_injection().set_input_length(1), 1)
    ]
    
    master_key = jax.random.PRNGKey(42)
    for op, expected_keys in crossovers:
        assert op.num_keys_per_atomic_operation == expected_keys
        
        # Test generation and recombination
        if "injection" in op.__class__.__name__.lower():
            # Test ValueError
            with pytest.raises(ValueError):
                op_uninit = op.__class__()
                op_uninit._generate_noise(master_key, config)

            # For injection, it splits the key based on input_length * num_offspring internally
            noise = op._generate_noise(master_key, config)
            # Extract the first row of noise for recombine_one
            if isinstance(noise, tuple):
                noise_single = tuple(n[0] for n in noise)
            else:
                noise_single = noise[0]
            out = op._recombine_one(p1, p2, noise_single, config)
        else:
            # For regular, we pass pre-split keys
            keys = jax.random.split(master_key, op.num_keys_per_atomic_operation)
            noise = op._generate_noise(keys, config)
            out = op._recombine_one(p1, p2, noise, config)
            
        assert isinstance(out, RealGenome)
        assert out.values.shape == config.shape

def test_mutation_coverage():
    config = RealGenomeConfig(shape=(2,), bounds=(-5.0, 5.0), dtype=jnp.float32)
    g = RealGenome(values=jnp.array([0.0, 0.0]))
    
    # Test mutations with clipping and scheduling
    mutations = [
        (GaussianMutation(clip=True), 2),
        (GaussianMutation(clip=True, schedule_type=ScheduleType.LINEAR_DECAY), 2),
        (GaussianMutation_injection(clip=True).set_input_length(1), 2),
        (GaussianMutation_injection(clip=True, schedule_type=ScheduleType.LINEAR_DECAY).set_input_length(1), 2),
        (BallMutation(clip=True), 3),
        (BallMutation(clip=True, schedule_type=ScheduleType.LINEAR_DECAY), 3),
        (BallMutation_injection(clip=True).set_input_length(1), 3),
        (BallMutation_injection(clip=True, schedule_type=ScheduleType.LINEAR_DECAY).set_input_length(1), 3),
        (PolynomialMutation(clip=True), 2),
        (PolynomialMutation_injection(clip=True).set_input_length(1), 2)
    ]
    
    master_key = jax.random.PRNGKey(42)
    for op, expected_keys in mutations:
        assert op.num_keys_per_atomic_operation == expected_keys
        
        if "injection" in op.__class__.__name__.lower():
            # Test ValueError
            with pytest.raises(ValueError):
                op_uninit = op.__class__()
                op_uninit._generate_noise(master_key, config)

            noise = op._generate_noise(master_key, config, generation=10)
            noise_single = noise[0]
            out = op._mutate_one(g, noise_single, config)
        else:
            keys = jax.random.split(master_key, op.num_keys_per_atomic_operation)
            noise = op._generate_noise(keys, config)
            out = op._mutate_one(g, noise, config)
            
        assert isinstance(out, RealGenome)
        assert out.values.shape == config.shape
