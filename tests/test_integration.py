"""
Integration tests for the full evolutionary pipeline.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

# 1. Import Core Components (Exposed at Top Level)
from malthusjax import (
    BinaryGenome, BinaryGenomeConfig, BinaryPopulation,
    BinarySumEvaluator, BinarySumConfig
)

# 2. Import The Library Namespace (For Operators)
import malthusjax as mjx

class TestBinaryEvolutionPipeline:
    def test_single_generation_binary(self, rng_key):
        # Setup
        binary_genome_config = BinaryGenomeConfig(length=10)
        pop_size = 10
        k1, k2, k3, k4 = jr.split(rng_key, 4)
        
        # 1. Init
        population = BinaryPopulation.init_random(k1, binary_genome_config, pop_size)
        assert len(population) == pop_size

        # 2. Eval
        eval_config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=eval_config, data=None)
        evaluated_pop = evaluator.evaluate_population(population)
        fitness = evaluated_pop.fitness

        # 3. Select (Using Clean Namespace)
        # mjx.selection.Tournament matches your new __init__.py
        selector = mjx.selection.Tournament(num_selections=pop_size, tournament_size=3)
        selected_indices = selector(k2, fitness)
        parents = population[selected_indices]

        # 4. Crossover (Batch-First)
        half = pop_size // 2
        p1 = parents[:half]
        p2 = parents[half:]
        
        # mjx.crossover.Uniform
        crossover = mjx.crossover.Uniform(num_offspring=2, crossover_rate=0.5)
        
        offspring_batch = crossover(k3, p1.genes, p2.genes, binary_genome_config)
        
        # FIX: Reshape explicitly (Batch -> Pop)
        # (Half, 2, Length) -> (Pop, Length)
        flat_bits = offspring_batch.bits.reshape(pop_size, binary_genome_config.length)
        offspring_genome = BinaryGenome(bits=flat_bits)
        
        assert flat_bits.shape == (pop_size, binary_genome_config.length)

        # 5. Mutation (Batch-First)
        # mjx.mutation.BitFlip
        mutator = mjx.mutation.BitFlip(num_offspring=1, mutation_rate=0.1)
        
        mutated_batch = mutator(k4, offspring_genome, binary_genome_config)
        
        # FIX: Reshape explicitly (1, Pop, Length) -> (Pop, Length)
        final_bits = mutated_batch.bits.reshape(pop_size, binary_genome_config.length)
        
        assert final_bits.shape == (pop_size, binary_genome_config.length)

    @pytest.mark.jit
    def test_jit_compiled_binary_generation(self, rng_key):
        """Verify JIT compilation of the loop."""
        config = BinaryGenomeConfig(length=10)
        pop_size = 20
        
        # Bake operators using Clean Namespace
        selector = mjx.selection.Tournament(num_selections=pop_size, tournament_size=3)
        crossover = mjx.crossover.Uniform(num_offspring=2, crossover_rate=0.5)
        mutator = mjx.mutation.BitFlip(num_offspring=1, mutation_rate=0.01)

        @jax.jit
        def evolution_step(key, current_bits, fitness):
            k_sel, k_cross, k_mut = jr.split(key, 3)
            
            # Select
            indices = selector(k_sel, fitness)
            selected_bits = current_bits[indices]
            
            # Crossover
            half = pop_size // 2
            p1 = BinaryGenome(bits=selected_bits[:half])
            p2 = BinaryGenome(bits=selected_bits[half:])
            
            off_gen = crossover(k_cross, p1, p2, config)
            # Flatten: (Half, 2, L) -> (Pop, L)
            off_bits = off_gen.bits.reshape(pop_size, config.length)
            
            # Mutate
            off_gen_obj = BinaryGenome(bits=off_bits)
            mut_gen = mutator(k_mut, off_gen_obj, config)
            
            # Explicit reshape instead of squeeze
            return mut_gen.bits.reshape(pop_size, config.length)

        # Run
        pop = BinaryPopulation.init_random(rng_key, config, pop_size)
        fitness = jnp.zeros(pop_size)
        new_bits = evolution_step(rng_key, pop.genes.bits, fitness)
        
        assert new_bits.shape == (pop_size, config.length)

    def test_large_scale_integration(self, rng_key):
        pop_size = 100
        length = 50
        config = BinaryGenomeConfig(length=length)
        pop = BinaryPopulation.init_random(rng_key, config, pop_size)
        
        assert pop.genes.bits.shape == (pop_size, length)
        
    def test_multi_operator_compatibility(self, rng_key):
        config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config)
        
        # Clean Namespace Usage
        op1 = mjx.mutation.BitFlip(num_offspring=1, mutation_rate=0.1)
        op2 = mjx.mutation.Scramble(num_offspring=1, mutation_rate=1.0)
        
        # Chain
        mutated1 = op1(rng_key, genome, config)
        child1 = BinaryGenome(bits=mutated1.bits.reshape(1, -1)[0]) 
        
        mutated2 = op2(rng_key, child1, config)
        
        assert mutated2.bits.shape == (1, config.length)