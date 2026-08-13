import time
import jax
import jax.numpy as jnp
import functools
from flax import struct
from typing import Any, Tuple, List, Optional, cast

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.engine.base import (
    AbstractEvolutionState,
    AbstractGenerationOutput,
)
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEvolutionState,
    GeneticGenerationOutput,
    GeneticEngineParams,
    TrackBest,
    TrackMetrics,
)

@functools.lru_cache(maxsize=128)
def _get_native_fast_kernel(params, compile_jit: bool = True, return_history: bool = True):
    def _evolve_flat(engine, initial_state, flat_state):
        def flat_step(carry, _):
            c_values, c_fitness, c_best_genes, c_best_fit, c_key, c_gen = carry
            
            dummy_state = initial_state.replace(rng_key=c_key)
            k_sel_slice, k_cross, k_mut, k_next = engine._allocate_entropy(dummy_state)
            
            sel_op = initial_state.operators.selection.set_num_selections(params.pop_size) if hasattr(initial_state.operators.selection, 'set_num_selections') else initial_state.operators.selection
            
            dummy_pop = initial_state.population.replace(
                genes=initial_state.population.genes.replace(values=c_values),
                fitness=c_fitness
            )
            
            k_sel = k_sel_slice[0]
            parent_indices, e1_idx = sel_op(k_sel, dummy_pop)
            
            num_pairs = initial_state.resource_map.crossover.input_count // 2
            p1_idx = parent_indices[:num_pairs]
            p2_idx = parent_indices[num_pairs : num_pairs * 2]
            
            p1_values = c_values[p1_idx]
            p2_values = c_values[p2_idx]
            
            offspring_values = initial_state.operators.crossover.apply_fastpath(
                k_cross, p1_values, p2_values, engine.genome_config, c_gen
            )
            
            offspring_values = offspring_values.reshape((-1,) + offspring_values.shape[2:]) if offspring_values.ndim > 2 else offspring_values
            
            mutant_values = initial_state.operators.mutation.apply_fastpath(
                k_mut, offspring_values, engine.genome_config, c_gen
            )
            
            if sel_op.n_elites > 0:
                elite_values = c_values[e1_idx]
                mutant_values = jax.lax.dynamic_update_slice(mutant_values, elite_values, (0,) * mutant_values.ndim)
            
            new_genes = initial_state.population.genes.replace(values=mutant_values)
            new_pop = initial_state.population.replace(genes=new_genes)
            evaluated_pop = engine.evaluator.evaluate_population(new_pop)
            new_fitness = evaluated_pop.fitness
            
            gen_best_fitness = jnp.min(new_fitness)
            if params.track_best == TrackBest.NONE:
                new_best_fit = c_best_fit
                new_best_genes = c_best_genes
                metric_best = gen_best_fitness
            elif params.track_best == TrackBest.LIGHT:
                new_best_fit = jnp.minimum(gen_best_fitness, c_best_fit)
                new_best_genes = c_best_genes
                metric_best = new_best_fit
            else:
                is_new = gen_best_fitness < c_best_fit
                best_idx = jnp.argmin(new_fitness)
                new_best_fit = jnp.where(is_new, gen_best_fitness, c_best_fit)
                best_cand_values = mutant_values[best_idx]
                new_best_genes = jnp.where(is_new, best_cand_values, c_best_genes)
                metric_best = new_best_fit
            
            new_carry = (mutant_values, new_fitness, new_best_genes, new_best_fit, k_next, c_gen + 1)
            
            if not return_history:
                return new_carry, ()
            
            if params.track_metrics == TrackMetrics.NONE:
                mean_fitness = jnp.zeros((), dtype=jnp.float32)
                std_fitness = jnp.zeros((), dtype=jnp.float32)
            elif params.track_metrics == TrackMetrics.BASIC:
                mean_fitness = jnp.mean(new_fitness)
                std_fitness = jnp.zeros((), dtype=jnp.float32)
            else:
                mean_fitness = jnp.mean(new_fitness)
                std_fitness = jnp.std(new_fitness)

            output = GeneticGenerationOutput(
                best_fitness=metric_best,
                mean_fitness=mean_fitness,
                std_fitness=std_fitness,
                generation=c_gen,
                random_key=k_next,
            )
            
            return new_carry, output
        
        gen_array = jnp.arange(params.num_generations)
        final_carry, history = jax.lax.scan(flat_step, flat_state, gen_array)
        return final_carry, history

    if compile_jit:
        return jax.jit(_evolve_flat, static_argnums=0)
    else:
        return _evolve_flat


class NativeFastEngine(GeneticEngine):
    """
    An optimized subclass of GeneticEngine that executes a purely flat-array
    scan loop, matching EvoSAX's raw performance. 
    
    It unwraps the PyTree population before the scan, calls the `apply_fastpath` 
    methods on the operators, and reconstructs the PyTree at the end.
    """
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    
    def run(
        self,
        initial_state: AbstractEvolutionState[BaseGenome, BasePopulation[Any]],
        time_it: bool = False,
        compile: bool = True,
        verbose: bool = False,
        return_history: bool = True,
    ) -> Tuple[AbstractEvolutionState[BaseGenome, BasePopulation[Any]], Any, Optional[float]]:
        
        state = cast(GeneticEvolutionState, initial_state)
        params = cast(GeneticEngineParams, self.engine_params)
        
        if verbose:
            print(f"Starting NativeFast evolution: {params.num_generations} generations, pop size {params.pop_size}")
            
        evolve_fn = _get_native_fast_kernel(params, compile_jit=compile, return_history=return_history)

        start_time = time.time()
        
        initial_flat_values = state.population.genes.values
        initial_best_values = state.best_genome.values if state.best_genome is not None else initial_flat_values[0]
        
        flat_state_in = (
            initial_flat_values,
            state.population.fitness,
            initial_best_values,
            state.best_fitness,
            state.rng_key,
            state.generation
        )
        
        final_carry, history = evolve_fn(self, state, flat_state_in)
        
        if time_it:
            _ = final_carry[0].block_until_ready()
            
        elapsed = time.time() - start_time if time_it else None
        if verbose and time_it:
            print(f"Done in {elapsed:.3f}s")
            
        f_values, f_fitness, f_best_values, f_best_fit, f_key, f_gen = final_carry
        
        final_pop = state.population.replace(
            genes=state.population.genes.replace(values=f_values),
            fitness=f_fitness
        )
        final_best_genome = state.best_genome.replace(values=f_best_values) if state.best_genome is not None else None
        
        final_state = state.replace(
            population=final_pop,
            best_genome=final_best_genome,
            best_fitness=f_best_fit,
            rng_key=f_key,
            generation=f_gen
        )
        
        return final_state, history, elapsed
