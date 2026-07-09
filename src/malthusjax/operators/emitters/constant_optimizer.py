import jax
import jax.numpy as jnp
import optax
import chex
from typing import Any, Tuple, Optional
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

@struct.dataclass
class ConstantOptimizerState(EmitterState):
    opt_state: Any

class ConstantOptimizationEmitter(BaseEmitter):
    """
    Emitter that refines continuous constants in elite genomes using gradient descent.
    Applies NSGA-II (Non-dominated Sorting) to select the elite genomes based on
    a multi-objective trade-off (e.g., MSE vs. active length).
    """
    
    def __init__(
        self, 
        batch_size: int, 
        evaluator: Any,
        genome_config: Any,
        num_optimization_steps: int = 50,
        learning_rate: float = 0.1,
        emit_frequency: int = 10,
        clip_grad_norm: float = 1.0
    ):
        self._batch_size = batch_size
        self.evaluator = evaluator
        self.genome_config = genome_config
        self.num_optimization_steps = num_optimization_steps
        self.emit_frequency = emit_frequency
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.adam(learning_rate)
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size
        
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def set_input_length(self, length: int) -> 'BaseEmitter':
        return self

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> Optional[EmitterState]:
        # Initialize optax state with dummy constants shape
        # Assuming the population genes have a 'constants' array
        dummy_constants = initial_population.genes.constants[0]
        opt_state = self.optimizer.init(dummy_constants)
        return ConstantOptimizerState(opt_state=opt_state)

    def _fast_non_dominated_sort(self, fitness: chex.Array, complexity: chex.Array) -> chex.Array:
        """
        Computes Pareto front ranks for NSGA-II.
        Since JAX loops are strict, we compute dominance matrix and rank by dominance count.
        """
        pop_size = fitness.shape[0]
        
        # A dominates B if A is strictly better or equal in all objectives, and strictly better in at least one
        # fitness and complexity are meant to be minimized
        f_leq = fitness[:, None] <= fitness[None, :]
        f_less = fitness[:, None] < fitness[None, :]
        
        c_leq = complexity[:, None] <= complexity[None, :]
        c_less = complexity[:, None] < complexity[None, :]
        
        dominates = (f_leq & c_leq) & (f_less | c_less)
        
        # Count how many individuals dominate each individual
        # The Pareto rank roughly correlates with domination count
        domination_count = jnp.sum(dominates, axis=0)
        return domination_count

    def ask(self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array, generation: int = 0, params: Any = None) -> Tuple[BasePopulation, Optional[EmitterState]]:
        """
        Takes the repertoire (population), selects elites via NSGA-II, optimizes their constants,
        and returns them as the new batch.
        """
        population = repertoire  # Treat repertoire as the current population
        
        # 1. NSGA-II Selection
        if hasattr(population.genes, "get_ancestor_sets"):
            # Use true active length of the readout prefix (last row)
            def _get_active_length(genome):
                ancestors = genome.get_ancestor_sets(self.genome_config)
                # Active length is sum of ancestors of the last row + 1 (for itself)
                return jnp.sum(ancestors[-1]) + 1
                
            complexity = jax.vmap(_get_active_length)(population.genes)
        else:
            complexity = jnp.zeros_like(population.fitness)
            
        ranks = self._fast_non_dominated_sort(population.fitness, complexity)
        
        # Get top-k indices (lowest domination count)
        top_indices = jnp.argsort(ranks)[:self._batch_size]
        
        elite_genes = jax.tree_util.tree_map(lambda x: x[top_indices], population.genes)
        elite_fitness = population.fitness[top_indices]
        
        # 2. Gradient Descent on Constants
        def _optimize(genes, state_opt):
            def optimize_constants(single_genome, opt_st):
                def loss_fn(constants):
                    mod_genome = single_genome.replace(constants=constants)
                    X, y = self.evaluator.data
                    preds = self.evaluator.get_program_prediction(mod_genome, X, instruction_idx=-1)
                    mse = jnp.mean(jnp.square(preds - y))
                    return mse
                    
                def step(carry, _):
                    c, os = carry
                    loss, grads = jax.value_and_grad(loss_fn)(c)
                    updates, new_os = self.optimizer.update(grads, os, c)
                    new_c = optax.apply_updates(c, updates)
                    return (new_c, new_os), loss
                    
                (final_constants, final_opt_state), _ = jax.lax.scan(
                    step, 
                    (single_genome.constants, opt_st), 
                    None, 
                    length=self.num_optimization_steps
                )
                
                return final_constants, final_opt_state

            batched_opt_state = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], self._batch_size, axis=0), state_opt)
            optimized_constants, new_batched_opt_state = jax.vmap(optimize_constants)(genes, batched_opt_state)
            optimized_genes = genes.replace(constants=optimized_constants)
            new_opt_state = jax.tree_util.tree_map(lambda x: x[0], new_batched_opt_state)
            return optimized_genes, new_opt_state

        def run_opt(_):
            return _optimize(elite_genes, state.opt_state)
            
        def skip_opt(_):
            return elite_genes, state.opt_state
            
        final_genes, new_opt_state = jax.lax.cond(
            generation % self.emit_frequency == 0,
            run_opt,
            skip_opt,
            operand=None
        )
        
        # Return optimized population
        offspring_pop = population.spawn_offspring(final_genes, fitness=elite_fitness)
        new_state = ConstantOptimizerState(opt_state=new_opt_state)
        
        return offspring_pop, new_state
