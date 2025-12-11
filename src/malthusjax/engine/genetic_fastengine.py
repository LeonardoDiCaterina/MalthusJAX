"""
Standard Genetic Algorithm Engine.
Refactored for 'Init-Phase Compilation': Resource mapping happens once at initialization.
"""
from functools import partial
from typing import Any, Dict, List, Tuple, Optional
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex

from .base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams, AbstractGenerationOutput, AbstractHook
from .resource_mapper import compute_resource_map, ResourceMap, get_resource_summary
from ..operators.base import BaseMutation, BaseCrossover, BaseSelection
from ..core.fitness.base import BaseEvaluator
from ..core.base import BasePopulation
from ..core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..core.genome.categorical_genome import CategoricalGenomeConfig, CategoricalPopulation


#TODO:  fix key that is not being passed properly
#TODO:  implement entropy buffer properly
#TODO:  add typed docstrings to all methods
#TODO:  optimize the operators with the double vmap trick
#TODO:  implement a proper logging mechanism
#TODO:  rename the traceable decorators to something more meaningful
def traceable(name):
    """
    Correctly wraps a method in jax.named_call for HLO profiling labels.
    """
    def decorator(fn):
        return jax.named_call(fn, name=name)
    return decorator

# --- Host-Side Callback ---
def _host_progress_callback(gen, best_fit):
    """CPU callback for logging progress."""
    print(f"Gen {gen}: Best Fitness = {best_fit:.4f}")

@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """Configuration for Genetic Engine."""
    pass

@struct.dataclass
class GeneticGenerationOutput(AbstractGenerationOutput):
    """KPIs returned by Genetic Engine."""
    random_key: chex.Array
    pass

# --- EXTENDED STATE ---
@struct.dataclass
class GeneticEvolutionState(AbstractEvolutionState):
    """
    State that carries its own execution plan (ResourceMap).
    This allows 'step' to be purely executive, with zero planning overhead.
    """
    # The 'Compiled' Execution Plan
    resource_map: ResourceMap = struct.field(pytree_node=False)

@struct.dataclass
class GeneticEngine(AbstractEngine):
    """
    The High-Performance Genetic Engine.
    
    Architecture: Init-Phase Compilation
    1. init_state: compiles the ResourceMap (allocation plan).
    2. step: executes the plan using pre-calculated slices.
    """
    # Core Components
    genome_config: Any
    evaluator: BaseEvaluator
    selection: BaseSelection
    crossover: BaseCrossover
    mutation: BaseMutation
    
    # Hooks
    hooks: Tuple[AbstractHook] = struct.field(default_factory=tuple)
    enable_progress_bar: bool = struct.field(pytree_node=False, default=False)
    _entropy_buffer: List[Any] = struct.field(pytree_node=False, default_factory=list)
    
    @traceable("Phase_0_allocate_entropy")
    def _allocate_entropy(self, state: GeneticEvolutionState) -> Tuple:
            
            rmap = state.resource_map
            all_keys = jar.split(state.rng_key, rmap.total_rng_budget)
            k_sel_slice = all_keys[rmap.get_key_slice('selection')]
            if rmap.selection.num_keys == 1:
                k_sel = k_sel_slice[0]
            else:
                k_sel = k_sel_slice
                
            k_cross = all_keys[rmap.get_key_slice('crossover')]
            k_mut   = all_keys[rmap.get_key_slice('mutation')]
        
            # Next Key: Always 1, so we explicitly index [0]
            k_next  = all_keys[rmap.get_key_slice('next_key')][0]
            
            return k_sel,k_cross,k_mut,k_next
    
    
    @traceable("Phase_1_Selection_Read")
    def _selection_phase(self, key_selection: chex.Array, population: Any, params: Any):
        """
        Input: Specific key slice for selection.
        Output: Elites (small copy) and Parents (copy).
        """
        # 1. Elitism
        _, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
        elites_genes = population[elite_idx].genes

        # 2. Selection
        selected_idx = self.selection(key_selection, population.fitness)
        parents = population[selected_idx]
        
        return elites_genes, parents
    
    
    @traceable("Phase_1b_Extract_Context")
    def _extract_context(self, state: AbstractEvolutionState) -> AbstractEvolutionState:
        """
        PRUNING STEP.
        Creates a lightweight shadow of the state for operators.
        Removes heavy buffers (Population/Fitness) to break the dependency chain.
        Preserves ALL user-defined scalar fields (extensibility).
        """
        return state.replace(
            population=None,      # Sever dependency on heavy buffer
            # best_genome=None,   # Optional: sever if operators don't need it
            # best_fitness remains available
            # generation remains available
            # custom_fields remain available
        )    
        
    @traceable("Phase_2_Reproduction_Compute")
    def _reproduction_phase(
        self, 
        keys_crossover: chex.Array,
        keys_mutation: chex.Array,
        parents: Any, 
        context_state: AbstractEvolutionState,
        rmap: ResourceMap 
    ):
        """
        Optimized Reproduction Phase.
        
        New Paradigm:
        - No manual reshaping of keys.
        - No explicit vmap over pairs/individuals here.
        - Operators accept FLAT batches of keys and stacked parent genomes.
        - Operators handle internal 'double vmap' scheduling.
        """
        
        # A. CROSSOVER
        # --------------------------------------
        # 1. Prepare Parent Batches
        # Split the selected parents into two groups (P1, P2)
        num_pairs = rmap.crossover.input_count // 2
        p1 = parents[:num_pairs]
        p2 = parents[num_pairs:]
        
        # 2. Execute Crossover
        # Input: 
        #   - key_crossover: Flat tensor (Total_Offspring * Keys_Per_Op, 2)
        #   - p1.genes, p2.genes: Stacked parent batches (Num_Pairs, ...)
        # Output:
        #   - offspring_genes: Flat batch (Total_Offspring, ...)
        
        print("******************************")
        print(f"p1 type: {type(p1)}")
        print("******************************")
        
        offspring_genes = self.crossover(
            keys_crossover, 
            p1, 
            p2, 
            self.genome_config
        )

        # B. MUTATION
        # -------------------------------------
        # 1. Execute Mutation
        # Input:
        #   - key_mutation: Flat tensor (Total_Mutants * Keys_Per_Op, 2)
        #   - offspring_genes: Flat batch from crossover
        # Output:
        #   - mutant_genes: Flat batch (Total_Mutants, ...)
        print("Mutation Keys Shape:", keys_mutation.shape)
        print("Offspring Genes Shape:", jax.tree_util.tree_leaves(offspring_genes)[0].shape)
        mutant_genes = self.mutation(
            keys_mutation, 
            offspring_genes, 
            self.genome_config
        )
        
        return mutant_genes
    
    
    @traceable("Phase_3a_Merge")
    def _merge(self, elites_genes, mutant_genes, old_state: AbstractEvolutionState)-> chex.Array:
        """
        Constructs the new population using the old_state shell.
        """
        target_size = len(old_state.population)
        
        # 1. Truncate 
        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        num_mutants = target_size - num_elites
        
        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants], mutant_genes)
        
        # 2. Merge
        next_genes = jax.tree_util.tree_map(
            lambda e, m: jnp.concatenate([e, m], axis=0),
            elites_genes, mutants_keep
        )
        
        return next_genes
    
    @traceable("Phase_3b_Evaluate")
    def _evaluate(self, new_genes, old_state: AbstractEvolutionState):
        new_population = old_state.population.replace(genes=new_genes)
        evaluated_pop = self.evaluator.evaluate_population(new_population)
        return evaluated_pop
    
    @traceable("Phase_3c_update_hof")
    def _update_hof(self, evaluated_pop: BasePopulation, old_state: AbstractEvolutionState, k_next: chex.Array) -> AbstractEvolutionState:
        # 4. Hall of Fame
        best_idx = jnp.argmax(evaluated_pop.fitness)
        curr_best_fit = evaluated_pop.fitness[best_idx]
        is_new = curr_best_fit > old_state.best_fitness
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            old_state.best_genome, evaluated_pop[best_idx].genes
        )
            # 6. FINALIZE
        next_state = old_state.replace(
            population=evaluated_pop,
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, curr_best_fit, old_state.best_fitness),
            stagnation_counter=jnp.where(is_new, 0, old_state.stagnation_counter + 1),
            generation=old_state.generation + 1,
            rng_key=k_next
        )
        
        return next_state


    @traceable("GeneticEngine_Step")
    def step(self, key: chex.Array, state: AbstractEvolutionState)-> Tuple[chex.Array, AbstractEvolutionState, AbstractGenerationOutput]:

        # 1. KEY ALLOCATION
        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        # 2. READ
        elites, parents = self._selection_phase(k_sel, state.population, self.engine_params)

        # 3. PRUNE
        context = self._extract_context(state)

        # 4. COMPUTE
        mutants = self._reproduction_phase(k_cross, k_mut, parents, context, state.resource_map)

        # 5. WRITE
        next_genes = self._merge(elites, mutants, state)
        
        
        new_pop = self._evaluate(next_genes, state)
        
        final_state = self._update_hof(new_pop, state, k_next)
        
        
        # 7. HOOKS & METRICS
        for hook in self.hooks:
            final_state = hook(final_state, self.engine_params)

        metrics = GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(new_pop.fitness),
            generation=final_state.generation,
            random_key=final_state.rng_key
        )
        
        if self.enable_progress_bar:
            jax.debug.callback(
                lambda g, f: print(f"Gen {g}: {f:.4f}"), 
                final_state.generation, final_state.best_fitness
            )
            
        return k_next, final_state, metrics
    
    def ask(self, state: AbstractEvolutionState) -> BasePopulation:
        """
        Get the current population parameters (genes) to be evaluated.
        
        Args:
            state: Current evolution state.
            
        Returns:
            Population
        """
        # We modify the list *contents*, not the class attribute
        self._entropy_buffer[:] = self._allocate_entropy(state)

        return state.population        
    
    def tell(self, state: GeneticEvolutionState, population: BasePopulation) -> GeneticEvolutionState:
        """
        Update state with externally evaluated population and produce next generation.
        
        This method:
        1. Retrieves pre-allocated keys from the entropy buffer
        2. Updates Hall of Fame with external fitness
        3. Executes Selection -> Reproduction -> Merge pipeline
        
        Args:
            state: Current evolution state
            population: Externally evaluated population with fitness values
            
        Returns:
            Updated state with next generation population
        """
        # 1. RETRIEVE ENTROPY (from sidecar buffer)
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask(). Call ask() first to allocate keys.")
        
        k_sel, k_cross, k_mut, k_next = self._entropy_buffer
        
        # 2. UPDATE STATE WITH EXTERNAL FITNESS
        state = state.replace(population=population)
        
        # 3. UPDATE HALL OF FAME
        best_idx = jnp.argmax(population.fitness)
        curr_best_fit = population.fitness[best_idx]
        is_new = curr_best_fit > state.best_fitness
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            state.best_genome, population[best_idx].genes
        )
        
        state = state.replace(
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, curr_best_fit, state.best_fitness),
            stagnation_counter=jnp.where(is_new, 0, state.stagnation_counter + 1)
        )
        
        # 4. SELECTION PHASE
        elites, parents = self._selection_phase(k_sel, state.population, self.engine_params)
        
        # 5. REPRODUCTION PHASE
        context = self._extract_context(state)
        mutants = self._reproduction_phase(k_cross, k_mut, parents, context, state.resource_map)
        
        # 6. MERGE & CREATE NEXT GENERATION
        next_genes = self._merge(elites, mutants, state)
        next_population = state.population.replace(genes=next_genes)
        
        # 7. FINALIZE STATE
        final_state = state.replace(
            population=next_population,
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        # 9. HOOKS
        for hook in self.hooks:
            final_state = hook(final_state, self.engine_params)
        
        return final_state
        

    # ==========================================
    # 3. INITIALIZATION (Compiler)
    # ==========================================
    
    def init_state(self, rng_key: chex.Array) -> GeneticEvolutionState:
        """
        Compiles the Execution Plan and initializes the population.
        """
        # 1. COMPILE: Compute the static Resource Map
        # This is the "memory allocation" step you asked for.
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.engine_params.pop_size
        )
        
        # Bake input sizes into operators for optimal performance
        # Each operator gets its specific input count from the resource map
        crossover_input_size = rmap.crossover.input_count 
        mutation_input_size = rmap.mutation.input_count
        
        # Log the compilation result (Plan Summary)
        # Note: In JIT, this prints only once at trace time.
        print("*"*60)
        print("Genetic Engine: Execution Plan Compiled")
        print(get_resource_summary(rmap))
        print(f"Operators optimized:")
        print(f"  - Crossover: {crossover_input_size} pairs")
        print(f"  - Mutation: {mutation_input_size} individuals")
        print("*"*60)

        # 2. Initialize Population
        init_pop_key, rng_key = jar.split(rng_key)
        
        if isinstance(self.genome_config, BinaryGenomeConfig):
            pop_cls = BinaryPopulation
        elif isinstance(self.genome_config, RealGenomeConfig):
            pop_cls = RealPopulation
        elif isinstance(self.genome_config, CategoricalGenomeConfig):
            pop_cls = CategoricalPopulation
        else:
            raise ValueError(f"Unsupported config: {type(self.genome_config)}")

        population = pop_cls.init_random(init_pop_key, self.genome_config, self.engine_params.pop_size)
        evaluated_pop = self.evaluator.evaluate_population(population)
        
        # 3. Create State with Embedded Plan
        best_idx = jnp.argmax(evaluated_pop.fitness)
        
        return GeneticEvolutionState(
            population=evaluated_pop,
            best_genome=evaluated_pop[best_idx].genes,
            best_fitness=evaluated_pop.fitness[best_idx],
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0,
            resource_map=rmap
        )