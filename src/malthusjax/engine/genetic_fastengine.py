"""
Standard Genetic Algorithm Engine.
Refactored for 'Init-Phase Compilation': Resource mapping happens once at initialization.
"""
from typing import Any, Tuple
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex

from .base import AbstractEngine, AbstractEvolutionState, AbstractEngineParams, AbstractGenerationOutput
from .resource_mapper import compute_resource_map, ResourceMap, get_resource_summary, ShardingManager
from ..operators.base import BaseMutation, BaseCrossover, BaseSelection
from ..core.fitness.base import BaseEvaluator
from ..core.base import BasePopulation
from ..core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from ..core.genome.real_genome import RealGenomeConfig, RealPopulation
from ..core.genome.categorical_genome import CategoricalGenomeConfig, CategoricalPopulation

#TODO: update selection doctring

def traceable(name):
    """Correctly wraps a method in jax.named_call for HLO profiling labels."""
    def decorator(fn):
        return jax.named_call(fn, name=name)
    return decorator

@struct.dataclass
class GeneticEngineParams(AbstractEngineParams):
    """Configuration for Genetic Engine."""
    pass

@struct.dataclass
class GeneticGenerationOutput(AbstractGenerationOutput):
    """KPIs returned by Genetic Engine."""
    random_key: chex.Array

@struct.dataclass
class OperatorState:
    """
    Holds the 'baked' operators with their static input sizes frozen.
    This ensures XLA compiles efficient kernels once.
    """
    selection: BaseSelection
    crossover: BaseCrossover
    mutation: BaseMutation

@struct.dataclass
class GeneticEvolutionState(AbstractEvolutionState):
    """
    State that carries its own execution plan (ResourceMap) and optimized tools (OperatorState).
    """
    # The 'Compiled' Execution Plan
    resource_map: ResourceMap = struct.field(pytree_node=False)
    # The 'Baked' Operators (Input sizes frozen)
    operators: OperatorState = struct.field(pytree_node=False)

@struct.dataclass
class GeneticEngine(AbstractEngine):
    """
    The High-Performance Genetic Engine.
    
    Architecture: Init-Phase Compilation
    1. init_state: compiles ResourceMap & Bakes Operators.
    2. step: executes using pre-baked tools from State.
    """
    # Core Blueprints (Used only during Init)
    genome_config: Any
    evaluator: BaseEvaluator
    selection: BaseSelection
    crossover: BaseCrossover
    mutation: BaseMutation
    
    # Hooks & Config
    #hooks: Tuple[AbstractHook] = struct.field(default_factory=tuple)
    enable_progress_bar: bool = struct.field(pytree_node=False, default=False)
    
    # Internal Buffer for ask/tell (tuple for hashability)
    _entropy_buffer: Tuple[Any, ...] = struct.field(pytree_node=False, default=())
    
    def __hash__(self) -> int:
        """Make engine hashable for JIT static_argnums."""
        return id(self)
    
    def __eq__(self, other) -> bool:
        """Identity-based equality for JIT caching consistency."""
        return self is other
    
    @traceable("Phase_0_Allocate_Entropy")
    def _allocate_entropy(self, state: GeneticEvolutionState) -> Tuple:
        rmap = state.resource_map
        all_keys = jar.split(state.rng_key, rmap.total_rng_budget)
        
        # Robust Slicing
        k_sel_slice = all_keys[rmap.get_key_slice('selection')]
        k_cross = all_keys[rmap.get_key_slice('crossover')]
        k_mut   = all_keys[rmap.get_key_slice('mutation')]
        
        # Next Key: Always 1, explicitly index [0] to get PRNGKey (2,)
        k_next  = all_keys[rmap.get_key_slice('next_key')][0]
        
        return k_sel_slice, k_cross, k_mut, k_next
    
    @traceable("Phase_1_Selection_Read")
    def _selection_phase(self, key_selection: chex.Array, population: Any, operators: OperatorState, params: Any) -> Tuple[chex.Array, chex.Array]:
        """
        Input: Specific key slice for selection.

        """
        # 1. Elitism
        _, elite_idx = jax.lax.top_k(population.fitness, params.elitism)
        elites_genes = population[elite_idx].genes

        # 2. Selection (Use Baked Operator)
        selected_idx = operators.selection(key_selection, population)
        parents = population[selected_idx]
        
        return elites_genes, selected_idx 
    
    
    @traceable("Phase_2_Reproduction_Fused")
    def _reproduction_phase(
        self, 
        keys_crossover: chex.Array,
        keys_mutation: chex.Array,
        parent_indices: chex.Array, 
        population: Any,             
        operators: OperatorState,
        rmap: ResourceMap
    ) -> Any: 
        
        # 1. Slice Indices (Cheap metadata op)
        num_pairs = rmap.crossover.input_count // 2 # 9//2=4
        p1_idx = parent_indices[:num_pairs] # 0:4
        p2_idx = parent_indices[num_pairs : num_pairs * 2] #4:8

        # 2. Gather (INSIDE the fusion scope of crossover)
        # XLA sees: Gather -> Crossover. It fuses them.
        p1_pop = population[p1_idx]
        p2_pop = population[p2_idx]
        
        # 2. Execute Crossover
        offspring_pop = operators.crossover(
            keys_crossover, 
            p1_pop, 
            p2_pop, 
            self.genome_config
        )
        # 3. Execute Mutation
        final_pop = operators.mutation(
            keys_mutation, 
            offspring_pop, 
            self.genome_config
        )
        
        return final_pop
    
    @traceable("Phase_3a_Merge")
    def _merge(self, elites_genes, mutant_genes, old_state: AbstractEvolutionState) -> chex.Array:
        """Constructs the new population using the old_state shell."""
        target_size = len(old_state.population)
        
        leaves = jax.tree_util.tree_leaves(elites_genes)
        num_elites = leaves[0].shape[0] if leaves else 0
        num_mutants = target_size - num_elites
        
        mutants_keep = jax.tree_util.tree_map(lambda x: x[:num_mutants], mutant_genes)
        
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
    
    @traceable("Phase_3c_Update_HOF")
    def _update_hof(self, evaluated_pop: BasePopulation, old_state: GeneticEvolutionState, k_next: chex.Array) -> GeneticEvolutionState:
        # Hall of Fame Update
        best_idx = jnp.argmax(evaluated_pop.fitness)
        curr_best_fit = evaluated_pop.fitness[best_idx]
        is_new = curr_best_fit > old_state.best_fitness
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            old_state.best_genome, evaluated_pop[best_idx].genes
        )
        
        # Preserve operators & rmap in the new state!
        next_state = old_state.replace(
            population=evaluated_pop,
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, curr_best_fit, old_state.best_fitness),
            stagnation_counter=jnp.where(is_new, 0, old_state.stagnation_counter + 1),
            generation=old_state.generation + 1,
            rng_key=k_next
            # operators=old_state.operators (Implicitly preserved by replace)
            # resource_map=old_state.resource_map (Implicitly preserved)
        )
        return next_state

    @traceable("GeneticEngine_Step")
    def step(self, state: GeneticEvolutionState) -> Tuple[GeneticEvolutionState, GeneticGenerationOutput]:
        
        # 1. KEY ALLOCATION
        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        # 2. SELECTION (Read)
        # Pass state.operators so it uses the optimized version
        elites, parent_indices = self._selection_phase(k_sel, state.population, state.operators, self.engine_params)
        # 3. REPRODUCTION (Compute)
        # Pass state.operators
        mutants = self._reproduction_phase(
                    k_cross, 
                    k_mut, 
                    parent_indices, 
                    state.population,
                    state.operators, 
                    state.resource_map
                )
        # 4. MERGE (Write)
        next_genes = self._merge(elites, mutants.genes, state)
        
        # 5. EVALUATE
        new_pop = self._evaluate(next_genes, state)
        
        # 6. FINALIZE
        final_state = self._update_hof(new_pop, state, k_next)
        
        # 7. HOOKS & METRICS
        #for hook in self.hooks:
        #   final_state = hook(final_state, self.engine_params)

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
            
        return final_state, metrics

    # ==========================================
    # 3. INITIALIZATION (Compiler)
    # ==========================================
    def init_state(self, rng_key: chex.Array) -> GeneticEvolutionState:
        """
        Compiles the Execution Plan (ResourceMap), Bakes Operators, 
        and Enforces GSPMD Sharding Layout.
        """
        # ==========================================
        # 1. COMPILE: Compute static Resource Map
        # ==========================================
        rmap = compute_resource_map(
            self.selection,
            self.crossover,
            self.mutation,
            self.genome_config,
            self.engine_params.pop_size
        )
        
        # ==========================================
        # 2. SETUP GSPMD: Create the Sharding Manager
        # ==========================================
        # This defines the "Batch-Parallel" layout rule.
        # Ensure ShardingManager is imported from resource_mapper!
        sharding_mgr = ShardingManager(axis_name='batch')
        
        # ==========================================
        # 3. BAKE: Create Optimized Operators
        # ==========================================
        # Freeze input sizes now so step() has zero shape-polymorphism overhead.
        active_sel = self.selection \
            .replace(num_selections=rmap.selection.output_count) \
            .set_input_length(rmap.selection.input_count)
        
        active_cross = self.crossover.set_input_length(rmap.crossover.input_count // 2)
        active_mut   = self.mutation.set_input_length(rmap.mutation.input_count)
        
        op_state = OperatorState(
            selection=active_sel,
            crossover=active_cross,
            mutation=active_mut
        )
        
        # ==========================================
        # 4. INITIALIZE POPULATION (Crucial Step!)
        # ==========================================
        # Determine appropriate Population Class
        if isinstance(self.genome_config, BinaryGenomeConfig):
            pop_cls = BinaryPopulation
        elif isinstance(self.genome_config, RealGenomeConfig):
            pop_cls = RealPopulation
        elif isinstance(self.genome_config, CategoricalGenomeConfig):
            pop_cls = CategoricalPopulation
        else:
            raise ValueError(f"Unsupported config: {type(self.genome_config)}")

        # Split key for initialization
        init_pop_key, rng_key = jar.split(rng_key)

        # Generate standard population (initially on default device/host)
        # THIS DEFINES 'population'
        population = pop_cls.init_random(
            init_pop_key, 
            self.genome_config, 
            self.engine_params.pop_size
        )
        
        # ==========================================
        # 5. ENFORCE SHARDING (The GSPMD Optimization)
        # ==========================================
        # We explicitly move the data to the correct sharded layout immediately.
        
        target_dtype = self.genome_config.dtype  # Ensure this is set to jnp.bfloat16
    
        def _enforce_layout(leaf):
            # 1. Cast to Target Precision (if it's a float)
            if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
                leaf = leaf.astype(target_dtype)

            # 2. Apply Sharding
            if hasattr(leaf, 'shape') and len(leaf.shape) >= 2 and leaf.shape[0] == self.engine_params.pop_size:
                return jax.device_put(leaf, sharding_mgr.matrix_sharding)
            elif hasattr(leaf, 'shape') and len(leaf.shape) == 1 and leaf.shape[0] == self.engine_params.pop_size:
                return jax.device_put(leaf, sharding_mgr.vector_sharding)
            
            return jax.device_put(leaf, sharding_mgr.replicated_sharding)

        # Apply to Genes (The Heavy Payload)
        sharded_genes = jax.tree_util.tree_map(_enforce_layout, population.genes)        
        # Apply to Fitness (The Metadata)
        fitness_casted = population.fitness.astype(target_dtype)
        sharded_fitness = jax.device_put(fitness_casted, sharding_mgr.vector_sharding)
        
        # Reconstruct Population with Sharded Data
        population = population.replace(genes=sharded_genes, fitness=sharded_fitness)

        # ==========================================
        # 6. INITIAL EVALUATION (Distributed)
        # ==========================================
        # Because inputs are explicitly sharded, JAX automatically creates a 
        # sharded computation graph for the evaluator.
        evaluated_pop = self.evaluator.evaluate_population(population)
        
        # ==========================================
        # 7. FINALIZE STATE
        # ==========================================
        best_idx = jnp.argmax(evaluated_pop.fitness)
        
        # Ensure the best genome is replicated (available on all devices)
        best_genome = jax.tree_util.tree_map(
            lambda x: jax.device_put(x[best_idx], sharding_mgr.replicated_sharding),
            evaluated_pop.genes
        )

        return GeneticEvolutionState(
            population=evaluated_pop,
            best_genome=best_genome,
            best_fitness=evaluated_pop.fitness[best_idx],
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0,
            resource_map=rmap,      
            operators=op_state      
        )    
        
    # ==========================================
    # ASK / TELL Interface
    # ==========================================
    def ask(self, state: GeneticEvolutionState) -> Tuple["GeneticEngine", BasePopulation]:
        """Allocate entropy for the next step and return population to evaluate.
        
        Returns:
            Tuple of (engine_with_entropy, population) - the engine carries the entropy buffer.
        """
        entropy = self._allocate_entropy(state)
        # Use object.__setattr__ to bypass flax immutability for internal buffer
        engine_with_entropy = self.replace(_entropy_buffer=entropy)
        return engine_with_entropy, state.population        
    
    def tell(self, state: GeneticEvolutionState, population: BasePopulation) -> GeneticEvolutionState:
        if not self._entropy_buffer:
            raise RuntimeError("tell() called before ask().")
        
        k_sel, k_cross, k_mut, k_next = self._entropy_buffer
        
        # Update with external data
        state = state.replace(population=population)
        
        # HOF Update (Partial)
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
        
        # Pipeline Execution using State Operators
        elites, parent_indices = self._selection_phase(k_sel, state.population, state.operators, self.engine_params)
        
        # Use state.resource_map and state.operators
        mutants = self._reproduction_phase(
                    k_cross, 
                    k_mut, 
                    parent_indices, 
                    state.population,
                    state.operators, 
                    state.resource_map
                )
        
        next_genes = self._merge(elites, mutants.genes, state)
        next_population = state.population.replace(genes=next_genes)
        
        final_state = state.replace(
            population=next_population,
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        #for hook in self.hooks:
        #    final_state = hook(final_state, self.engine_params)
        
        return final_state