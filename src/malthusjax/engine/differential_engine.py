"""
Differential Evolution Engine.
Implements the 'Inverted Pipeline' (Mutate -> Cross -> Select)
using the Static Resource Compilation (SRC) architecture.
"""
from typing import Any, Tuple, List
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex

from ..core.genome.real_genome import RealPopulation
from .base import AbstractEngine, AbstractEngineParams, AbstractGenerationOutput
from .genetic_fastengine import GeneticEvolutionState, OperatorState, traceable, GeneticEngineParams, GeneticGenerationOutput
from .resource_mapper import ResourceMap, OperatorAllocation, ShardingManager
from ..operators.base import BaseMutation, BaseCrossover

# ==============================================================================
# 1. SPECIALIZED RESOURCE COMPILER
# ==============================================================================
def compute_de_resource_map(
    pop_size: int,
    genome_config: Any,
    mutation_op: BaseMutation,
    crossover_op: BaseCrossover
) -> ResourceMap:
    """
    Compiles the RNG budget for Differential Evolution.
    
    Mapping Strategy:
    - 'selection' slot -> Used for GATHER keys (r1, r2, r3 generation)
    - 'mutation' slot  -> Used for Mutation keys (Dither/Jitter)
    - 'crossover' slot -> Used for Crossover keys (Bitmasks)
    """
    current_key_idx = 0
    
    # --- 1. GATHER BUDGET (Mapped to 'selection' slot) ---
    # We need to pick 3 indices per individual: (N, 3)
    # This requires randomness. We treat this as a "Selection" of 3*N items.
    # Cost: 1 key usually allows generating a block of ints. 
    # Let's allocate 1 key per individual to be safe/simple for splitting.
    gather_keys = pop_size 
    
    selection_alloc = OperatorAllocation(
        num_keys=gather_keys,
        start_idx=current_key_idx,
        end_idx=current_key_idx + gather_keys,
        input_count=pop_size,
        output_count=pop_size * 3, # We produce 3 indices per person
        operator_type='de_gather'
    )
    current_key_idx += gather_keys
    
    # --- 2. MUTATION BUDGET ---
    # DE Mutation operates on N triplets.
    mut_keys = mutation_op.num_keys(input_shape=(pop_size,))
    
    mutation_alloc = OperatorAllocation(
        num_keys=mut_keys,
        start_idx=current_key_idx,
        end_idx=current_key_idx + mut_keys,
        input_count=pop_size, # N triplets
        output_count=pop_size, # N mutants
        operator_type='de_mutation'
    )
    current_key_idx += mut_keys
    
    # --- 3. CROSSOVER BUDGET ---
    # DE Crossover operates on N pairs (Target, Mutant).
    # Standard Crossover might expect input_shape[0] to be num_pairs.
    # Here we have N pairs.
    cross_keys = crossover_op.num_keys(input_shape=(pop_size,))
    
    crossover_alloc = OperatorAllocation(
        num_keys=cross_keys,
        start_idx=current_key_idx,
        end_idx=current_key_idx + cross_keys,
        input_count=pop_size, # N pairs
        output_count=pop_size, # N trials
        operator_type='de_crossover'
    )
    current_key_idx += cross_keys
    
    # --- 4. NEXT KEY ---
    next_key_alloc = OperatorAllocation(
        num_keys=1,
        start_idx=current_key_idx,
        end_idx=current_key_idx + 1,
        input_count=0, output_count=1, operator_type='next_key'
    )
    current_key_idx += 1
    
    # Determine genome shape for metadata
    shape = genome_config.shape if hasattr(genome_config, 'shape') else ()

    return ResourceMap(
        total_rng_budget=current_key_idx,
        selection=selection_alloc,
        crossover=crossover_alloc,
        mutation=mutation_alloc,
        next_key=next_key_alloc,
        pop_size=pop_size,
        genome_shape=shape
    )


# ==============================================================================
# 2. THE DIFFERENTIAL EVOLUTION ENGINE
# ==============================================================================
@struct.dataclass
class DifferentialEvolutionEngine(AbstractEngine):
    """
    High-Performance Differential Evolution (DE/rand/1/bin).
    Uses 'Init-Phase Compilation' but with DE-specific data flow.
    """
    # Core Blueprints
    genome_config: Any
    evaluator: Any
    mutation: BaseMutation     # Must support Tuple[G,G,G] -> G
    crossover: BaseCrossover   # Must support (G, G) -> G
    
    enable_progress_bar: bool = struct.field(pytree_node=False, default=False)
    
    # Internal Buffer
    _entropy_buffer: List[Any] = struct.field(pytree_node=False, default_factory=list)

    # --------------------------------------------------------------------------
    # COMPILER (Init State)
    # --------------------------------------------------------------------------
    def init_state(self, rng_key: chex.Array) -> GeneticEvolutionState:
        # 1. COMPILE RESOURCE MAP (Using the DE-specific compiler)
        rmap = compute_de_resource_map(
            self.engine_params.pop_size,
            self.genome_config,
            self.mutation,
            self.crossover
        )
        
        # 2. SETUP SHARDING (Identical to GA - It's just a batch)
        sharding_mgr = ShardingManager(axis_name='batch')
        
        # 3. BAKE OPERATORS
        # Freeze input sizes. Note: DE operators process the WHOLE population (N).
        # We don't divide by 2 like in GA crossover.
        active_mut = self.mutation.set_input_length(rmap.mutation.input_count)
        active_cross = self.crossover.set_input_length(rmap.crossover.input_count)
        
        # We don't use a 'Selection' operator in the traditional sense, 
        # but we need to store something in OperatorState to keep typing happy 
        # or we can leave it None if we handle Gather manually.
        # Let's mock it to reuse GeneticEvolutionState or just ignore it.
        # Ideally, define a new DEOperatorState, but reusing is fine for now.
        op_state = OperatorState(
            selection=None, # Not used in DE step
            crossover=active_cross,
            mutation=active_mut
        )
        
        # 4. INITIALIZE POPULATION (Standard)
        init_pop_key, rng_key = jar.split(rng_key)
        population = RealPopulation.init_random(
            init_pop_key, 
            self.genome_config, 
            self.engine_params.pop_size
        )
        
        # 5. ENFORCE SHARDING (Copy-Paste from GeneticEngine)
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
        
        # 6. INITIAL EVAL
        evaluated_pop = self.evaluator.evaluate_population(population)
        
        # 7. FINALIZE
        best_idx = jnp.argmax(evaluated_pop.fitness)
        best_genome = jax.tree_util.tree_map(
            lambda x: x[best_idx], evaluated_pop.genes
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

    # --------------------------------------------------------------------------
    # EXECUTION (Step)
    # --------------------------------------------------------------------------
    @traceable("DE_Entropy_Alloc")
    def _allocate_entropy(self, state):
        rmap = state.resource_map
        all_keys = jar.split(state.rng_key, rmap.total_rng_budget)
        
        # Map slices to DE semantics
        k_gather = all_keys[rmap.get_key_slice('selection')] # Reused slot
        k_mut    = all_keys[rmap.get_key_slice('mutation')]
        k_cross  = all_keys[rmap.get_key_slice('crossover')]
        k_next   = all_keys[rmap.get_key_slice('next_key')][0]
        
        return k_gather, k_mut, k_cross, k_next

    @traceable("DE_Gather_Phase")
    def _gather_candidates(self, key, population, pop_size):
        """
        Selects r1, r2, r3 for every individual i.
        Constraint: r1 != r2 != r3 != i (Ideally).
        Fast Approximation: Random sampling (valid for N >> 3).
        """
        # Shape: (3, N)
        # We use one key to generate a massive block of ints? 
        # Or split the gather key?
        # rmap allocated 'pop_size' keys. Let's use the first one for simplicity 
        # or implement proper splitting if we want strict independence.
        
        # Simple high-perf gather:
        # Just generate (3, N) integers.
        indices = jax.random.randint(
            key[0], # Just use the first key of the block
            shape=(3, pop_size), 
            minval=0, 
            maxval=pop_size
        )
        
        # Gather Genes (Structure of Arrays)
        # (3, N, D)
        candidates = jax.tree_util.tree_map(
            lambda g: jnp.take(g, indices, axis=0),
            population.genes
        )
        
        # Unpack to Tuple of Genomes
        # We need to construct 3 Genome objects from this stack        
        def extract_layer(i):
            return jax.tree_util.tree_map(lambda x: x[i], candidates)
        
        r1 = population.genes.__class__(**extract_layer(0).__dict__) # Hacky reconstruction or just use replace
        r2 = population.genes.__class__(**extract_layer(1).__dict__)
        
        # Cleaner way if Genome is PyTree:
        r1 = jax.tree_util.tree_map(lambda x: x[0], candidates)
        r2 = jax.tree_util.tree_map(lambda x: x[1], candidates)
        r3 = jax.tree_util.tree_map(lambda x: x[2], candidates)
        
        return (r1, r2, r3)

    @traceable("DE_Step")
    def step(self, state: GeneticEvolutionState) -> Tuple[GeneticEvolutionState, Any]:
        
        # 1. ENTROPY
        k_gather, k_mut, k_cross, k_next = self._allocate_entropy(state)
        
        pop = state.population
        N = self.engine_params.pop_size
        
        # 2. GATHER
        # Returns Tuple[Genome, Genome, Genome]
        triplets = self._gather_candidates(k_gather, pop, N)
        
        # 3. MUTATION (Triplets -> Mutants)
        # DifferentialMutation expects Tuple input
        mutants = state.operators.mutation(k_mut, triplets, self.genome_config)
        
        # 4. CROSSOVER (Target + Mutant -> Trial)
        # BinomialCrossover expects (p1, p2)
        trials = state.operators.crossover(k_cross, pop.genes, mutants, self.genome_config)
        
        # 5. EVALUATE TRIALS
        # Create temp population
        trial_pop = pop.replace(genes=trials)
        trial_pop = self.evaluator.evaluate_population(trial_pop)
        
        # 6. GREEDY SELECTION
        mask = trial_pop.fitness > pop.fitness
        
        # Fused Select
        next_genes = jax.tree_util.tree_map(
            lambda t, p: jnp.where(mask.reshape(N, *([1]*(t.ndim-1))), t, p),
            trial_pop.genes, pop.genes
        )
        next_fitness = jnp.where(mask, trial_pop.fitness, pop.fitness)
        
        # 7. UPDATE HOF (Same as GA)
        best_idx = jnp.argmax(next_fitness)
        is_new = next_fitness[best_idx] > state.best_fitness
        
        new_best_genome = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_new, new, old),
            state.best_genome, 
            jax.tree_util.tree_map(lambda x: x[best_idx], next_genes)
        )

        final_state = state.replace(
            population=pop.replace(genes=next_genes, fitness=next_fitness),
            best_genome=new_best_genome,
            best_fitness=jnp.where(is_new, next_fitness[best_idx], state.best_fitness),
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        return final_state, GeneticGenerationOutput(
            best_fitness=final_state.best_fitness,
            mean_fitness=jnp.mean(next_fitness),
            generation=final_state.generation,
            random_key=final_state.rng_key
        )
        
    def ask(self, state: GeneticEvolutionState) -> Any:
        raise NotImplementedError("DE Engine does not support separate ask(). Use step() instead.")
        
    def tell(self, state: GeneticEvolutionState, evaluated_population: Any) -> GeneticEvolutionState:
        # This method is not used in DE since step() handles evaluation internally.
        raise NotImplementedError("DE Engine does not support separate tell(). Use step() instead.")