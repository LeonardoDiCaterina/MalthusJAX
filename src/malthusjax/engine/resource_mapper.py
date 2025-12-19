"""
Resource Mapper - Static RNG Budget Allocator & Data Flow Calculator.

Step 3 of Optimization Roadmap: 
Pre-calculates exact RNG requirements and operator output shapes
to enable static allocation and precise "cascade" data flow.
"""
from typing import NamedTuple, Any, Tuple
from flax import struct
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.numpy as jnp

from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection

class ShardingManager:
    """
    Manages the GSPMD layout for the population.
    Works for 1 Device (Layout Optimization) and N Devices (Parallelism).
    """
    def __init__(self, axis_name='batch'):
        self.axis_name = axis_name
        self.devices = jax.devices()
        self.mesh = Mesh(self.devices, (self.axis_name,))
        
        # Rule 1: Matrices (Genomes) -> (Batch, Length)
        # Split dim 0, Keep dim 1 whole
        self.matrix_spec = P(self.axis_name, None)
        self.matrix_sharding = NamedSharding(self.mesh, self.matrix_spec)

        # Rule 2: Vectors (Fitness) -> (Batch,)  <--- NEW
        # Split dim 0, no other dims exist
        self.vector_spec = P(self.axis_name)
        self.vector_sharding = NamedSharding(self.mesh, self.vector_spec)
        
        # Rule 3: Replicated (Scalars/Config)
        self.replicated_spec = P()
        self.replicated_sharding = NamedSharding(self.mesh, self.replicated_spec)        
        # 1. Create the Mesh
        # Even if you have 1 GPU, creating a mesh tells XLA:
        # "This axis exists conceptually."
        self.mesh = Mesh(self.devices, (self.axis_name,))
        
        # 2. Define the Rules (PartitionSpecs)
        # Rule: Split the first dim (Population), keep the rest whole (None).
        self.pop_spec = P(self.axis_name, None)  
        
        # Rule: Replicate scalars (Mutation Rate, etc.) on all devices.
        self.replicated_spec = P() 
        
        # 3. Create the Sharding Objects
        self.pop_sharding = NamedSharding(self.mesh, self.pop_spec)
        self.replicated_sharding = NamedSharding(self.mesh, self.replicated_spec)

    def alloc_population(self, shape, dtype=jnp.float32):
        """
        Allocates a zero-filled population tensor with enforced sharding.
        """
        # We use jax.device_put to force the layout immediately upon creation.
        # This prevents XLA from creating it on Host then moving to Device.
        return jax.device_put(jnp.zeros(shape, dtype=dtype), self.pop_sharding)

    def split_key_sharded(self, key, num):
        """
        Splits RNG keys such that each device gets its own independent stream.
        This is crucial for Multi-GPU stochasticity.
        """
        # Standard split
        keys = jax.random.split(key, num)
        # Enforce that the keys are sharded across the batch dimension
        return jax.device_put(keys, self.pop_sharding)
    

class OperatorAllocation(NamedTuple):
    """
    Allocation details for a single operator stage.
    """
    num_keys: int
    start_idx: int
    end_idx: int
    input_count: int
    output_count: int
    operator_type: str

@struct.dataclass
class ResourceMap:
    total_rng_budget: int = struct.field(pytree_node=False)
    selection: OperatorAllocation = struct.field(pytree_node=False)
    crossover: OperatorAllocation = struct.field(pytree_node=False)
    mutation:  OperatorAllocation = struct.field(pytree_node=False)
    next_key:  OperatorAllocation = struct.field(pytree_node=False)
    pop_size: int = struct.field(pytree_node=False)
    genome_shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def get_key_slice(self, op_name: str) -> slice:
        alloc = getattr(self, op_name)
        return slice(alloc.start_idx, alloc.end_idx)

# --- THE OPTIMIZED LOGIC ---
def compute_resource_map(
    selection: BaseSelection,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    genome_config: Any,
    pop_size: int
) -> ResourceMap:
    """
    Compiles the RNG requirements and Data Flow for the entire evolution loop.
    Respects operator configuration for Exact Allocation.
    """
    current_key_idx = 0
    
    # Metadata
    if hasattr(genome_config, 'length'): genome_shape = (genome_config.length,)
    elif hasattr(genome_config, 'size'): genome_shape = (genome_config.size,)
    elif hasattr(genome_config, 'shape'): genome_shape = genome_config.shape
    else: genome_shape = ()

    # ==========================================
    # 1. SELECTION (The Source)
    # ==========================================
    # Logic: Did the user specify exactly how many parents to select?
    # If yes (num_selections > 0), use that.
    # If no (default/sentinel), calculate needed for full replacement.
    
    sel_input_count = pop_size
    
    if hasattr(selection, 'num_selections') and selection.num_selections > 0:
        # EXACT MODE: User specified count (e.g., 40)
        sel_output_count = selection.num_selections
    else:
        # COVERAGE MODE: Calculate to fill pop_size
        offspring_per_pair = getattr(crossover, 'num_offspring', 2)
        pairs_needed = (pop_size + offspring_per_pair - 1) // offspring_per_pair
        sel_output_count = pairs_needed * 2

    # Calculate Keys
    # We update the temp operator to reflect the decision so num_keys is correct
    temp_sel = selection.replace(num_selections=sel_output_count).set_input_length(sel_input_count)
    sel_keys_needed = temp_sel.num_keys(input_shape=(sel_input_count,))
    
    selection_alloc = OperatorAllocation(
        num_keys=sel_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + sel_keys_needed,
        input_count=sel_input_count,
        output_count=sel_output_count,
        operator_type='selection'
    )
    current_key_idx += sel_keys_needed

    # ==========================================
    # 2. CROSSOVER (The Filter)
    # ==========================================
    # Logic: Input matches Selection Output exactly.
    
    cross_input_count = sel_output_count
    num_pairs = cross_input_count // 2  # Integer division drops odd parent
    
    # Update Operator Context
    # This allows the operator to know its batch size for key calc
    temp_cross = crossover.set_input_length(num_pairs)
    
    # Output Flow
    cross_output_count = num_pairs * getattr(crossover, 'num_offspring', 1)
    
    # Calculate Keys
    cross_keys_needed = temp_cross.num_keys(input_shape=(num_pairs,))
    
    crossover_alloc = OperatorAllocation(
        num_keys=cross_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + cross_keys_needed,
        input_count=cross_input_count,
        output_count=cross_output_count,
        operator_type='crossover'
    )
    current_key_idx += cross_keys_needed

    # ==========================================
    # 3. MUTATION (The Transform)
    # ==========================================
    # Logic: Input matches Crossover Output exactly.
    
    mut_input_count = cross_output_count
    
    temp_mut = mutation.set_input_length(mut_input_count)
    mut_output_count = mut_input_count * getattr(mutation, 'num_offspring', 1)
    
    mut_keys_needed = temp_mut.num_keys(input_shape=(mut_input_count,))
    
    mutation_alloc = OperatorAllocation(
        num_keys=mut_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + mut_keys_needed,
        input_count=mut_input_count,
        output_count=mut_output_count,
        operator_type='mutation'
    )
    current_key_idx += mut_keys_needed

    # ==========================================
    # 4. NEXT KEY
    # ==========================================
    next_key_alloc = OperatorAllocation(
        num_keys=1, start_idx=current_key_idx, end_idx=current_key_idx + 1,
        input_count=0, output_count=1, operator_type='next_key'
    )
    current_key_idx += 1

    return ResourceMap(
        total_rng_budget=current_key_idx,
        selection=selection_alloc,
        crossover=crossover_alloc,
        mutation=mutation_alloc,
        next_key=next_key_alloc,
        pop_size=pop_size,
        genome_shape=genome_shape
    )

def get_resource_summary(rmap: ResourceMap) -> str:
    """Generate a cascade data flow summary."""
    s = rmap.selection
    c = rmap.crossover
    m = rmap.mutation
    
    lines = [
        f"Pipeline Resource & Flow Summary:",
        f"  Total RNG Budget: {rmap.total_rng_budget} keys",
        "",
        "  [1. SELECTION]",
        f"     In: {s.input_count} (Pop Size) -> Out: {s.output_count} indices (Parents needed)",
        f"     Keys: {s.num_keys} (Slice {s.start_idx}:{s.end_idx})",
        "",
        "  [2. CROSSOVER]",
        f"     In: {c.input_count} parents ({c.input_count//2} pairs) -> Out: {c.output_count} offspring",
        f"     Keys: {c.num_keys} (Slice {c.start_idx}:{c.end_idx})",
        "",
        "  [3. MUTATION]",
        f"     In: {m.input_count} -> Out: {m.output_count} mutants",
        f"     Keys: {m.num_keys} (Slice {m.start_idx}:{m.end_idx})",
        "",
        "  [4. NEXT GENERATION KEY]",
        f"     Keys: {rmap.next_key.num_keys} (Slice {rmap.next_key.start_idx}:{rmap.next_key.end_idx})",
    ]
    return "\n".join(lines)