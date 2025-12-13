"""
Resource Mapper - Static RNG Budget Allocator & Data Flow Calculator.

Step 3 of Optimization Roadmap: 
Pre-calculates exact RNG requirements and operator output shapes
to enable static allocation and precise "cascade" data flow.
"""
from typing import NamedTuple, Any, Tuple
from flax import struct

from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection

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
    """
    Master plan for RNG distribution and Data Flow in one generation.
    """
    total_rng_budget: int = struct.field(pytree_node=False)
    
    # Per-Operator Allocations
    selection: OperatorAllocation = struct.field(pytree_node=False)
    crossover: OperatorAllocation = struct.field(pytree_node=False)
    mutation:  OperatorAllocation = struct.field(pytree_node=False)
    next_key:  OperatorAllocation = struct.field(pytree_node=False)
    
    # Metadata
    pop_size: int = struct.field(pytree_node=False)
    genome_shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def get_key_slice(self, op_name: str) -> slice:
        """Returns the slice to extract this operator's keys from the master buffer."""
        alloc = getattr(self, op_name)
        return slice(alloc.start_idx, alloc.end_idx)
    
    def get_output_count(self, op_name: str) -> int:
        """Returns the number of items produced by this operator."""
        return getattr(self, op_name).output_count


def compute_resource_map(
    selection: BaseSelection,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    genome_config: Any,
    pop_size: int
) -> ResourceMap:
    """
    Compiles the RNG requirements and Data Flow for the entire evolution loop.
    
    Calculates the 'Cascade Effect':
    Selection(N) -> Parents(P) -> Crossover(P/2) -> Offspring(O) -> Mutation(O) -> Mutants(M)
    """
    # Helper to track key indices
    current_key_idx = 0
    
    # Determine genome shape (for metadata)
    if hasattr(genome_config, 'length'):
        genome_shape = (genome_config.length,)
    elif hasattr(genome_config, 'size'):
        genome_shape = (genome_config.size,)
    elif hasattr(genome_config, 'shape'):
        genome_shape = genome_config.shape
    else:
        genome_shape = ()

    # ==========================================
    # 1. SELECTION & PARENT CALCULATIONS (The Fix)
    # ==========================================
    # Input: Current Population (pop_size)
    # Logic: We must ensure we select enough parents to generate AT LEAST pop_size offspring.
    # ------------------------------------------
    
    # 1. Determine how many offspring one pair produces (usually 2)
    offspring_per_pair = getattr(crossover, 'num_offspring', 2)
    
    # 2. Calculate pairs needed to satisfy pop_size (Coverage Strategy)
    # Formula: ceil(pop_size / offspring_per_pair)
    # Implementation: (pop_size + n - 1) // n
    pairs_needed = (pop_size + offspring_per_pair - 1) // offspring_per_pair
    
    # 3. Parents needed (2 parents per pair)
    parents_needed = pairs_needed * 2
    
    # 4. Selection Configuration
    sel_input_count = pop_size
    sel_output_count = parents_needed
    
    # Update Selection Context: 
    # We create a temporary shadow operator with the CORRECT num_selections
    # so num_keys returns the right amount for the resource budget.
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
    # 2. CROSSOVER
    # ==========================================
    # Input: Selected Parents (sel_output_count)
    # Logic: Parents are paired.
    # ------------------------------------------
    cross_input_count = sel_output_count # e.g. 18 (if pop_size=17)
    num_pairs = cross_input_count // 2   # e.g. 9
    
    # Update Operator Context with number of PAIRS
    crossover = crossover.set_input_length(num_pairs)
    
    # Output: Pairs * num_offspring (per pair)
    # This might be slightly larger than pop_size (e.g. 18), which is fine.
    cross_output_count = num_pairs * crossover.num_offspring
    
    # RNG: Calculate keys needed for 'num_pairs' operations
    cross_keys_needed = crossover.num_keys(input_shape=(num_pairs,))
    
    crossover_alloc = OperatorAllocation(
        num_keys=cross_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + cross_keys_needed,
        input_count=cross_input_count, # Total parents entering
        output_count=cross_output_count,
        operator_type='crossover'
    )
    current_key_idx += cross_keys_needed

    # ==========================================
    # 3. MUTATION
    # ==========================================
    # Input: Crossover Offspring (cross_output_count)
    # Output: Mutated Individuals (Input * num_offspring per mutant)
    # ------------------------------------------
    mut_input_count = cross_output_count
    
    # Update Operator Context
    mutation = mutation.set_input_length(mut_input_count)
    
    # Output Logic
    mut_output_count = mut_input_count * mutation.num_offspring
    
    # RNG: Calculate keys needed
    mut_keys_needed = mutation.num_keys(input_shape=(mut_input_count,))
    
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
    # 4. NEXT GENERATION KEY
    # ==========================================
    # System requirement: 1 key to seed the next step
    # ------------------------------------------
    next_key_alloc = OperatorAllocation(
        num_keys=1,
        start_idx=current_key_idx,
        end_idx=current_key_idx + 1,
        input_count=0,  # N/A
        output_count=1, # N/A
        operator_type='next_key'
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