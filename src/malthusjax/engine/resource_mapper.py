"""
Resource Mapper - RNG Budget Calculation and Key Map Generation

This module implements Step 3 of the Optimization Roadmap:
Pre-calculation of RNG requirements for all operators to enable
static allocation and eliminate runtime splitting overhead.
"""
from typing import NamedTuple, Dict, Any
from flax import struct
import jax.numpy as jnp

from ..operators.base import BaseMutation, BaseCrossover, BaseSelection


class OperatorRNGBudget(NamedTuple):
    """
    RNG budget information for a single operator.
    
    Attributes:
        num_keys: Number of random keys required
        start_idx: Starting index in the global key array
        end_idx: Ending index (exclusive) in the global key array
        operator_type: Type of operator ('mutation', 'crossover', 'selection')
    """
    num_keys: int
    start_idx: int
    end_idx: int
    operator_type: str


@struct.dataclass
class ResourceMap:
    """
    Complete resource allocation map for an engine.
    
    Attributes:
        total_rng_budget: Total number of random keys needed per step
        selection_budget: RNG budget info for selection operator
        crossover_budget: RNG budget info for crossover operator
        mutation_budget: RNG budget info for mutation operator
        pop_size: Population size (static parameter)
        genome_shape: Shape of a single genome
    """
    total_rng_budget: int = struct.field(pytree_node=False)
    selection_budget: OperatorRNGBudget = struct.field(pytree_node=False)
    crossover_budget: OperatorRNGBudget = struct.field(pytree_node=False)
    mutation_budget: OperatorRNGBudget = struct.field(pytree_node=False)
    pop_size: int = struct.field(pytree_node=False)
    genome_shape: tuple = struct.field(pytree_node=False)
    
    def get_key_slice(self, operator_type: str) -> slice:
        """
        Get the slice for a specific operator's keys.
        
        Args:
            operator_type: One of 'selection', 'crossover', 'mutation'
            
        Returns:
            slice object for indexing into the global key array
            
        Example:
            >>> resource_map = compute_resource_map(...)
            >>> keys = jax.random.split(main_key, resource_map.total_rng_budget)
            >>> selection_keys = keys[resource_map.get_key_slice('selection')]
        """
        if operator_type == 'selection':
            budget = self.selection_budget
        elif operator_type == 'crossover':
            budget = self.crossover_budget
        elif operator_type == 'mutation':
            budget = self.mutation_budget
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
        
        return slice(budget.start_idx, budget.end_idx)


def compute_operator_budget(
    operator: Any,
    operator_type: str,
    config: Any,
    input_shape: tuple,
    start_idx: int = 0
) -> OperatorRNGBudget:
    """
    Compute RNG budget for a single operator.
    
    Args:
        operator: The genetic operator (mutation/crossover/selection)
        operator_type: Type string ('mutation', 'crossover', 'selection')
        config: Genome configuration for the operator (not used for selection)
        input_shape: Shape of input data (e.g., genome shape or fitness array shape)
        start_idx: Starting index in the global key array
        
    Returns:
        OperatorRNGBudget with computed requirements
        
    Example:
        >>> mutation = GaussianMutation(mutation_rate=0.1, num_offspring=2)
        >>> budget = compute_operator_budget(
        ...     mutation, 'mutation', genome_config, (100,), start_idx=0
        ... )
        >>> print(budget.num_keys)  # 2 (for 2 offspring)
    """
    # Call the operator's num_keys method to get requirements
    # Selection operators only take input_shape, mutation/crossover take config and input_shape
    if operator_type == 'selection':
        num_keys = operator.num_keys(input_shape)
    else:
        num_keys = operator.num_keys(config, input_shape)
    
    return OperatorRNGBudget(
        num_keys=num_keys,
        start_idx=start_idx,
        end_idx=start_idx + num_keys,
        operator_type=operator_type
    )


def compute_resource_map(
    selection: BaseSelection,
    crossover: BaseCrossover,
    mutation: BaseMutation,
    genome_config: Any,
    pop_size: int
) -> ResourceMap:
    """
    Compute complete resource map for an engine's operators.
    
    This function implements the core of Step 3: it queries each operator
    for its RNG requirements and builds a static allocation plan that
    can be used throughout the evolution run.
    
    Args:
        selection: Selection operator
        crossover: Crossover operator
        mutation: Mutation operator
        genome_config: Genome configuration object
        pop_size: Population size
        
    Returns:
        ResourceMap with complete RNG budget allocation
        
    Example:
        >>> from malthusjax.operators.mutation import GaussianMutation
        >>> from malthusjax.operators.crossover import UniformCrossover
        >>> from malthusjax.operators.selection import TournamentSelection
        >>> 
        >>> resource_map = compute_resource_map(
        ...     TournamentSelection(tournament_size=3, num_selections=100),
        ...     UniformCrossover(num_offspring=2),
        ...     GaussianMutation(mutation_rate=0.1, num_offspring=1),
        ...     RealGenomeConfig(size=10, low=-5.0, high=5.0),
        ...     pop_size=100
        ... )
        >>> print(resource_map.total_rng_budget)
    """
    # Determine genome shape from config
    if hasattr(genome_config, 'length'):
        genome_shape = (genome_config.length,)
    elif hasattr(genome_config, 'size'):
        genome_shape = (genome_config.size,)
    elif hasattr(genome_config, 'num_categories'):
        genome_shape = (genome_config.num_categories,)
    else:
        # Fallback: assume scalar or simple shape
        genome_shape = ()
    
    # Compute budgets in order: selection, crossover, mutation
    current_idx = 0
    
    # Selection budget
    selection_budget = compute_operator_budget(
        selection, 'selection', genome_config, (pop_size,), current_idx
    )
    current_idx = selection_budget.end_idx
    
    # Crossover budget (operates on pairs of genomes)
    crossover_budget = compute_operator_budget(
        crossover, 'crossover', genome_config, genome_shape, current_idx
    )
    current_idx = crossover_budget.end_idx
    
    # Mutation budget
    mutation_budget = compute_operator_budget(
        mutation, 'mutation', genome_config, genome_shape, current_idx
    )
    current_idx = mutation_budget.end_idx
    
    total_budget = current_idx
    
    return ResourceMap(
        total_rng_budget=total_budget,
        selection_budget=selection_budget,
        crossover_budget=crossover_budget,
        mutation_budget=mutation_budget,
        pop_size=pop_size,
        genome_shape=genome_shape
    )


def get_resource_summary(resource_map: ResourceMap) -> str:
    """
    Generate a human-readable summary of resource allocation.
    
    Useful for debugging and logging during engine initialization.
    
    Args:
        resource_map: The resource map to summarize
        
    Returns:
        Formatted string with resource allocation details
    """
    lines = [
        f"Resource Allocation Summary:",
        f"  Total RNG Budget: {resource_map.total_rng_budget} keys",
        f"  Population Size: {resource_map.pop_size}",
        f"  Genome Shape: {resource_map.genome_shape}",
        "",
        "Operator Allocations:",
        f"  Selection:  {resource_map.selection_budget.num_keys} keys "
        f"[{resource_map.selection_budget.start_idx}:{resource_map.selection_budget.end_idx}]",
        f"  Crossover:  {resource_map.crossover_budget.num_keys} keys "
        f"[{resource_map.crossover_budget.start_idx}:{resource_map.crossover_budget.end_idx}]",
        f"  Mutation:   {resource_map.mutation_budget.num_keys} keys "
        f"[{resource_map.mutation_budget.start_idx}:{resource_map.mutation_budget.end_idx}]",
    ]
    return "\n".join(lines)
