"""
Operator Inspector - Kernel Support Detection

This module implements Step 2 of the Optimization Roadmap:
Automatic detection of operator kernel capabilities to enable
transparent switching between FAST_LANE and LEGACY execution modes.
"""
from enum import Enum
from typing import Any, NamedTuple, Optional
from flax import struct

from ..operators.base import BaseMutation, BaseCrossover, BaseSelection


class ExecutionMode(Enum):
    """Engine execution modes based on operator capabilities."""
    FAST_LANE = "fast_lane"  # All operators support kernel interface
    LEGACY = "legacy"        # At least one operator lacks kernel support


class OperatorIdentityCard(NamedTuple):
    """
    Metadata about an operator's kernel support.
    
    Attributes:
        has_num_keys: Whether operator implements custom num_keys()
        has_get_output_shape: Whether operator implements custom get_output_shape()
        has_apply_kernel: Whether operator implements custom apply_kernel()
        supports_kernel: True if all three kernel methods are implemented
        operator_type: 'mutation', 'crossover', or 'selection'
    """
    has_num_keys: bool
    has_get_output_shape: bool
    has_apply_kernel: bool
    supports_kernel: bool
    operator_type: str


@struct.dataclass
class EngineInspectionResult:
    """
    Results from inspecting operators at engine initialization.
    
    Attributes:
        mode: ExecutionMode.FAST_LANE or ExecutionMode.LEGACY
        mutation_card: Identity card for mutation operator
        crossover_card: Identity card for crossover operator
        selection_card: Identity card for selection operator
        all_support_kernel: True if all operators support kernel interface
    """
    mode: ExecutionMode = struct.field(pytree_node=False)
    mutation_card: OperatorIdentityCard = struct.field(pytree_node=False)
    crossover_card: OperatorIdentityCard = struct.field(pytree_node=False)
    selection_card: OperatorIdentityCard = struct.field(pytree_node=False)
    all_support_kernel: bool = struct.field(pytree_node=False)


def _has_custom_implementation(operator: Any, method_name: str, base_class: type) -> bool:
    """
    Check if an operator has a custom implementation of a method.
    
    Returns True if the method is overridden from the base class.
    """
    if not hasattr(operator, method_name):
        return False
    
    operator_method = getattr(operator.__class__, method_name, None)
    base_method = getattr(base_class, method_name, None)
    
    # If both exist and are different objects, it's a custom implementation
    return operator_method is not None and operator_method is not base_method


def inspect_operator(operator: Any) -> OperatorIdentityCard:
    """
    Inspect a single operator to determine its kernel support capabilities.
    
    Args:
        operator: A genetic operator (mutation, crossover, or selection)
        
    Returns:
        OperatorIdentityCard with detailed capability information
        
    Example:
        >>> from malthusjax.operators.mutation import BitFlipMutation
        >>> mutation = BitFlipMutation(mutation_rate=0.1)
        >>> card = inspect_operator(mutation)
        >>> print(card.supports_kernel)  # False (default implementation)
    """
    # Determine operator type and base class
    if isinstance(operator, BaseMutation):
        operator_type = "mutation"
        base_class = BaseMutation
    elif isinstance(operator, BaseCrossover):
        operator_type = "crossover"
        base_class = BaseCrossover
    elif isinstance(operator, BaseSelection):
        operator_type = "selection"
        base_class = BaseSelection
    else:
        raise ValueError(f"Unknown operator type: {type(operator)}")
    
    # Check for custom implementations of kernel methods
    has_num_keys = _has_custom_implementation(operator, "num_keys", base_class)
    has_get_output_shape = _has_custom_implementation(operator, "get_output_shape", base_class)
    has_apply_kernel = _has_custom_implementation(operator, "apply_kernel", base_class)
    
    # All three methods must be custom for full kernel support
    supports_kernel = has_num_keys and has_get_output_shape and has_apply_kernel
    
    return OperatorIdentityCard(
        has_num_keys=has_num_keys,
        has_get_output_shape=has_get_output_shape,
        has_apply_kernel=has_apply_kernel,
        supports_kernel=supports_kernel,
        operator_type=operator_type
    )


def inspect_engine_operators(
    mutation: BaseMutation,
    crossover: BaseCrossover,
    selection: BaseSelection
) -> EngineInspectionResult:
    """
    Inspect all operators in an engine to determine execution mode.
    
    This function implements the "Inspector" component from Step 2 of the
    optimization roadmap. It detects whether operators support the kernel
    interface and flags the engine for FAST_LANE or LEGACY execution.
    
    Args:
        mutation: Mutation operator
        crossover: Crossover operator
        selection: Selection operator
        
    Returns:
        EngineInspectionResult with mode and detailed operator cards
        
    Example:
        >>> from malthusjax.operators.mutation import GaussianMutation
        >>> from malthusjax.operators.crossover import UniformCrossover
        >>> from malthusjax.operators.selection import TournamentSelection
        >>> 
        >>> result = inspect_engine_operators(
        ...     GaussianMutation(mutation_rate=0.1),
        ...     UniformCrossover(),
        ...     TournamentSelection(tournament_size=3)
        ... )
        >>> print(result.mode)  # ExecutionMode.LEGACY (default implementations)
    """
    # Inspect each operator
    mutation_card = inspect_operator(mutation)
    crossover_card = inspect_operator(crossover)
    selection_card = inspect_operator(selection)
    
    # Determine if all operators support kernel interface
    all_support_kernel = (
        mutation_card.supports_kernel and
        crossover_card.supports_kernel and
        selection_card.supports_kernel
    )
    
    # Set execution mode
    mode = ExecutionMode.FAST_LANE if all_support_kernel else ExecutionMode.LEGACY
    
    return EngineInspectionResult(
        mode=mode,
        mutation_card=mutation_card,
        crossover_card=crossover_card,
        selection_card=selection_card,
        all_support_kernel=all_support_kernel
    )


def get_kernel_support_summary(result: EngineInspectionResult) -> str:
    """
    Generate a human-readable summary of kernel support.
    
    Useful for debugging and logging during engine initialization.
    
    Args:
        result: Inspection result from inspect_engine_operators()
        
    Returns:
        Formatted string with kernel support details
    """
    lines = [
        f"Engine Mode: {result.mode.value.upper()}",
        f"All operators support kernel: {result.all_support_kernel}",
        "",
        "Operator Kernel Support:",
        f"  Mutation:  {_format_card(result.mutation_card)}",
        f"  Crossover: {_format_card(result.crossover_card)}",
        f"  Selection: {_format_card(result.selection_card)}",
    ]
    return "\n".join(lines)


def _format_card(card: OperatorIdentityCard) -> str:
    """Format an identity card for display."""
    if card.supports_kernel:
        return "✓ Full kernel support"
    else:
        missing = []
        if not card.has_num_keys:
            missing.append("num_keys")
        if not card.has_get_output_shape:
            missing.append("get_output_shape")
        if not card.has_apply_kernel:
            missing.append("apply_kernel")
        return f"✗ Missing: {', '.join(missing)}"
