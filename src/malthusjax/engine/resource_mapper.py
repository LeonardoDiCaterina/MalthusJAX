"""
Resource Mapper - Static RNG Budget Allocator & Data Flow Calculator.

Step 3 of Optimization Roadmap:
Pre-calculates exact RNG requirements and operator output shapes
to enable static allocation and precise "cascade" data flow.
"""

from enum import Enum
from typing import Any, NamedTuple, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection

_field: Any = struct.field  # Helper for typed contexts


class KeyDerivationStrategy(Enum):
    """
    Strategy for deriving random keys from a master key.

    SPLIT: Uses jax.random.split repeatedly - produces uncorrelated keys
           but requires sequential splits (not fully parallelizable)

    FOLD: Uses jax.random.fold_in with indices - produces deterministic
          keys that can be generated in parallel from the same master key
    """

    SPLIT = "split"
    FOLD = "fold_in"


class ShardingManager:
    """
    Manages the GSPMD layout for the population.
    Works for 1 Device (Layout Optimization) and N Devices (Parallelism).
    """

    def __init__(self, axis_name: str = "batch") -> None:
        self.axis_name: str = axis_name
        self.devices = jax.devices()
        self.mesh = Mesh(self.devices, (self.axis_name,))
        self.matrix_spec = P(self.axis_name, None)  # type: ignore[no-untyped-call]
        self.matrix_sharding = NamedSharding(self.mesh, self.matrix_spec)
        self.vector_spec = P(self.axis_name)  # type: ignore[no-untyped-call]
        self.vector_sharding = NamedSharding(self.mesh, self.vector_spec)
        self.replicated_spec = P()  # type: ignore[no-untyped-call]
        self.replicated_sharding = NamedSharding(self.mesh, self.replicated_spec)
        self.mesh = Mesh(self.devices, (self.axis_name,))
        self.pop_spec = P(self.axis_name, None)  # type: ignore[no-untyped-call]
        self.replicated_spec = P()  # type: ignore[no-untyped-call]
        self.pop_sharding = NamedSharding(self.mesh, self.pop_spec)
        self.replicated_sharding = NamedSharding(self.mesh, self.replicated_spec)

    def alloc_population(self, shape: Tuple[int, ...], dtype: Any = jnp.float32) -> chex.Array:
        """
        Allocates a zero-filled population tensor with enforced sharding.

        We use jax.device_put to force the layout immediately upon creation.
        This prevents XLA from creating it on Host then moving to Device.
        """
        return cast(chex.Array, jax.device_put(jnp.zeros(shape, dtype=dtype), self.pop_sharding))

    def split_key_sharded(self, key: chex.Array, num: int) -> chex.Array:
        """
        Splits RNG keys such that each device gets its own independent stream.
        This is crucial for Multi-GPU stochasticity.

        Enforce that the keys are sharded across the batch dimension
        """
        keys = jax.random.split(key, num)

        return cast(chex.Array, jax.device_put(keys, self.pop_sharding))


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

    total_rng_budget: int = _field(pytree_node=False)

    selection: OperatorAllocation = _field(pytree_node=False)
    crossover: OperatorAllocation = _field(pytree_node=False)
    mutation: OperatorAllocation = _field(pytree_node=False)
    next_key: OperatorAllocation = _field(pytree_node=False)

    pop_size: int = _field(pytree_node=False)
    genome_shape: Tuple[int, ...] = _field(pytree_node=False)
    key_derivation: KeyDerivationStrategy = _field(
        pytree_node=False, default=KeyDerivationStrategy.SPLIT
    )

    def get_key_slice(self, op_name: str) -> slice:
        """Returns the slice to extract this operator's keys from the master buffer."""
        alloc = getattr(self, op_name)
        return slice(alloc.start_idx, alloc.end_idx)

    def get_output_count(self, op_name: str) -> int:
        """Returns the number of items produced by this operator."""
        alloc: OperatorAllocation = getattr(self, op_name)
        return int(alloc.output_count)

    def get_keys(self, master_key: chex.Array) -> chex.Array:
        """
        Generate all RNG keys for one generation using the configured strategy.

        Args:
            master_key: The master random key for this generation

        Returns:
            Array of shape (total_rng_budget,) containing all keys
        """
        if self.key_derivation == KeyDerivationStrategy.SPLIT:
            # Sequential splitting
            return cast(chex.Array, jax.random.split(master_key, int(self.total_rng_budget)))
        elif self.key_derivation == KeyDerivationStrategy.FOLD:
            # Parallel fold_in: avoid Python-level conversions so this works with JAX tracers
            def _fold_in(i: chex.Array) -> chex.Array:
                # Call fold_in with a JAX value directly (no int()), which is safe under vmap
                return jax.random.fold_in(master_key, i)

            indices = jnp.arange(int(self.total_rng_budget))
            return cast(chex.Array, jax.vmap(_fold_in)(indices))
        else:
            raise ValueError(f"Unknown key derivation strategy: {self.key_derivation}")


def compute_resource_map(
    selection: BaseSelection[Any, Any],
    crossover: BaseCrossover[Any, Any, Any],
    mutation: BaseMutation[Any, Any, Any],
    genome_config: Any,
    pop_size: int,
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT,
) -> ResourceMap:
    """
    Compiles the RNG requirements and Data Flow for the entire evolution loop.

    Calculates the 'Cascade Effect':
    Selection(N) -> Parents(P) -> Crossover(P/2) -> Offspring(O) -> Mutation(O) -> Mutants(M)
    """
    current_key_idx = 0

    # Determine genome shape (for metadata)
    # Prefer `resolved_shape` if available (handles legacy `length` alias)
    # Normalize genome_shape to Tuple[int, ...]
    genome_shape: Tuple[int, ...]
    if hasattr(genome_config, "resolved_shape"):
        genome_shape = cast(Tuple[int, ...], genome_config.resolved_shape)
    elif getattr(genome_config, "length", None) is not None:
        genome_shape = (int(genome_config.length),)
    elif getattr(genome_config, "size", None) is not None:
        genome_shape = (int(genome_config.size),)
    elif getattr(genome_config, "shape", None) is not None:
        genome_shape = cast(Tuple[int, ...], genome_config.shape)
    else:
        genome_shape = cast(Tuple[int, ...], ())

    offspring_per_pair = getattr(crossover, "num_offspring", 2)
    pairs_needed = (pop_size + offspring_per_pair - 1) // offspring_per_pair
    parents_needed = pairs_needed * 2

    sel_input_count = pop_size
    sel_output_count = parents_needed
    temp_sel = (
        cast(Any, selection)
        .replace(num_selections=sel_output_count)
        .set_input_length(sel_input_count)
    )
    sel_keys_needed = temp_sel.num_keys(input_shape=(sel_input_count,))

    selection_alloc = OperatorAllocation(
        num_keys=sel_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + sel_keys_needed,
        input_count=sel_input_count,
        output_count=sel_output_count,
        operator_type="selection",
    )
    current_key_idx += sel_keys_needed

    # crossover
    cross_input_count = sel_output_count  # e.g. 18 (if pop_size=17)
    num_pairs = cross_input_count // 2  # e.g. 9

    crossover = crossover.set_input_length(num_pairs)

    # Output: Pairs * num_offspring (per pair)
    # This might be slightly larger than pop_size (e.g. 18), we will allow that
    cross_output_count = num_pairs * crossover.num_offspring

    cross_keys_needed = crossover.num_keys(input_shape=(num_pairs,))

    crossover_alloc = OperatorAllocation(
        num_keys=cross_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + cross_keys_needed,
        input_count=cross_input_count,  # Total parents entering
        output_count=cross_output_count,
        operator_type="crossover",
    )
    current_key_idx += cross_keys_needed

    # mutation
    mut_input_count = cross_output_count
    mutation = mutation.set_input_length(mut_input_count)
    mut_output_count = mut_input_count * mutation.num_offspring
    mut_keys_needed = mutation.num_keys(input_shape=(mut_input_count,))

    mutation_alloc = OperatorAllocation(
        num_keys=mut_keys_needed,
        start_idx=current_key_idx,
        end_idx=current_key_idx + mut_keys_needed,
        input_count=mut_input_count,
        output_count=mut_output_count,
        operator_type="mutation",
    )
    current_key_idx += mut_keys_needed

    # next generation key
    next_key_alloc = OperatorAllocation(
        num_keys=1,
        start_idx=current_key_idx,
        end_idx=current_key_idx + 1,
        input_count=0,  # N/A
        output_count=1,  # N/A
        operator_type="next_key",
    )
    current_key_idx += 1

    return ResourceMap(
        total_rng_budget=current_key_idx,
        selection=selection_alloc,
        crossover=crossover_alloc,
        mutation=mutation_alloc,
        next_key=next_key_alloc,
        pop_size=pop_size,
        genome_shape=genome_shape,
        key_derivation=key_derivation,
    )


def get_resource_summary(rmap: ResourceMap) -> str:
    """Generate a cascade data flow summary."""
    s = rmap.selection
    c = rmap.crossover
    m = rmap.mutation

    lines = [
        "Pipeline Resource & Flow Summary:",
        f"  Total RNG Budget: {rmap.total_rng_budget} keys",
        "",
        "  [1. SELECTION]",
        f"     In: {s.input_count} (Pop Size) -> Out: {s.output_count} indices (Parents needed)",
        f"     Keys: {s.num_keys} (Slice {s.start_idx}:{s.end_idx})",
        "",
        "  [2. CROSSOVER]",
        f"     In: {c.input_count} parents ({c.input_count // 2} pairs) -> "
        f"Out: {c.output_count} offspring",
        f"     Keys: {c.num_keys} (Slice {c.start_idx}:{c.end_idx})",
        "",
        "  [3. MUTATION]",
        f"     In: {m.input_count} -> Out: {m.output_count} mutants",
        f"     Keys: {m.num_keys} (Slice {m.start_idx}:{m.end_idx})",
        "",
        "  [4. NEXT GENERATION KEY]",
        f"     Keys: {rmap.next_key.num_keys} "
        f"(Slice {rmap.next_key.start_idx}:{rmap.next_key.end_idx})",
    ]
    return "\n".join(lines)
