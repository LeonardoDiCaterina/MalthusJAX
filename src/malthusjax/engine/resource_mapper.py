"""
Resource Mapper - Static RNG Budget Allocator & Data Flow Calculator.

Step 3 of Optimization Roadmap:
Pre-calculates exact RNG requirements and operator output shapes
to enable static allocation and precise "cascade" data flow.
"""

import logging
from enum import Enum
from typing import Any, NamedTuple, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection, _field

_logger = logging.getLogger(__name__)


class KeyDerivationStrategy(Enum):
    """
    Strategy for deriving RNG keys from master key (resource_map.get_keys).
    SPLIT: Sequential jax.random.split → uncorrelated keys, single-threaded, lower memory.
    FOLD: Parallel jax.random.fold_in with indices → deterministic keys, parallelizable,
          scales better to large key budgets (suitable for multi-device).
    Trade-off: SPLIT guaranteed uncorrelated (statistical gold standard) but blocks on
    split sequencing. FOLD parallelizable but fold_in determinism replaces randomness.
    """

    SPLIT = "split"
    FOLD = "fold_in"


class ShardingManager:
    """
    GSPMD (General and Simplified Parallelization) layout for population sharding.
    Optimizes memory layout for both single-device (layout optimization) and
    multi-device (parallelism) execution.
    Axes: matrix (pop_size, features) sharded on pop_size for data parallelism.
    vector (pop_size,) sharded on pop_size (fitness array).
    replicated: Metadata (best_genome, scalars) replicated across devices.
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
        """Create a zero-initialized array for a population using the configured sharding.

        The tensor is immediately placed on device with the desired layout
        to avoid host‑device transfers during execution.
        """
        return cast(chex.Array, jax.device_put(jnp.zeros(shape, dtype=dtype), self.pop_sharding))

    def split_key_sharded(self, key: chex.Array, num: int) -> chex.Array:
        """Produce a sharded sequence of RNG keys for multi‑device execution.

        The resulting array is placed on the population sharding mesh so that
        each device receives its own independent key stream.
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
    num_pairs: int = _field(pytree_node=False)
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
        """Derive a flat buffer of RNG keys according to the selected strategy.

        For ``SPLIT`` the method simply calls ``jax.random.split``; for
        ``FOLD`` it vmap‑folds the master key over integer indices. In either
        case the output length matches ``total_rng_budget``.
        """
        if self.key_derivation == KeyDerivationStrategy.SPLIT:
            return cast(chex.Array, jax.random.split(master_key, int(self.total_rng_budget)))
        elif self.key_derivation == KeyDerivationStrategy.FOLD:

            def _fold_in(i: chex.Array) -> chex.Array:
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
    elitism: int = 0,
    key_derivation: KeyDerivationStrategy = KeyDerivationStrategy.SPLIT,
) -> ResourceMap:
    """Calculate RNG budget and data-flow allocations for one generation.

    The function computes how many parents, pairs and offspring will be
    produced and allocates key ranges for each operator accordingly. It also
    applies safety checks (e.g. fold_in compatibility) and returns a
    :class:`ResourceMap` describing the allocations.
    ----------
    The logical flow is:
    1. Selection: pop_size → parents_needed (indices)
    2. Crossover: parents_needed → offspring_count (genomes)
    3. Mutation: offspring_count → mutant_count (genomes)
    4. Next Key: single key for next generation derivation.
    ----------
    in case of overproduction (offspring > pop_size), the excess is noted but not prevented;
    users are advised to adjust parameters to minimize waste.
    in case of odd population size in crossover, the last parent may be duplicated to form a pair;
    this results in the self-crossover of the last parent
    """
    if key_derivation == KeyDerivationStrategy.FOLD:
        prng_impl = getattr(jax.config, "jax_default_prng_impl", "threefry2x32")
        if prng_impl in ("rbg", "unsafe_rbg"):
            raise ValueError(
                f"KeyDerivationStrategy.FOLD is incompatible with PRNG impl '{prng_impl}'. "
                "fold_in is not supported for RBG/UNSAFE_RBG backends. "
                "Use KeyDerivationStrategy.SPLIT or switch to 'threefry2x32'."
            )

    current_key_idx = 0

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

    # Plan offspring only for slots not occupied by elites.
    # If callers do not provide `elitism`, fallback to selection.n_elites.
    effective_elitism = int(elitism if elitism is not None else getattr(selection, "n_elites", 0))
    target_offspring = pop_size - max(0, effective_elitism)

    offspring_per_pair = getattr(crossover, "num_offspring", 2)
    pairs_needed = (target_offspring + offspring_per_pair - 1) // offspring_per_pair
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

    cross_input_count = sel_output_count  # e.g. 18 (if pop_size=17)
    num_pairs = cross_input_count // 2  # e.g. 9

    crossover = crossover.set_input_length(num_pairs)
    cross_output_count = num_pairs * crossover.num_offspring

    overproduction = cross_output_count - target_offspring
    if overproduction > 0 and target_offspring > 0 and overproduction / target_offspring > 0.10:
        _logger.warning(
            "Crossover overproduction ratio %.1f%% (producing %d offspring for target_offspring=%d). "
            "Consider adjusting pop_size or num_offspring to reduce waste.",
            100.0 * overproduction / target_offspring,
            cross_output_count,
            target_offspring,
        )

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
        num_pairs=num_pairs,
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


def get_step_dimension_flow(
    rmap: ResourceMap,
    elitism: int = 0,
    pop_symbol: str = "n",
    genome_symbol: str = "d",
    genome_width: int | None = None,
) -> str:
    """Generate an exact phase-by-phase dimension flow for one step.

    The flow is symbolic in the population size ``n`` and genome width ``d``.
    It uses the concrete allocations in ``rmap`` to parameterize the pair count
    and offspring sizes used by the engine.
    """
    n = pop_symbol
    d = genome_symbol
    d_exact = str(genome_width) if genome_width is not None else d
    e = max(0, int(elitism))
    p = int(rmap.num_pairs)
    offspring_per_pair = int(rmap.crossover.output_count // max(1, p))
    mutation_per_item = int(rmap.mutation.output_count // max(1, rmap.mutation.input_count))
    target_keep = max(0, rmap.pop_size - e)
    pair_formula = f"p = ceil(({n} - {e}) / {offspring_per_pair}) = {p}"

    lines = [
        "Step Dimension Flow:",
        "  Let n = pop size, d = genome width, e = elitism.",
        f"  {pair_formula}",
        "",
        "  [0. ENTROPY ALLOCATION]",
        "     master key -> 4 sub-buffers",
        "     selection keys: (1, 2)",
        "     crossover keys: (1, 2)",
        "     mutation keys:   (1, 2)",
        "     next key:        (2,)",
        "",
        "  [1. SELECTION]",
        f"     fitness:        ({n},)",
        f"     parent_idx:      (2p,) = ({2 * p},)",
        f"     elite_idx:       (e,) = ({e},)",
        f"     elites_genes:    (e, d) = ({e}, {d_exact})",
        "",
        "  [2. REPRODUCTION]",
        f"     p1_idx/p2_idx:   (p,) = ({p},)",
        f"     p1_pop/p2_pop:   (p, d) = ({p}, {d_exact})",
        f"     crossover in:    (p, d) -> out: (p * {offspring_per_pair}, d)",
        f"     mutation in:     (p * {offspring_per_pair}, d) -> out: (p * {offspring_per_pair} * {mutation_per_item}, d)",
        "",
        "  [3a. MERGE]",
        f"     keep elites:     (e, d) = ({e}, {d_exact})",
        f"     keep mutants:    (n - e, d) = ({target_keep}, {d_exact})",
        f"     next_genes:      (n, d) = ({n}, {d_exact})",
        "",
        "  [3b. EVALUATE]",
        f"     genes:           (n, d) = ({n}, {d_exact})",
        f"     fitness:         (n,) = ({n},)",
    ]
    return "\n".join(lines)
