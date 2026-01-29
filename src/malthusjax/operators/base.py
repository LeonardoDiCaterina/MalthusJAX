from typing import Any, Generic, Optional, Protocol, Tuple, TypeVar, cast, runtime_checkable

import chex
import jax
from flax import struct

# 1. Strict TypeVars
G = TypeVar("G")  # Genome Data
C = TypeVar("C")  # Config Data
P = TypeVar("P", bound="PopulationProtocol[Any]")

@runtime_checkable
class PopulationProtocol(Protocol[G]):
    genes: G
    fitness: chex.Array
    def spawn_offspring(self, new_genes: G) -> "PopulationProtocol[G]": ...
    def replace(self, **kwargs: Any) -> "PopulationProtocol[G]": ...
    def __len__(self) -> int: ...

_field: Any = struct.field

# ==========================================
# 1. MUTATION
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C, P]):
    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseMutation[G, C, P]":
        return cast("BaseMutation[G, C, P]", cast(Any, self).replace(input_length=length))

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C, **kwargs: Any) -> G:
        raise NotImplementedError

    def __call__(self, all_keys: chex.Array, population: P, config: C, **kwargs: Any) -> P:
        pop_size = jax.tree_util.tree_leaves(population.genes)[0].shape[0]
        keys_reshaped = all_keys.reshape(pop_size, self.num_offspring, -1, 2)

        def _process_population(p_keys: chex.PRNGKey, p_genome: G) -> G:
            def mutatate_fn(o_keys: chex.PRNGKey) -> G:
                return self._mutate_one(o_keys, p_genome, config, **kwargs)
            return jax.vmap(mutatate_fn)(p_keys)

        nested_genes = jax.vmap(_process_population)(keys_reshaped, population.genes)
        new_genes = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), nested_genes)
        return cast(P, population.spawn_offspring(new_genes))

# ==========================================
# 2. CROSSOVER
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C, P]):
    num_offspring: int = _field(pytree_node=False, default=2) # Renamed to match ResourceMapper
    input_length: int = _field(pytree_node=False, default=-1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        num_pairs = input_shape[0]
        return num_pairs * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseCrossover[G, C, P]":
        return cast("BaseCrossover[G, C, P]", cast(Any, self).replace(input_length=length))

    def _cross_one(
        self, key: chex.PRNGKey, p1: G, p2: G, config: C, **kwargs: Any
        ) -> Tuple[G, ...]:
        raise NotImplementedError

    def __call__(self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, **kwargs: Any) -> P:
        num_pairs = jax.tree_util.tree_leaves(p1_pop.genes)[0].shape[0]
        keys_reshaped = all_keys.reshape(num_pairs, self.num_keys_per_atomic_operation, 2)

        def _process_pairs(k: chex.PRNGKey, g1: G, g2: G) -> Tuple[G, ...]:
            return self._cross_one(k, g1, g2, config, **kwargs)

        nested_offspring = jax.vmap(_process_pairs)(keys_reshaped, p1_pop.genes, p2_pop.genes)
        def reshape_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])
        new_genes = jax.tree_util.tree_map(reshape_fn, nested_offspring)
        return cast(P, p1_pop.spawn_offspring(new_genes))

# ==========================================
# 3. SELECTION
# ==========================================
@struct.dataclass
class BaseSelection(Generic[P, C]):
    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)

    def set_input_length(self, length: int) -> "BaseSelection[P, C]":
        return cast("BaseSelection[P, C]", cast(Any, self).replace(input_length=length))

    @property
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return self.num_keys_per_atomic_operation

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
        ) -> chex.Array:
        raise NotImplementedError

    def __call__(
        self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
        ) -> chex.Array:
        return self._select(keys, population.fitness, config, **kwargs)
