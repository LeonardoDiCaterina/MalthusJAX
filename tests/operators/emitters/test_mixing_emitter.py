from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.operators.emitters.base import AtomicEmitter, EmitterState
from malthusjax.operators.emitters.mixing import MixingEmitter


# Re-use Mock classes
@struct.dataclass
class MockPopulation(BasePopulation):
    genes: Any = struct.field(pytree_node=True)
    fitness: chex.Array = struct.field(pytree_node=True)
    info: dict = struct.field(pytree_node=True, default_factory=dict)
    config: Any = struct.field(pytree_node=False, default=None)


@struct.dataclass
class MockAtomicEmitter(AtomicEmitter):
    _batch_size: int = struct.field(pytree_node=False, default=10)
    _keys_per_op: int = struct.field(pytree_node=False, default=2)
    _sampling_keys: int = struct.field(pytree_node=False, default=1)
    value: int = struct.field(pytree_node=False, default=0)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self._keys_per_op

    def num_keys_for_sampling(self) -> int:
        return self._sampling_keys

    def set_input_length(self, length: int) -> "MockAtomicEmitter":
        return self.replace(_batch_size=length)

    def init(
        self, key: chex.Array, initial_population: BasePopulation, params: Any = None
    ) -> Optional[EmitterState]:
        return None

    def _sample_parents(
        self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array
    ) -> Tuple[Any, dict, Optional[EmitterState]]:
        parents = (jnp.full((self.batch_size, 5), self.value),)
        metadata = {"dummy_meta": jnp.zeros((self.batch_size, 2))}
        return parents, metadata, state

    def _emit_one(self, state: Optional[EmitterState], key: chex.Array, *parents, **kwargs) -> Any:
        p1 = parents[0]
        return p1

    def _wrap_population(self, offspring_genes: Any) -> BasePopulation:
        return MockPopulation(genes=offspring_genes, fitness=jnp.zeros(self.batch_size))


def test_mixing_emitter_keys():
    em_a = MockAtomicEmitter(_batch_size=10, _keys_per_op=2, _sampling_keys=1)  # 21 keys
    em_b = MockAtomicEmitter(_batch_size=20, _keys_per_op=3, _sampling_keys=2)  # 62 keys
    mixing = MixingEmitter(emitter_a=em_a, emitter_b=em_b)

    assert mixing.batch_size == 30
    assert mixing.num_keys() == 21 + 62


def test_mixing_emitter_nested():
    em_a = MockAtomicEmitter(_batch_size=10, _keys_per_op=1, _sampling_keys=1)  # 11
    em_b = MockAtomicEmitter(_batch_size=15, _keys_per_op=1, _sampling_keys=1)  # 16
    em_c = MockAtomicEmitter(_batch_size=20, _keys_per_op=1, _sampling_keys=1)  # 21

    mix_inner = MixingEmitter(emitter_a=em_a, emitter_b=em_b)
    mix_outer = MixingEmitter(emitter_a=mix_inner, emitter_b=em_c)

    assert mix_outer.batch_size == 45
    assert mix_outer.num_keys() == 11 + 16 + 21

    total_keys = mix_outer.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(0), total_keys)

    pop, _ = mix_outer.ask(None, None, keys)
    assert pop.genes.shape == (45, 5)


def test_mixing_emitter_routing():
    # Value 1 and Value 2 to distinguish populations
    em_a = MockAtomicEmitter(_batch_size=10, _keys_per_op=1, _sampling_keys=1, value=1)
    em_b = MockAtomicEmitter(_batch_size=20, _keys_per_op=1, _sampling_keys=1, value=2)
    mixing = MixingEmitter(emitter_a=em_a, emitter_b=em_b)

    total_keys = mixing.num_keys()
    keys = jax.random.split(jax.random.PRNGKey(0), total_keys)

    pop, _ = mixing.ask(None, None, keys)

    # Check that first 10 genes have value 1, last 20 have value 2
    assert jnp.all(pop.genes[:10] == 1)
    assert jnp.all(pop.genes[10:] == 2)
