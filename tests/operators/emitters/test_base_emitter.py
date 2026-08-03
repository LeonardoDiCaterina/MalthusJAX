import pytest
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple, Optional
import chex

from malthusjax.operators.emitters.base import AtomicEmitter, EmitterState
from malthusjax.core.base import BasePopulation

# Mock classes for testing
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

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self._keys_per_op

    def set_input_length(self, length: int) -> 'MockAtomicEmitter':
        return self.replace(_batch_size=length)

    def init(self, key: chex.Array, initial_population: BasePopulation, params: Any = None) -> Optional[EmitterState]:
        return None

    def _sample_parents(self, state: Optional[EmitterState], repertoire: Any, keys: chex.Array) -> Tuple[Any, dict, Optional[EmitterState]]:
        # Dummy parent sampling
        parents = (jnp.zeros((self.batch_size, 5)),)
        metadata = {'dummy_meta': jnp.ones((self.batch_size, 2))}
        return parents, metadata, state

    def _emit_one(self, state: Optional[EmitterState], key: chex.Array, *parents, **kwargs) -> Any:
        # Dummy emission
        p1 = parents[0]
        meta = kwargs.get('dummy_meta')
        return p1 + meta.sum() + key.sum()

    def _wrap_population(self, offspring_genes: Any) -> BasePopulation:
        return MockPopulation(
            genes=offspring_genes,
            fitness=jnp.zeros(self.batch_size)
        )

def test_atomic_emitter_num_keys():
    emitter = MockAtomicEmitter(_batch_size=10, _keys_per_op=2)
    # 1 for sampling + (10 * 2) for operations
    assert emitter.num_keys_for_sampling() == 1
    assert emitter.num_keys() == 21

def test_atomic_emitter_ask():
    emitter = MockAtomicEmitter(_batch_size=10, _keys_per_op=2)
    
    # Pre-allocate flat keys array
    total_keys = emitter.num_keys()
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, total_keys)
    
    # Mock repertoire
    class MockRepertoire:
        pass
        
    repertoire = MockRepertoire()
    
    offspring_pop, next_state = emitter.ask(None, repertoire, keys)
    
    assert isinstance(offspring_pop, MockPopulation)
    assert offspring_pop.genes.shape == (10, 5)
    
def test_atomic_emitter_jit():
    emitter = MockAtomicEmitter(_batch_size=10, _keys_per_op=2)
    total_keys = emitter.num_keys()
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, total_keys)
    
    @jax.jit
    def jit_ask(keys):
        return emitter.ask(None, None, keys)
        
    offspring_pop, _ = jit_ask(keys)
    assert offspring_pop.genes.shape == (10, 5)
