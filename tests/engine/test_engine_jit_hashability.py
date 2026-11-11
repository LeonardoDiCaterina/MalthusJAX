import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.engine.base import AbstractEngine, AbstractEngineParams
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams


def test_abstract_engine_subclass_hash_inheritance():
    """Verify that any @struct.dataclass subclass of AbstractEngine preserves identity hashing."""
    @struct.dataclass
    class DummyEngine(AbstractEngine):
        unhashable_leaf: any = struct.field(pytree_node=False)

        def init_state(self, rng_key):
            return None

        def step(self, state):
            return state, None

    arr = jnp.array([1.0, 2.0, 3.0])
    engine = DummyEngine(
        engine_params=AbstractEngineParams(pop_size=10, num_generations=5),
        unhashable_leaf=arr,
    )

    # Must be hashable via id(engine)
    h = hash(engine)
    assert h == id(engine)
    assert engine == engine
    assert engine != DummyEngine(
        engine_params=AbstractEngineParams(pop_size=10, num_generations=5),
        unhashable_leaf=arr,
    )

    # Must be accepted as static_argnums in jax.jit
    @jax.jit(static_argnums=0)
    def static_fn(eng, x):
        return x + 1.0

    res = static_fn(engine, jnp.array(4.0))
    assert float(res) == 5.0


def test_map_elites_engine_hashability_with_array_evaluator():
    """Verify MapElitesEngine hashes via id even when holding an evaluator with array leaves."""
    @struct.dataclass
    class DummyArrayHolder:
        model_arrays: any

    engine = MapElitesEngine(
        emitter=DummyArrayHolder(model_arrays=jnp.zeros((10, 10))),
        evaluator=DummyArrayHolder(model_arrays=jnp.ones((5, 5))),
        engine_params=MapElitesEngineParams(pop_size=16, num_generations=5),
    )

    h = hash(engine)
    assert h == id(engine)
    assert engine == engine

    @jax.jit(static_argnums=0)
    def static_fn(eng, x):
        return x * 2.0

    res = static_fn(engine, jnp.array(3.0))
    assert float(res) == 6.0


def test_mo_engine_hashability_with_array_evaluator():
    """Verify MOEngine hashes via id even when holding an evaluator with array leaves."""
    @struct.dataclass
    class DummyArrayHolder:
        model_arrays: any

    engine = MOEngine(
        emitter=DummyArrayHolder(model_arrays=jnp.zeros((10, 10))),
        evaluator=DummyArrayHolder(model_arrays=jnp.ones((5, 5))),
        engine_params=MOEngineParams(pop_size=16, num_generations=5),
    )

    h = hash(engine)
    assert h == id(engine)
    assert engine == engine
