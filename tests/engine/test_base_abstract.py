import jax.numpy as jnp
import pytest

from malthusjax.engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    _get_evolution_kernel,
    compute_unroll_num,
    validate_engine_params,
)


class DummyEngine(AbstractEngine):
    def init_state(self, rng_key):
        raise NotImplementedError

    def step(self, state):
        raise NotImplementedError


class DummyEngineWithInterfaces(AbstractEngine):
    def init_state(self, rng_key):
        pass

    def step(self, state):
        return state, AbstractGenerationOutput(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0)
        )

    def ask(self, state):
        return self, None

    def tell(self, state, pop):
        return state


def test_engine_base_coverage():
    assert compute_unroll_num(10) == 1

    with pytest.raises(ValueError):
        validate_engine_params(AbstractEngineParams(pop_size=-1))
    with pytest.raises(ValueError):
        validate_engine_params(AbstractEngineParams(num_generations=-1))
    with pytest.raises(ValueError):
        validate_engine_params(AbstractEngineParams(elitism=200, pop_size=100))

    assert AbstractGenerationOutput.get_kpi_names() == [
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "generation",
    ]

    params = AbstractEngineParams()
    engine = DummyEngine(engine_params=params)
    assert hash(engine) == id(engine)
    assert engine == engine

    with pytest.raises(NotImplementedError):
        engine.init_state(0)
    with pytest.raises(NotImplementedError):
        engine.step(None)

    with pytest.raises(NotImplementedError):
        engine.ask_with_key(None, None)
    with pytest.raises(NotImplementedError):
        engine.tell_with_key(None, None, None)

    # Interfaces
    engine_with_interfaces = DummyEngineWithInterfaces(engine_params=params)
    engine_with_interfaces.ask_with_key(None, None)
    engine_with_interfaces.tell_with_key(None, None, None)

    # Run loops
    state = AbstractEvolutionState(
        population=None,
        best_genome=None,
        generation=0,
        best_fitness=jnp.array(0.0),
        rng_key=jnp.array([0, 0]),
    )
    new_state, out = engine_with_interfaces.debug_step(state)
    final_state, hist = engine_with_interfaces.debug_run(state)

    # get_evolution_kernel warning branch
    with pytest.warns(DeprecationWarning):
        _get_evolution_kernel(params, unroll_num=2)

    # unjitted loop
    kernel = _get_evolution_kernel(AbstractEngineParams(num_generations=2), compile_jit=False)
    final_state, hist = kernel(engine_with_interfaces, state)

    # Run block
    # test time_it and verbose
    final_state, hist, time_spent = engine_with_interfaces.run(
        state, time_it=True, compile=False, verbose=True
    )
    assert time_spent is not None

    with pytest.raises(RuntimeError):
        # engine step throws NotImplementedError
        engine.run(state, compile=False)

    # get_hlo_text
    # Can't easily test the fully compiled JIT HLO string since it requires a valid JAX graph,
    # but we can call it on a simple wrapper or just skip that line for now if it requires valid state.
