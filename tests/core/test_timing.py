"""Readiness must follow PyTree structure and must never hide device errors."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import pytest

from malthusjax.core.timing import block_until_ready


class Pending:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def block_until_ready(self):
        self.events.append(self.name)


@jax.tree_util.register_pytree_node_class
class SlotsTree:
    __slots__ = ("child",)

    def __init__(self, child):
        self.child = child

    def tree_flatten(self):
        return (self.child,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


def test_registered_slots_tree_and_nested_containers_are_synchronized():
    events = []
    tree = {
        "state": SlotsTree(Pending("state", events)),
        "history": [Pending("history", events)],
        "static": (None, 1, "label"),
    }
    assert block_until_ready(tree) is tree
    assert sorted(events) == ["history", "state"]


def test_plain_dataclass_and_namespace_have_explicit_container_support():
    events = []

    @dataclass(slots=True)
    class Container:
        state: object

    block_until_ready(Container(SimpleNamespace(child=Pending("ready", events))))
    assert events == ["ready"]


@pytest.mark.parametrize("error", [RuntimeError("device failed"), AttributeError("device failed")])
def test_nested_synchronization_error_is_propagated(error):
    class Broken:
        def block_until_ready(self):
            raise error

    with pytest.raises(type(error), match="device failed"):
        block_until_ready(SimpleNamespace(child=Broken()))


def test_core_timer_encloses_all_state_and_history_waits(monkeypatch):
    import malthusjax.engine.base as base
    from tests.timing_helpers import TimingEngine

    events = []
    engine = TimingEngine(engine_params=base.AbstractEngineParams(num_generations=2))
    state = SimpleNamespace(child=Pending("input", events))
    final = SimpleNamespace(best_fitness=Pending("best", events), other=Pending("other", events))
    history = {"curve": SlotsTree(Pending("history", events))}
    monkeypatch.setattr(
        base, "_get_evolution_kernel", lambda *args, **kwargs: lambda *args: (final, history)
    )

    def clock():
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(base, "time", SimpleNamespace(perf_counter=clock))
    engine.run(state, time_it=True)
    assert events[0:2] == ["input", "clock"]
    assert events[-1] == "clock"
    assert sorted(events[2:-1]) == ["best", "history", "other"]


def test_preparation_waits_for_inputs_before_lowering(monkeypatch):
    import malthusjax.engine.base as base
    from tests.timing_helpers import TimingEngine

    events = []

    class Kernel:
        def lower(self, *args):
            events.append("lower")
            return self

        def compile(self):
            events.append("compile")
            return lambda state: state

    monkeypatch.setattr(base, "_get_evolution_kernel", lambda *args, **kwargs: Kernel())
    engine = TimingEngine(engine_params=base.AbstractEngineParams())
    engine.prepare_run(SlotsTree(Pending("input", events)))
    assert events == ["input", "lower", "compile"]


def test_universal_compatibility_wrapper_uses_registered_tree():
    from malthusjax.composer.adapters.base import _block_all_until_ready

    events = []
    _block_all_until_ready(SlotsTree(Pending("ready", events)))
    assert events == ["ready"]


def test_universal_compatibility_wrapper_propagates_nested_failure():
    from malthusjax.composer.adapters.base import _block_all_until_ready

    class Broken:
        def block_until_ready(self):
            raise RuntimeError("device error")

    with pytest.raises(RuntimeError, match="device error"):
        _block_all_until_ready(SimpleNamespace(child=Broken()))
