"""Host-side readiness barriers for measured JAX execution."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, TypeVar

import jax

T = TypeVar("T")


def block_until_ready(value: T, *, effects: bool = False) -> T:
    """Wait for all supported leaves, preserving the object and propagating errors.

    Registered JAX PyTrees are authoritative. Plain dataclasses and
    SimpleNamespace are additionally supported for adapter/plugin state. Other
    opaque Python objects are static leaves; arbitrary attributes are not walked.
    Real JAX arrays are batched into a single public JAX readiness call.
    Benchmark boundaries additionally drain effects (such as debug callbacks)
    when ``effects=True``; these need not be dependencies of a returned array.
    """
    arrays: list[jax.Array] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, jax.Array):
            arrays.append(obj)
            return
        wait = getattr(obj, "block_until_ready", None)
        if callable(wait):
            wait()  # An AttributeError from inside a device wait is also an error.
            return
        leaves = jax.tree_util.tree_leaves(obj)
        if len(leaves) != 1 or leaves[0] is not obj:
            for leaf in leaves:
                visit(leaf)
        elif is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                visit(getattr(obj, field.name))
        elif isinstance(obj, SimpleNamespace):
            for child in vars(obj).values():
                visit(child)

    visit(value)
    jax.block_until_ready(arrays)
    if effects:
        jax.effects_barrier()
    return value


def execution_metadata(reason: str | None = None) -> dict[str, Any]:
    """Minimal phase declaration; full versioned sample records are PT4 work."""
    return {
        "status": "unavailable" if reason else "valid",
        "phase": None if reason else "prepared_first_execution",
        "reason": reason,
        "scope": "evolution_and_device_finalization; host_postprocessing_excluded",
        "steady_state_validated": False,
    }
