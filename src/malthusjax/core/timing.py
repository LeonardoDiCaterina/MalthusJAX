"""Host-side readiness barriers for measured JAX execution."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, TypeVar

import jax

T = TypeVar("T")


def block_until_ready(value: T) -> T:
    """Wait for all supported leaves, preserving the object and propagating errors.

    Registered JAX PyTrees are authoritative. Plain dataclasses and
    SimpleNamespace are additionally supported for adapter/plugin state. Other
    opaque Python objects are static leaves; arbitrary attributes are not walked.
    Real JAX arrays are batched into a single public JAX readiness call.
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
    return value
