"""Operator Set and Symbolic Derivatives for Differentiable Evaluation."""

from typing import Callable

import jax.numpy as jnp
from flax import struct

PROTECTED_DIV_EPS: float = 1e-6


@struct.dataclass
class OperatorSet:
    """Struct to hold forward and derivative functions for differentiable evaluation."""

    op_names: tuple[str, ...] = struct.field(pytree_node=False)
    forward_fns: tuple[Callable, ...] = struct.field(pytree_node=False)
    dx1_fns: tuple[Callable, ...] = struct.field(pytree_node=False)
    dx2_fns: tuple[Callable, ...] = struct.field(pytree_node=False)
    dx3_fns: tuple[Callable, ...] = struct.field(pytree_node=False)


# -----------------------------------------------------------------------------
# Forward Functions
# -----------------------------------------------------------------------------
def add_f(x, y, z):
    return x + y


def sub_f(x, y, z):
    return x - y


def mul_f(x, y, z):
    return x * y


def div_f(x, y, z):
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, x / y)


def sin_f(x, y, z):
    return jnp.sin(jnp.pi * x)


def cos_f(x, y, z):
    return jnp.cos(jnp.pi * x)


def exp_f(x, y, z):
    return jnp.exp(jnp.clip(x, -20.0, 20.0))


def abs_f(x, y, z):
    return jnp.abs(x)


def max_f(x, y, z):
    return jnp.maximum(x, y)


def min_f(x, y, z):
    return jnp.minimum(x, y)


def sign_f(x, y, z):
    return jnp.sign(x)


# -----------------------------------------------------------------------------
# Partial Derivatives w.r.t arg 1 (dx1)
# -----------------------------------------------------------------------------
def add_dx1(x, y, z):
    return 1.0


def sub_dx1(x, y, z):
    return 1.0


def mul_dx1(x, y, z):
    return y


def div_dx1(x, y, z):
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, 1.0 / y)


def sin_dx1(x, y, z):
    return jnp.pi * jnp.cos(jnp.pi * x)


def cos_dx1(x, y, z):
    return -jnp.pi * jnp.sin(jnp.pi * x)


def exp_dx1(x, y, z):
    in_bounds = (x >= -20.0) & (x <= 20.0)
    return jnp.where(in_bounds, jnp.exp(jnp.clip(x, -20.0, 20.0)), 0.0)


def abs_dx1(x, y, z):
    return jnp.sign(x)


def max_dx1(x, y, z):
    return jnp.where(x > y, 1.0, 0.0)


def min_dx1(x, y, z):
    return jnp.where(x < y, 1.0, 0.0)


def sign_dx1(x, y, z):
    return 0.0


# -----------------------------------------------------------------------------
# Partial Derivatives w.r.t arg 2 (dx2)
# -----------------------------------------------------------------------------
def zero_d(x, y, z):
    return 0.0


def add_dx2(x, y, z):
    return 1.0


def sub_dx2(x, y, z):
    return -1.0


def mul_dx2(x, y, z):
    return x


def div_dx2(x, y, z):
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, -x / (y * y))


def max_dx2(x, y, z):
    return jnp.where(y >= x, 1.0, 0.0)


def min_dx2(x, y, z):
    return jnp.where(y >= x, 1.0, 0.0)


# -----------------------------------------------------------------------------
# Default Differentiable Operator Set
# -----------------------------------------------------------------------------
DEFAULT_DIFFERENTIABLE_OPS = OperatorSet(
    op_names=("ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "ABS", "MAX", "MIN", "SIGN"),
    forward_fns=(add_f, sub_f, mul_f, div_f, sin_f, cos_f, exp_f, abs_f, max_f, min_f, sign_f),
    dx1_fns=(
        add_dx1,
        sub_dx1,
        mul_dx1,
        div_dx1,
        sin_dx1,
        cos_dx1,
        exp_dx1,
        abs_dx1,
        max_dx1,
        min_dx1,
        sign_dx1,
    ),
    dx2_fns=(
        add_dx2,
        sub_dx2,
        mul_dx2,
        div_dx2,
        zero_d,
        zero_d,
        zero_d,
        zero_d,
        max_dx2,
        min_dx2,
        zero_d,
    ),
    dx3_fns=(zero_d,) * 11,
)
