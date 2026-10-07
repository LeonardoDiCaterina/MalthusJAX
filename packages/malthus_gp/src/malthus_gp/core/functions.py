"""Mathematical primitive functions and neural activations for Cartesian GP and dCGP.

All binary primitive functions accept (a0, a1).
Activation functions accept (x) for neural/differentiable graph nodes.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp


def func_add(a0: jax.Array, a1: jax.Array) -> jax.Array:
    return a0 + a1


def func_sub(a0: jax.Array, a1: jax.Array) -> jax.Array:
    return a0 - a1


def func_mul(a0: jax.Array, a1: jax.Array) -> jax.Array:
    return a0 * a1


def func_div(a0: jax.Array, a1: jax.Array) -> jax.Array:
    """Protected division: returns 1.0 when divisor is near zero."""
    return jnp.where(jnp.abs(a1) > 1e-6, a0 / a1, 1.0)


def func_sin(a0: jax.Array, a1: jax.Array) -> jax.Array:
    return jnp.sin(a0)


def func_cos(a0: jax.Array, a1: jax.Array) -> jax.Array:
    return jnp.cos(a0)


def func_exp(a0: jax.Array, a1: jax.Array) -> jax.Array:
    """Protected exponential with clipping to prevent overflow."""
    return jnp.exp(jnp.clip(a0, -10.0, 10.0))


def func_log(a0: jax.Array, a1: jax.Array) -> jax.Array:
    """Protected natural logarithm."""
    return jnp.log(jnp.abs(a0) + 1e-6)


def func_sqrt(a0: jax.Array, a1: jax.Array) -> jax.Array:
    """Protected square root."""
    return jnp.sqrt(jnp.abs(a0))


# Standard 4-op arithmetic set (Miller & Thomson 2000)
ARITHMETIC_4_FUNCTIONS: Tuple[Callable[[jax.Array, jax.Array], jax.Array], ...] = (
    func_add,
    func_sub,
    func_mul,
    func_div,
)
ARITHMETIC_4_NAMES: Tuple[str, ...] = ("+", "-", "*", "/")
ARITHMETIC_4_FUNCTION_NAMES = ARITHMETIC_4_NAMES

# Extended mathematical primitive set
EXTENDED_FUNCTIONS: Tuple[Callable[[jax.Array, jax.Array], jax.Array], ...] = (
    func_add,
    func_sub,
    func_mul,
    func_div,
    func_sin,
    func_cos,
    func_exp,
    func_log,
)
EXTENDED_NAMES: Tuple[str, ...] = ("+", "-", "*", "/", "sin", "cos", "exp", "log")


# Neural Activation Functions (Izzo et al. 2016, Märtens & Izzo 2019)
def act_tanh(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


def act_relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


def act_sigmoid(x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(x)


def act_linear(x: jax.Array) -> jax.Array:
    return x


def act_elu(x: jax.Array) -> jax.Array:
    return jax.nn.elu(x)


def act_sin(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


def act_cos(x: jax.Array) -> jax.Array:
    return jnp.cos(x)


def act_exp(x: jax.Array) -> jax.Array:
    return jnp.exp(jnp.clip(x, -10.0, 10.0))


ACTIVATION_FUNCTIONS: Tuple[Callable[[jax.Array], jax.Array], ...] = (
    act_tanh,
    act_relu,
    act_sigmoid,
    act_linear,
    act_elu,
    act_sin,
    act_cos,
    act_exp,
)
ACTIVATION_NAMES: Tuple[str, ...] = (
    "tanh",
    "relu",
    "sigmoid",
    "linear",
    "elu",
    "sin",
    "cos",
    "exp",
)
