"""Linear genetic programming evaluator and primitive operations.

Provides the LinearGPEvaluator implementation along with a suite of
vectorized arithmetic, logical and ternary operators used by the
interpreter. Operators maintain a uniform signature to facilitate
jax.lax.switch dispatch.
"""

from __future__ import annotations

from typing import Any, List

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig, RegressionData
from malthusjax.core.genome.linear_genome import LinearGenome

PROTECTED_DIV_EPS: float = 1e-6

# --- 1. Math Operators ---
# Note: Every operator must accept (x, y, z) to maintain a consistent signature
# for jax.lax.switch, regardless of whether it uses all three arguments.


def op_add(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Return the sum of *x* and *y*.

    The third parameter is ignored but kept for a consistent function
    signature required by the LGP instruction dispatcher.
    """
    return x + y


def op_sub(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Subtract *y* from *x*; *z* is ignored."""
    return x - y


def op_mult(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Multiply *x* and *y*; third argument disregarded."""
    return x * y


def op_div(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Protected division of *x* by *y* with epsilon guard.

    Returns zero when *y* is very small to avoid numerical issues.
    """
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, x / y)


def op_abs(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Return the absolute value of *x* (ignore other args)."""
    return jnp.abs(x)


def op_neg(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Negate *x*; signature preserved for dispatcher."""
    return -x


def op_sin(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Sine of *x* scaled by π; others unused."""
    return jnp.sin(jnp.pi * x)


def op_cos(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Cosine of *x* scaled by π."""
    return jnp.cos(jnp.pi * x)


def op_tan(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Tangent of *x* with infinities or NaNs clipped to zero."""
    val = jnp.tan(jnp.pi * x)
    return jnp.where(jnp.isinf(val) | jnp.isnan(val), 0.0, val)


def op_log(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Natural logarithm of *x* if positive, else -1."""
    return jnp.where(x > 0, jnp.log(x), -1.0)


def op_sqrt(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Square root of positive *x*, otherwise zero."""
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


def op_pow(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Raise ``|x|`` to the power ``|y|``, returning zero when base is zero."""
    base = jnp.abs(x)
    exp = jnp.abs(y)
    return jnp.where(base == 0, 0.0, jnp.power(base, exp))


def op_exp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Compute e to the power of *x*."""
    return jnp.exp(x)


def op_sign(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Signum of *x* (-1, 0, or 1)."""
    return jnp.sign(x)


def op_max(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Elementwise maximum of *x* and *y*."""
    return jnp.maximum(x, y)


def op_min(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Elementwise minimum of *x* and *y*."""
    return jnp.minimum(x, y)


def op_mod(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Modulo operation *x* mod *y*."""
    return jnp.mod(x, y)


def op_frac(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Fractional component of *x* (x - floor(x))."""
    return x - jnp.floor(x)


def op_mdist(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Mean distance: compute (x + y)/2."""
    return 0.5 * (x + y)


def op_len(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Compute Euclidean length ``sqrt(x**2 + y**2)``."""
    return jnp.sqrt(x**2 + y**2)


# --- 2. Logic / Bitwise Operators ---
def _bitwise_helper(a: chex.Numeric, b: chex.Numeric, func: Any) -> chex.Numeric:
    """Convert float inputs to scaled ints, apply *func*, and rescale.

    This helper permits bitwise operations on floating‑point values by
    temporarily casting to integers. It is considered internal hence the
    leading underscore.
    """
    a_int = (a * 1e6).astype(jnp.int32)
    b_int = (b * 1e6).astype(jnp.int32)
    return func(a_int, b_int).astype(jnp.float32) / 1e6


def op_and(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Bitwise AND of *x* and *y* via helper conversion."""
    return _bitwise_helper(x, y, jnp.bitwise_and)


def op_or(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Bitwise OR of *x* and *y*."""
    return _bitwise_helper(x, y, jnp.bitwise_or)


def op_xor(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Bitwise XOR of *x* and *y*."""
    return _bitwise_helper(x, y, jnp.bitwise_xor)


def op_step(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Step function: returns -1 if ``x < 0`` else 1."""
    return jnp.where(x < 0, -1.0, 1.0)


# --- 3. Ternary Operators ---
def op_if(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Ternary conditional: choose *y* if ``x < 0`` else *z*."""
    return jnp.where(x < 0, y, z)


def op_lerp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Linear interpolation between *x* and *y* controlled by *z*."""
    return x + (y - x) * z


def op_clip(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Clip *x* between bounds *y* and *z*."""
    return jnp.clip(x, y, z)


# --- 4. Smooth Operators ---
def op_sstep(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Smoothstep function: cubic polynomial in [0,1] range."""
    x_c = jnp.clip(x, 0.0, 1.0)
    return x_c**2 * (3.0 - 2.0 * x_c)


def op_sstepp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    """Smoothstep‑prime: quintic easing polynomial between 0 and 1."""
    x_c = jnp.clip(x, 0.0, 1.0)
    return x_c**3 * (x_c * (x_c * 6.0 - 15.0) + 10.0)


TENSORGP_FUNCTIONS = (
    op_add,
    op_sub,
    op_mult,
    op_div,
    op_abs,
    op_neg,
    op_sin,
    op_cos,
    op_tan,
    op_log,
    op_sqrt,
    op_pow,
    op_exp,
    op_sign,
    op_max,
    op_min,
    op_mod,
    op_frac,
    op_mdist,
    op_len,
    op_and,
    op_or,
    op_xor,
    op_step,
    op_if,
    op_lerp,
    op_clip,
    op_sstep,
    op_sstepp,
)

TENSORGP_NAMES: List[str] = [
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "ABS",
    "NEG",
    "SIN",
    "COS",
    "TAN",
    "LOG",
    "SQRT",
    "POW",
    "EXP",
    "SIGN",
    "MAX",
    "MIN",
    "MOD",
    "FRAC",
    "MDIST",
    "LEN",
    "AND",
    "OR",
    "XOR",
    "STEP",
    "IF",
    "LERP",
    "CLIP",
    "SSTEP",
    "SSTEPP",
]


from malthusjax.core.fitness.base import StochasticEvaluator, BaseEvaluatorConfig, RegressionData
from malthusjax.core.genome.linear_genome import LinearGenome


@struct.dataclass
class LinearGPEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Linear GP Evaluator."""

    num_inputs: int = struct.field(pytree_node=False, default=10)  # type: ignore[no-untyped-call]
    length: int = struct.field(pytree_node=False, default=100)  # type: ignore[no-untyped-call]
    batch_size: int | None = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    loss_function: str = struct.field(pytree_node=False, default="mse")  # type: ignore[no-untyped-call]


@struct.dataclass
class LinearGPEvaluator(StochasticEvaluator[LinearGenome, LinearGPEvaluatorConfig, RegressionData]):
    """
    Evaluates linear genomes using symbiotic selection, treating each
    instruction as a potential terminal output.
    """

    def predict_one(self, genome: LinearGenome, x_input: chex.Array) -> chex.Array:
        """Execute one genome on one input vector to get all intermediate results."""
        total_mem = self.config.num_inputs + self.config.length
        memory = jnp.zeros(total_mem).at[: self.config.num_inputs].set(x_input)

        def step(current_mem: Any, inputs: Any) -> Any:
            mem, write_idx = current_mem
            op_code, arg_indices = inputs

            args_val = jnp.take(mem, arg_indices)
            result = jax.lax.switch(
                op_code, TENSORGP_FUNCTIONS, args_val[0], args_val[1], args_val[2]
            )
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            new_mem = mem.at[write_idx].set(result)

            return (new_mem, write_idx + 1), result

        init_state = (memory, self.config.num_inputs)
        _, instruction_outputs = jax.lax.scan(step, init_state, (genome.ops, genome.args))
        return instruction_outputs

    def evaluate(self, genome: LinearGenome, rng: chex.PRNGKey | None = None) -> chex.Numeric:
        """Returns the MSE of the best instruction (Symbiotic Selection)."""
        X, y = self.data

        if self.config.batch_size is not None and rng is not None:
            indices = jax.random.choice(rng, X.shape[0], shape=(self.config.batch_size,), replace=False)
            X = X[indices]
            y = y[indices]

        all_preds = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        if self.config.loss_function == "mse":
            Y_bcast = y.reshape(-1, 1)
            squared_errors = jnp.square(all_preds - Y_bcast)
            loss_per_tree = jnp.mean(squared_errors, axis=0)
            return jnp.min(loss_per_tree)
        elif self.config.loss_function == "bce":
            probs = jax.nn.sigmoid(all_preds)
            Y_bcast = y.reshape(-1, 1)
            bce = -(Y_bcast * jnp.log(probs + 1e-7) + (1 - Y_bcast) * jnp.log(1 - probs + 1e-7))
            loss_per_tree = jnp.mean(bce, axis=0)
            return jnp.min(loss_per_tree)
        else:
            raise ValueError("Standard LinearGPEvaluator does not support CCE directly.")

    def get_best_instruction_fitness(self, fitness: chex.Array) -> chex.Numeric:
        """Returns scalar fitness of the best performing instruction."""
        return jnp.max(fitness)

    def get_program_prediction(
        self, genome: LinearGenome, X: chex.Array, instruction_idx: int = -1
    ) -> chex.Array:
        """Retrieves data-wide predictions from a target instruction index."""
        all_outputs = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        return all_outputs[:, instruction_idx]
