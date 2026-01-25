from __future__ import annotations
from typing import Any, List

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig, RegressionData
from malthusjax.core.genome.linear import LinearGenome

PROTECTED_DIV_EPS: float = 1e-6

# --- 1. Math Operators ---
# Note: Every operator must accept (x, y, z) to maintain a consistent signature 
# for jax.lax.switch, regardless of whether it uses all three arguments.

def op_add(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return x + y
def op_sub(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return x - y
def op_mult(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return x * y

def op_div(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, x / y)

def op_abs(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.abs(x)
def op_neg(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return -x
def op_sin(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.sin(jnp.pi * x)
def op_cos(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.cos(jnp.pi * x)

def op_tan(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    val = jnp.tan(jnp.pi * x)
    return jnp.where(jnp.isinf(val) | jnp.isnan(val), 0.0, val)

def op_log(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    return jnp.where(x > 0, jnp.log(x), -1.0)

def op_sqrt(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)

def op_pow(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    base = jnp.abs(x)
    exp = jnp.abs(y)
    return jnp.where(base == 0, 0.0, jnp.power(base, exp))

def op_exp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.exp(x)
def op_sign(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.sign(x)
def op_max(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.maximum(x, y)
def op_min(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.minimum(x, y)
def op_mod(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.mod(x, y)
def op_frac(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return x - jnp.floor(x)
def op_mdist(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return 0.5 * (x + y)
def op_len(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.sqrt(x**2 + y**2)

# --- 2. Logic / Bitwise Operators ---
def _bitwise_helper(a: chex.Numeric, b: chex.Numeric, func: Any) -> chex.Numeric:
    a_int = (a * 1e6).astype(jnp.int32)
    b_int = (b * 1e6).astype(jnp.int32)
    return func(a_int, b_int).astype(jnp.float32) / 1e6

def op_and(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return _bitwise_helper(x, y, jnp.bitwise_and)
def op_or(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:  return _bitwise_helper(x, y, jnp.bitwise_or)
def op_xor(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return _bitwise_helper(x, y, jnp.bitwise_xor)
def op_step(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.where(x < 0, -1.0, 1.0)

# --- 3. Ternary Operators ---
def op_if(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.where(x < 0, y, z)
def op_lerp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return x + (y - x) * z
def op_clip(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric: return jnp.clip(x, y, z)

# --- 4. Smooth Operators ---
def op_sstep(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    x_c = jnp.clip(x, 0.0, 1.0)
    return x_c**2 * (3.0 - 2.0 * x_c)

def op_sstepp(x: chex.Numeric, y: chex.Numeric, z: chex.Numeric) -> chex.Numeric:
    x_c = jnp.clip(x, 0.0, 1.0)
    return x_c**3 * (x_c * (x_c * 6.0 - 15.0) + 10.0)

TENSORGP_FUNCTIONS = (
    op_add, op_sub, op_mult, op_div, op_abs, op_neg, op_sin, op_cos, op_tan,
    op_log, op_sqrt, op_pow, op_exp, op_sign, op_max, op_min, op_mod, op_frac,
    op_mdist, op_len, op_and, op_or, op_xor, op_step, op_if, op_lerp, op_clip,
    op_sstep, op_sstepp
)

TENSORGP_NAMES: List[str] = [
    "ADD", "SUB", "MUL", "DIV", "ABS", "NEG", "SIN", "COS", "TAN",
    "LOG", "SQRT", "POW", "EXP", "SIGN", "MAX", "MIN", "MOD", "FRAC",
    "MDIST", "LEN", "AND", "OR", "XOR", "STEP", "IF", "LERP", "CLIP",
    "SSTEP", "SSTEPP"
]


@struct.dataclass
class LinearGPEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Linear GP Evaluator."""
    num_inputs: int = struct.field(pytree_node=False) # type: ignore[no-untyped-call]
    length: int = struct.field(pytree_node=False)     # type: ignore[no-untyped-call]


@struct.dataclass
class LinearGPEvaluator(BaseEvaluator[LinearGenome, LinearGPEvaluatorConfig, RegressionData]):
    """
    Evaluates linear genomes using symbiotic selection, treating each 
    instruction as a potential terminal output.
    """

    def predict_one(self, genome: LinearGenome, x_input: chex.Array) -> chex.Array:
        """Execute one genome on one input vector to get all intermediate results."""
        total_mem = self.config.num_inputs + self.config.length
        memory = jnp.zeros(total_mem).at[:self.config.num_inputs].set(x_input)

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
        _, instruction_outputs = jax.lax.scan(
            step, init_state, (genome.ops, genome.args)
        )
        return instruction_outputs

    def evaluate(self, genome: LinearGenome) -> chex.Numeric:
        """Returns the negative MSE of the best instruction (Symbiotic Selection)."""
        X, y = self.data
        all_preds = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        Y_bcast = y[:, None]
        squared_errors = jnp.square(all_preds - Y_bcast)
        mse_per_tree = jnp.mean(squared_errors, axis=0)
        best_mse = jnp.min(mse_per_tree)

        return jax.lax.select(self.config.maximize, -best_mse, best_mse)

    def get_best_instruction_fitness(self, fitness: chex.Array) -> chex.Numeric:
        """Returns scalar fitness of the best performing instruction."""
        return jnp.max(fitness)

    def get_program_prediction(
        self, genome: LinearGenome, X: chex.Array, instruction_idx: int = -1
    ) -> chex.Array:
        """Retrieves data-wide predictions from a target instruction index."""
        all_outputs = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        return all_outputs[:, instruction_idx]