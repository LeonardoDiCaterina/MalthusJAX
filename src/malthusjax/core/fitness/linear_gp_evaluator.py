"""
Linear Genetic Programming fitness evaluator with symbiotic evaluation.

Implements sophisticated evaluation where each instruction in a linear genome
is treated as an atomic tree, enabling symbiotic evolution where the best
sub-components compete and are selected independently.
"""

from typing import Optional, Tuple
from functools import partial
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.core.fitness.evaluators import BaseEvaluator, RegressionData
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation

PROTECTED_DIV_EPS = 1e-6


# --- 1. Math Operators ---
def op_add(x, y, z): return x + y
def op_sub(x, y, z): return x - y
def op_mult(x, y, z): return x * y

# TensorGP Div: Returns 0 if nan/inf, but standard is usually 1.0. 
# TensorGP uses `tf.math.divide_no_nan`.
def op_div(x, y, z): 
    return jnp.where(jnp.abs(y) < PROTECTED_DIV_EPS, 0.0, x / y)

def op_abs(x, y, z): return jnp.abs(x)
def op_neg(x, y, z): return -x

# TensorGP scales trig inputs by PI: cos(pi * x)
def op_sin(x, y, z): return jnp.sin(jnp.pi * x)
def op_cos(x, y, z): return jnp.cos(jnp.pi * x)
def op_tan(x, y, z): 
    # Protected tan
    val = jnp.tan(jnp.pi * x)
    return jnp.where(jnp.isinf(val) | jnp.isnan(val), 0.0, val)

# Protected Log: log(x) if x > 0 else -1
def op_log(x, y, z): 
    return jnp.where(x > 0, jnp.log(x), -1.0)

# Protected Sqrt: sqrt(x) if x > 0 else 0
def op_sqrt(x, y, z):
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)

# Protected Pow: |x|^|y| (returns 0 if x=0)
def op_pow(x, y, z):
    base = jnp.abs(x)
    exp = jnp.abs(y)
    return jnp.where(base == 0, 0.0, jnp.power(base, exp))

def op_exp(x, y, z): return jnp.exp(x)

def op_sign(x, y, z): return jnp.sign(x)
def op_max(x, y, z): return jnp.maximum(x, y)
def op_min(x, y, z): return jnp.minimum(x, y)
def op_mod(x, y, z): return jnp.mod(x, y)
def op_frac(x, y, z): return x - jnp.floor(x)

# Mean Distance: (x + y) / 2
def op_mdist(x, y, z): return 0.5 * (x + y)

# Length: sqrt(x^2 + y^2)
def op_len(x, y, z): return jnp.sqrt(x**2 + y**2)

# --- 2. Logic / Bitwise Operators ---
# TensorGP casts to int, scales by 1e6, does bitwise, then scales back
def _bitwise_helper(a, b, func):
    a_int = (a * 1e6).astype(jnp.int32)
    b_int = (b * 1e6).astype(jnp.int32)
    return func(a_int, b_int).astype(jnp.float32) / 1e6

def op_and(x, y, z): return _bitwise_helper(x, y, jnp.bitwise_and)
def op_or(x, y, z):  return _bitwise_helper(x, y, jnp.bitwise_or)
def op_xor(x, y, z): return _bitwise_helper(x, y, jnp.bitwise_xor)

# Step: 1 if x >= 0 else -1 (TensorGP style)
def op_step(x, y, z):
    return jnp.where(x < 0, -1.0, 1.0)

# --- 3. Ternary Operators (Requires max_arity=3) ---

# If x < 0 return y else z
def op_if(x, y, z): return jnp.where(x < 0, y, z)

# Linear Interpolation: x + (y - x) * z
def op_lerp(x, y, z): return x + (y - x) * z

# Clip: constrain x between y and z
def op_clip(x, y, z): return jnp.clip(x, y, z)

# --- 4. Special "Smooth" Operators (Image Gen) ---
# SmoothStep: x^2 * (3 - 2x)
def op_sstep(x, y, z):
    # TensorGP clamps input to [0,1] domain first usually, but check implementation
    x_c = jnp.clip(x, 0.0, 1.0) 
    return x_c**2 * (3.0 - 2.0 * x_c)

# Perlin SmoothStep: x^3 * (x * (x * 6 - 15) + 10)
def op_sstepp(x, y, z):
    x_c = jnp.clip(x, 0.0, 1.0)
    return x_c**3 * (x_c * (x_c * 6.0 - 15.0) + 10.0)

# The registry used by lax.switch (Index matches OpCode)
TENSORGP_FUNCTIONS = (
    op_add, op_sub, op_mult, op_div, 
    op_abs, op_neg, op_sin, op_cos, op_tan,
    op_log, op_sqrt, op_pow, op_exp,
    op_sign, op_max, op_min, op_mod, op_frac,
    op_mdist, op_len,
    op_and, op_or, op_xor, op_step,
    op_if, op_lerp, op_clip,
    op_sstep, op_sstepp
)

# Usage in Evaluator:
# res = jax.lax.switch(op_code, TENSORGP_FUNCTIONS, args[0], args[1], args[2])# The string names for rendering (Index matches OpCode in TENSORGP_FUNCTIONS)
TENSORGP_NAMES = [
    "ADD", "SUB", "MUL", "DIV", 
    "ABS", "NEG", "SIN", "COS", "TAN",
    "LOG", "SQRT", "POW", "EXP",
    "SIGN", "MAX", "MIN", "MOD", "FRAC",
    "MDIST", "LEN",
    "AND", "OR", "XOR", "STEP",
    "IF", "LERP", "CLIP",
    "SSTEP", "SSTEPP"
]


@struct.dataclass
class LinearGPEvaluator(BaseEvaluator[LinearGenome, LinearGenomeConfig, RegressionData]):
    """
    Linear Genetic Programming evaluator with symbiotic fitness.
    
    Evaluates each instruction as an atomic tree and returns fitness
    for all instructions, enabling sophisticated selection strategies
    that can pick the best sub-components of programs.
    """
    
    def predict_one(self, genome: LinearGenome, x_input: chex.Array) -> chex.Array:
        """
        Execute one genome on one input vector.
        
        Args:
            genome: Linear genome to execute
            x_input: Input vector
            
        Returns:
            Array of shape (length,) containing output of each instruction
        """
        # 1. Initialize memory: inputs + instruction outputs
        total_mem = self.config.num_inputs + self.config.length
        memory = jnp.zeros(total_mem)
        memory = memory.at[:self.config.num_inputs].set(x_input)
        
        # 2. Execute instructions sequentially
        def step(current_mem, inputs):
            mem, write_idx = current_mem
            op_code, arg_indices = inputs
            
            # Fetch arguments and execute operation
            args_val = jnp.take(mem, arg_indices)
            result = jax.lax.switch(op_code, TENSORGP_FUNCTIONS, args_val[0], args_val[1], args_val[2])
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Store result in memory
            new_mem = mem.at[write_idx].set(result)
            
            # Return new memory state and the instruction output
            return (new_mem, write_idx + 1), result

        init_state = (memory, self.config.num_inputs)
        
        # Execute all instructions and collect outputs
        (_, _), instruction_outputs = jax.lax.scan(
            step, init_state, (genome.ops, genome.args)
        )
        
        return instruction_outputs

    def evaluate(self, genome: LinearGenome) -> chex.Array:
        """
        Evaluate genome with symbiotic fitness.
        
        Args:
            genome: Genome to evaluate
            data: Tuple of (X, y) for regression
            
        Returns:
            Array of shape (length,) with fitness of each instruction
        """        
        # 1. Vectorize Prediction (Data Parallelism)
        all_preds = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, self.config.X)
        
        # 2. Calculate MSE for EVERY instruction column
        Y_bcast = self.config.y[:, None]
        squared_errors = (all_preds - Y_bcast) ** 2
        mse_per_tree = jnp.mean(squared_errors, axis=0)
        
        # 3. The "Symbiotic" Selection (Best Atomic Tree)
        best_mse = jnp.min(mse_per_tree)
        
        return -best_mse
    
    def evaluate_batch(self, population: LinearPopulation) -> chex.Array:
        """Evaluate entire population."""
        return jax.vmap(self.evaluate)(population.genes)

    def get_best_instruction_fitness(self, fitness: chex.Array) -> float:
        """
        Get the fitness of the best instruction in a genome.
        
        Args:
            fitness: Array of instruction fitnesses
            
        Returns:
            Scalar fitness of best instruction
        """
        return jnp.max(fitness)

    def get_program_prediction(self, genome: LinearGenome, X: chex.Array, instruction_idx: int = -1) -> chex.Array:
        """
        Get predictions from a specific instruction or the last instruction.
        
        Args:
            genome: Genome to execute
            X: Input data
            instruction_idx: Which instruction to use (-1 for last)
            
        Returns:
            Predictions from the specified instruction
        """
        all_outputs = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        return all_outputs[:, instruction_idx]