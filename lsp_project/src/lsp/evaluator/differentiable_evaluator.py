"""Differentiable Linear GP Evaluator."""

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.evaluator.base import LinearGPEvaluatorConfig
from lsp.evaluator.differentiable_interpreter import predict_one_jacobian
from lsp.evaluator.operator_set import DEFAULT_DIFFERENTIABLE_OPS
from lsp.genome import BasePrefixAwareGenome, PrefixGenomeConfig

from malthusjax.composer.decorators import register_fitness
from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.core.genome.linear_genome import LinearGenome
from lsp.evaluator.cartesian import CartesianGPEvaluatorConfig
from malthusjax.core.genome.cartesian_genome import CartesianGenome

# X, Y, dY_dX
DifferentiableRegressionData = Tuple[chex.Array, chex.Array, chex.Array]


@struct.dataclass
class DifferentiableLinearGPEvaluatorConfig(LinearGPEvaluatorConfig):
    """Configuration for Differentiable Linear GP Evaluator."""

    grad_weight: float = struct.field(pytree_node=False, default=1.0)


@register_fitness("lsp_differentiable")
@struct.dataclass
class DifferentiableLinearGPEvaluator(
    BaseEvaluator[LinearGenome, DifferentiableLinearGPEvaluatorConfig, DifferentiableRegressionData]
):
    """
    Evaluates linear genomes using symbiotic selection on both function values
    and symbolic derivatives.
    """

    def predict_one(
        self, genome: LinearGenome, x_input: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Execute one genome on one input vector to get values and Jacobians."""
        return predict_one_jacobian(
            genome,
            x_input,
            num_inputs=self.config.num_inputs,
            length=self.config.length,
            operator_set=DEFAULT_DIFFERENTIABLE_OPS,
        )

    def evaluate(self, genome: LinearGenome) -> chex.Numeric:
        """Returns the combined fitness (Value MSE + Grad Weight * Jacobian MSE) of the best instruction.

        Minimization convention: lower fitness is better.
        """
        X, Y, dY_dX = self.data

        # vmap over batch of inputs
        # X shape: (B, N)
        # all_preds_v shape: (B, L)
        # all_preds_g shape: (B, L, N)
        all_preds_v, all_preds_g = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        # 1. Compute Value MSE per instruction
        Y_bcast = Y[:, None]  # (B, 1)
        squared_errors_v = jnp.square(all_preds_v - Y_bcast)  # (B, L)
        mse_v = jnp.mean(squared_errors_v, axis=0)  # (L,)

        # 2. Compute Jacobian MSE per instruction
        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)  # (B, 1, N)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)  # (B, L, N)
        # Average over batch and features
        mse_g = jnp.mean(squared_errors_g, axis=(0, 2))  # (L,)

        # 3. Combined Fitness
        combined_fitness = mse_v + self.config.grad_weight * mse_g

        # Best instruction is the one with the lowest combined loss
        best_fitness = jnp.min(combined_fitness)

        return best_fitness

    def get_best_instruction_fitness(self, fitness: chex.Array) -> chex.Numeric:
        """Returns scalar fitness of the best performing instruction."""
        return jnp.max(fitness)

    def get_program_prediction(
        self, genome: LinearGenome, X: chex.Array, instruction_idx: int = -1
    ) -> chex.Array:
        """Retrieves data-wide predictions from a target instruction index."""
        all_preds_v, _ = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        return all_preds_v[:, instruction_idx]


@register_fitness("lsp_differentiable_mo")
@struct.dataclass
class DifferentiableMOEvaluator(
    BaseMOEvaluator[
        LinearGenome, DifferentiableLinearGPEvaluatorConfig, DifferentiableRegressionData
    ]
):
    """Multi-Objective Differentiable Evaluator optimizing Value MSE, Gradient MSE, and Active Node Count.

    Objective 0: Training Value MSE (minimize).
    Objective 1: Training Gradient MSE (minimize).
    Objective 2: Active node count (minimize).
    """

    def predict_one(
        self, genome: LinearGenome, x_input: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Execute one genome on one input vector to get values and Jacobians."""
        return predict_one_jacobian(
            genome,
            x_input,
            num_inputs=self.config.num_inputs,
            length=self.config.length,
            operator_set=DEFAULT_DIFFERENTIABLE_OPS,
        )

    def evaluate(self, genome: LinearGenome) -> chex.Array:
        """Evaluate a single genome, returning [mse_v, mse_g, active_nodes] vector."""
        X, Y, dY_dX = self.data

        # 1. Forward pass over all inputs
        all_preds_v, all_preds_g = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        # 2. Compute MSE per instruction row
        Y_bcast = Y[:, None]
        squared_errors_v = jnp.square(all_preds_v - Y_bcast)
        mse_v_per_tree = jnp.mean(squared_errors_v, axis=0)

        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)
        mse_g_per_tree = jnp.mean(squared_errors_g, axis=(0, 2))

        # 3. Identify the best performing row using combined loss
        combined_fitness = mse_v_per_tree + self.config.grad_weight * mse_g_per_tree
        best_idx = jnp.argmin(combined_fitness)

        best_mse_v = mse_v_per_tree[best_idx]
        best_mse_g = mse_g_per_tree[best_idx]

        # 4. Compute active node count for that row
        prefix_genome = BasePrefixAwareGenome(ops=genome.ops, args=genome.args)
        config = PrefixGenomeConfig(
            length=self.config.length,
            num_inputs=self.config.num_inputs,
            num_ops=len(DEFAULT_DIFFERENTIABLE_OPS.op_names),
            max_arity=3,
        )
        ancestors = prefix_genome.get_ancestor_sets(config)
        # active nodes is sum of ancestors + 1 (the row itself)
        active_nodes = ancestors[best_idx].sum() + 1.0

        # Return the three objective values
        return jnp.stack([best_mse_v, best_mse_g, active_nodes])


# -------------------------------------------------------------------------
# Differentiable Cartesian GP Evaluators (Pure dCGP)
# -------------------------------------------------------------------------

@struct.dataclass
class DifferentiableCartesianGPEvaluatorConfig(CartesianGPEvaluatorConfig):
    """Configuration for Differentiable Cartesian GP Evaluator."""

    grad_weight: float = struct.field(pytree_node=False, default=1.0)


@register_fitness("lsp_differentiable_cartesian")
@struct.dataclass
class DifferentiableCartesianGPEvaluator(
    BaseEvaluator[CartesianGenome, DifferentiableCartesianGPEvaluatorConfig, DifferentiableRegressionData]
):
    """Evaluates discrete Cartesian genomes using analytical derivatives (Pure dCGP)."""

    def predict_one(self, genome: CartesianGenome, x_input: chex.Array) -> chex.Array:
        N = self.config.genome_config.num_inputs
        L = self.config.genome_config.num_nodes
        num_outs = self.config.genome_config.num_outputs

        total_mem = N + L
        memory = jnp.zeros(total_mem).at[:N].set(x_input)

        def step(current_mem: Any, inputs: Any) -> Any:
            mem, write_idx = current_mem
            op_code, arg_indices = inputs

            args_val = jnp.take(mem, arg_indices)

            arg0 = args_val[0] if args_val.shape[0] > 0 else 0.0
            arg1 = args_val[1] if args_val.shape[0] > 1 else 0.0
            arg2 = args_val[2] if args_val.shape[0] > 2 else 0.0

            result = jax.lax.switch(op_code, DEFAULT_DIFFERENTIABLE_OPS.forward_fns, arg0, arg1, arg2)
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

            new_mem = mem.at[write_idx].set(result)

            return (new_mem, write_idx + 1), result

        init_state = (memory, N)
        (final_mem, _), _ = jax.lax.scan(step, init_state, (genome.ops, genome.args))

        outputs = jnp.take(final_mem, genome.out_nodes)
        return outputs.reshape(num_outs)

    def evaluate(self, genome: CartesianGenome) -> chex.Numeric:
        X, Y, dY_dX = self.data

        # 1. Forward Pass & Jacobians
        all_preds_v = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        all_preds_g = jax.vmap(jax.jacfwd(self.predict_one, argnums=1), in_axes=(None, 0))(genome, X)
        
        # 2. Value MSE
        squared_errors_v = jnp.square(all_preds_v - Y)
        mse_v = jnp.mean(squared_errors_v)
        
        # 3. Jacobian MSE
        # all_preds_g shape: (B, num_outs, N). dY_dX shape: (B, N)
        # We assume num_outs == 1 for gradient matching in this simple evaluator
        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)
        mse_g = jnp.mean(squared_errors_g)
        
        # 4. Combined Loss
        return mse_v + self.config.grad_weight * mse_g


@register_fitness("lsp_differentiable_cartesian_mo")
@struct.dataclass
class DifferentiableCartesianMOEvaluator(
    BaseMOEvaluator[CartesianGenome, DifferentiableCartesianGPEvaluatorConfig, DifferentiableRegressionData]
):
    """Multi-Objective Pure dCGP Evaluator."""

    def predict_one(self, genome: CartesianGenome, x_input: chex.Array) -> chex.Array:
        N = self.config.genome_config.num_inputs
        L = self.config.genome_config.num_nodes
        num_outs = self.config.genome_config.num_outputs

        total_mem = N + L
        memory = jnp.zeros(total_mem).at[:N].set(x_input)

        def step(current_mem: Any, inputs: Any) -> Any:
            mem, write_idx = current_mem
            op_code, arg_indices = inputs

            args_val = jnp.take(mem, arg_indices)

            arg0 = args_val[0] if args_val.shape[0] > 0 else 0.0
            arg1 = args_val[1] if args_val.shape[0] > 1 else 0.0
            arg2 = args_val[2] if args_val.shape[0] > 2 else 0.0

            result = jax.lax.switch(op_code, DEFAULT_DIFFERENTIABLE_OPS.forward_fns, arg0, arg1, arg2)
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

            new_mem = mem.at[write_idx].set(result)

            return (new_mem, write_idx + 1), result

        init_state = (memory, N)
        (final_mem, _), _ = jax.lax.scan(step, init_state, (genome.ops, genome.args))

        outputs = jnp.take(final_mem, genome.out_nodes)
        return outputs.reshape(num_outs)
        
    def get_active_nodes(self, genome: CartesianGenome) -> chex.Array:
        N = self.config.genome_config.num_inputs
        L = self.config.genome_config.num_nodes
        
        active = jnp.zeros(N + L, dtype=jnp.bool_)
        active = active.at[genome.out_nodes].set(True)
        
        def step(active_mask, idx):
            is_active = active_mask[idx]
            node_idx = idx - N
            args = genome.args[node_idx]
            
            def activate_args(mask):
                return mask.at[args].set(True)
                
            active_mask = jax.lax.cond(is_active, activate_args, lambda m: m, active_mask)
            return active_mask, None
            
        indices = jnp.arange(N + L - 1, N - 1, -1)
        final_mask, _ = jax.lax.scan(step, active, indices)
        
        return jnp.sum(final_mask[N:]).astype(jnp.float32)

    def evaluate(self, genome: CartesianGenome) -> chex.Array:
        X, Y, dY_dX = self.data

        # 1. Forward Pass & Jacobians
        all_preds_v = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        all_preds_g = jax.vmap(jax.jacfwd(self.predict_one, argnums=1), in_axes=(None, 0))(genome, X)
        
        # 2. Value MSE
        squared_errors_v = jnp.square(all_preds_v - Y)
        mse_v = jnp.mean(squared_errors_v)
        
        # 3. Jacobian MSE
        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)
        mse_g = jnp.mean(squared_errors_g)
        
        # 4. Active Nodes
        active_nodes = self.get_active_nodes(genome)
        
        return jnp.stack([mse_v, mse_g, active_nodes])
