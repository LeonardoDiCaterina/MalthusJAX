"""Quality-Diversity Evaluators for LSP Architectures."""

from typing import Any, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import struct
import optax

from lsp.evaluator.cartesian import CartesianGPEvaluatorConfig
from lsp.evaluator.base import LinearGPEvaluatorConfig
from lsp.evaluator.differentiable_evaluator import (
    DifferentiableCartesianGPEvaluatorConfig,
    DifferentiableLinearGPEvaluatorConfig,
    DifferentiableRegressionData,
)
from lsp.evaluator.operator_set import DEFAULT_DIFFERENTIABLE_OPS
from lsp.genome import BasePrefixAwareGenome, PrefixGenomeConfig
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.core.genome.cartesian_genome import CartesianGenome
from malthusjax.core.genome.qd.population import QDPopulation
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.fitness.base import BaseEvaluator
from lsp.evaluator.differentiable_interpreter import predict_one_jacobian


# -------------------------------------------------------------------------
# Differentiable Cartesian GP (Pure dCGP)
# -------------------------------------------------------------------------

@struct.dataclass
class DifferentiableCartesianQDEvaluator(
    BaseQDEvaluator[CartesianGenome, DifferentiableCartesianGPEvaluatorConfig, DifferentiableRegressionData]
):
    """QD Evaluator for Pure dCGP.
    
    BDs: [Active Nodes, Output Variance]
    """

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

    def evaluate_qd(self, genome: CartesianGenome) -> Tuple[chex.Numeric, chex.Array]:
        X, Y, dY_dX = self.data

        # Forward Pass
        all_preds_v = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        all_preds_g = jax.vmap(jax.jacfwd(self.predict_one, argnums=1), in_axes=(None, 0))(genome, X)
        
        # Fitness
        squared_errors_v = jnp.square(all_preds_v - Y)
        mse_v = jnp.mean(squared_errors_v)
        
        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)
        mse_g = jnp.mean(squared_errors_g)
        
        # QD natively maximizes, but MSE is a loss. We return negative MSE.
        fitness = -(mse_v + self.config.grad_weight * mse_g)

        # BDs
        active_nodes = self.get_active_nodes(genome)
        variance = jnp.var(all_preds_v)

        return fitness, jnp.stack([active_nodes, variance])


# -------------------------------------------------------------------------
# Differentiable Linear GP (Pure dMEP)
# -------------------------------------------------------------------------

@struct.dataclass
class DifferentiableLinearQDEvaluator(
    BaseQDEvaluator[LinearGenome, DifferentiableLinearGPEvaluatorConfig, DifferentiableRegressionData]
):
    """QD Evaluator for Pure dMEP."""

    def predict_one(self, genome: LinearGenome, x_input: chex.Array) -> Tuple[chex.Array, chex.Array]:
        return predict_one_jacobian(
            genome,
            x_input,
            num_inputs=self.config.num_inputs,
            length=self.config.length,
            operator_set=DEFAULT_DIFFERENTIABLE_OPS,
        )

    def evaluate_qd(self, genome: LinearGenome) -> Tuple[chex.Numeric, chex.Array]:
        X, Y, dY_dX = self.data

        all_preds_v, all_preds_g = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        squared_errors_v = jnp.square(all_preds_v - Y)
        mse_v_per_tree = jnp.mean(squared_errors_v, axis=0)

        dY_dX_bcast = jnp.expand_dims(dY_dX, axis=1)
        squared_errors_g = jnp.square(all_preds_g - dY_dX_bcast)
        mse_g_per_tree = jnp.mean(squared_errors_g, axis=(0, 2))

        combined_loss = mse_v_per_tree + self.config.grad_weight * mse_g_per_tree
        best_idx = jnp.argmin(combined_loss)

        best_loss = combined_loss[best_idx]
        fitness = -best_loss  # Maximize negative loss

        prefix_genome = BasePrefixAwareGenome(ops=genome.ops, args=genome.args)
        config = PrefixGenomeConfig(
            length=self.config.length,
            num_inputs=self.config.num_inputs,
            num_ops=len(DEFAULT_DIFFERENTIABLE_OPS.op_names),
            max_arity=3,
        )
        ancestors = prefix_genome.get_ancestor_sets(config)
        active_nodes = ancestors[best_idx].sum() + 1.0

        # Variance of the best tree's predictions
        best_preds = all_preds_v[:, best_idx]
        variance = jnp.var(best_preds)

        return fitness, jnp.stack([active_nodes, variance])


# -------------------------------------------------------------------------
# Lamarckian QD Evaluator (dCGPANN / dMEP)
# -------------------------------------------------------------------------

@struct.dataclass
class LSMFQDEvaluator(BaseQDEvaluator[Any, Any, Any]):
    """Memetic QD wrapper applying SGD to continuous parameters before returning fitness & BDs.
    
    Overrides evaluate_population directly to support Lamarckian Write-Back.
    """

    base_evaluator: Any = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)

    def evaluate_qd(self, genome: Any) -> Tuple[chex.Numeric, chex.Array]:
        raise NotImplementedError("LSMFQDEvaluator should be called via evaluate_population.")

    def evaluate_population(
        self, population: BasePopulation[Any], key: chex.PRNGKey | None = None
    ) -> QDPopulation[Any]:
        
        if self.epochs <= 0:
            return self.base_evaluator.evaluate_population(population, key)

        def loss_fn(weights: chex.Array, biases: chex.Array, genome: Any, rng: chex.PRNGKey | None):
            temp_genome = genome.replace(weights=weights, biases=biases)
            # Ensure base_evaluator returns standard scalar fitness (loss)
            if rng is not None:
                return self.base_evaluator.evaluate(temp_genome, rng)
            else:
                return self.base_evaluator.evaluate(temp_genome)

        def opt_step(state, _):
            weights, biases, opt_state, genome, rng = state
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(weights, biases, genome, rng)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, (weights, biases))
            new_weights = optax.apply_updates(weights, updates[0])
            new_biases = optax.apply_updates(biases, updates[1])
            new_state = (new_weights, new_biases, new_opt_state, genome, rng)
            return new_state, loss

        def train_individual(genome: Any, rng: chex.PRNGKey | None):
            opt_state = self.optimizer.init((genome.weights, genome.biases))
            initial_state = (genome.weights, genome.biases, opt_state, genome, rng)
            final_state, losses = jax.lax.scan(opt_step, initial_state, None, length=self.epochs)
            final_weights, final_biases, _, _, _ = final_state
            
            final_fitness = -losses[-1] # QD Native Maximize
            return final_weights, final_biases, final_fitness

        if key is not None:
            rngs = jax.random.split(key, population.fitness.shape[0])
            new_weights, new_biases, new_fitness = jax.vmap(train_individual)(population.genes, rngs)
        else:
            def train_wrapper(g):
                return train_individual(g, None)
            new_weights, new_biases, new_fitness = jax.vmap(train_wrapper)(population.genes)

        # Lamarckian Update
        new_genes = population.genes.replace(weights=new_weights, biases=new_biases)

        # Compute BDs on the new genomes
        # We need a dedicated method on base_evaluator or a general one. 
        # For Neural Genomes, active nodes depend on genome type. 
        # For simplicity, let's proxy to the base evaluator if it has a QD equivalent, 
        # OR just assign dummy BDs if not implemented.
        def get_bds(genome):
            if hasattr(self.base_evaluator, "get_active_nodes"):
                nodes = self.base_evaluator.get_active_nodes(genome)
            else:
                nodes = 0.0 # Placeholder
            return jnp.stack([jnp.array(nodes), jnp.array(0.0)]) # Variance placeholder

        descriptors = jax.vmap(get_bds)(new_genes)

        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors

        return QDPopulation(
            genes=new_genes, 
            fitness=new_fitness, 
            config=population.config, 
            info=new_info
        )
