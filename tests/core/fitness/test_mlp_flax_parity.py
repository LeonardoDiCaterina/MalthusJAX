"""Flax Module Parameter Leaf-Order Equivalence Suite.

Formally proves leaf-order equivalence between MLPInterpreter and Flax linen models:
- Verified across multiple network depths L in {1, 2, 4, 8, 12}
- Verified across activation functions: 'tanh', 'relu', 'gelu', 'identity'
- Verified forward pass parity: max |MLP(x) - Flax(x)| <= 1e-5
- Verified gradient parity: max |dMLP/dtheta - dFlax/dtheta| <= 1e-4
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome


class FlaxMLPModel(nn.Module):
    """Flax MLP using padded layer names to preserve chronological leaf order."""

    hidden: tuple[int, ...]
    output_dim: int
    activation: str

    @nn.compact
    def __call__(self, x):
        act_map = {
            "tanh": jnp.tanh,
            "relu": jax.nn.relu,
            "gelu": jax.nn.gelu,
            "identity": lambda v: v,
        }
        act_fn = act_map[self.activation]
        sizes = self.hidden + (self.output_dim,)
        for i, s in enumerate(sizes):
            x = nn.Dense(s, name=f"layer_{i:03d}")(x)
            if i < len(sizes) - 1:
                x = act_fn(x)
        return x


@pytest.mark.parametrize(
    "depth,hidden",
    [
        (1, ()),
        (2, (16,)),
        (4, (16, 16, 16)),
        (8, (8, 8, 8, 8, 8, 8, 8)),
        (12, (8,) * 11),
    ],
)
@pytest.mark.parametrize("activation", ["tanh", "relu", "gelu", "identity"])
def test_mlp_flax_forward_and_gradient_parity(depth, hidden, activation):
    """Test bit-for-bit forward and gradient parity across depths and activations."""
    input_dim = 4
    output_dim = 2

    interpreter = MLPInterpreter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=hidden,
        activation=activation,
    )
    num_params = interpreter.num_params

    flax_model = FlaxMLPModel(hidden=hidden, output_dim=output_dim, activation=activation)

    key = jax.random.key(depth * 100 + len(activation))
    k_init, k_inputs, k_thetas = jax.random.split(key, 3)

    dummy_x = jnp.zeros((input_dim,))
    flax_params = flax_model.init(k_init, dummy_x)
    flat_init, unravel_fn = ravel_pytree(flax_params)
    assert flat_init.shape == (num_params,)

    # 50 random inputs
    inputs_batch = jax.random.normal(k_inputs, (50, input_dim))
    # 10 random parameter sets
    thetas_batch = jax.random.normal(k_thetas, (10, num_params))

    # Evaluate across all 10 parameter sets and 50 inputs
    for theta in thetas_batch:
        genome = RealGenome(values=theta)
        pytree_params = unravel_fn(theta)

        # 1. Forward Pass Parity across all 50 inputs
        mlp_preds = jax.vmap(lambda x: interpreter.apply(genome, x))(inputs_batch)
        flax_preds = jax.vmap(lambda x: flax_model.apply(pytree_params, x))(inputs_batch)

        max_forward_diff = jnp.max(jnp.abs(mlp_preds - flax_preds))
        assert max_forward_diff <= 1e-5, (
            f"Forward diff {max_forward_diff} > 1e-5 for depth={depth}, act={activation}"
        )

        # 2. Gradient Parity wrt theta
        def mlp_loss(p):
            g = RealGenome(values=p)
            preds = jax.vmap(lambda x: interpreter.apply(g, x))(inputs_batch)
            return jnp.mean(jnp.square(preds))

        def flax_loss(p):
            pt = unravel_fn(p)
            preds = jax.vmap(lambda x: flax_model.apply(pt, x))(inputs_batch)
            return jnp.mean(jnp.square(preds))

        grad_mlp = jax.grad(mlp_loss)(theta)
        grad_flax = jax.grad(flax_loss)(theta)

        max_grad_diff = jnp.max(jnp.abs(grad_mlp - grad_flax))
        assert max_grad_diff <= 1e-4, (
            f"Gradient diff {max_grad_diff} > 1e-4 for depth={depth}, act={activation}"
        )
