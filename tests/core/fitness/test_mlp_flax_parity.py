"""Flax Module Parameter Leaf-Order Equivalence Suite.

Checks leaf-order equivalence between MLPInterpreter and Flax linen models:
- Verified across multiple network depths L in {1, 2, 4, 8, 12}
- Verified across activation functions: 'tanh', 'relu', 'gelu', 'identity'
- Verified forward pass parity: max |MLP(x) - Flax(x)| <= 1e-5
- Float32 cases use fan-in-scaled kernels and nonzero random biases.
- Float64 cases retain unscaled random weights to stress deep architectures.
- Gradients must be finite and agree with precision-appropriate tolerances.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from malthusjax.core.fitness.composable.interpreters import MLPInterpreter
from malthusjax.core.genome.real_genome import RealGenome


class FlaxMLPModel(nn.Module):
    """Flax MLP using padded layer names to preserve chronological leaf order."""

    hidden: tuple[int, ...]
    output_dim: int
    activation: str
    param_dtype: type = jnp.float32

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
            x = nn.Dense(s, name=f"layer_{i:03d}", param_dtype=self.param_dtype)(x)
            if i < len(sizes) - 1:
                x = act_fn(x)
        return x


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def numeric_dtype(request):
    # Scope float64 support to these cases; restore JAX's configuration afterward.
    with jax.enable_x64(True):
        yield request.param


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
def test_mlp_flax_forward_and_gradient_parity(depth, hidden, activation, numeric_dtype):
    """Test numerical forward and gradient parity across depths and activations."""
    input_dim = 4
    output_dim = 2

    interpreter = MLPInterpreter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=hidden,
        activation=activation,
    )
    num_params = interpreter.num_params

    flax_model = FlaxMLPModel(
        hidden=hidden, output_dim=output_dim, activation=activation, param_dtype=numeric_dtype
    )

    key = jax.random.key(depth * 100 + len(activation))
    k_init, k_inputs, k_thetas = jax.random.split(key, 3)

    dummy_x = jnp.zeros((input_dim,), dtype=numeric_dtype)
    flax_params = flax_model.init(k_init, dummy_x)
    flat_init, unravel_fn = ravel_pytree(flax_params)
    assert flat_init.shape == (num_params,)

    # 50 random inputs
    inputs_batch = jax.random.normal(k_inputs, (50, input_dim), dtype=numeric_dtype)
    # 10 random parameter sets
    thetas_batch = jax.random.normal(k_thetas, (10, num_params), dtype=numeric_dtype)
    if numeric_dtype == jnp.float32:
        # Derive scales from the independent Flax parameter tree, not the
        # interpreter's unflattening logic. Unit-variance weights can amplify
        # float32 cancellation through 12 layers and obscure leaf-order checks.
        scales = jax.tree.map(
            lambda leaf: jnp.full_like(leaf, 1 / np.sqrt(leaf.shape[0]) if leaf.ndim == 2 else 0.1),
            flax_params,
        )
        flat_scales, _ = ravel_pytree(scales)
        thetas_batch = thetas_batch * flat_scales

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

        assert jnp.all(jnp.isfinite(grad_mlp))
        assert jnp.all(jnp.isfinite(grad_flax))
        # Float32 reduction order can differ across Dense and flat-parameter
        # implementations. Check relative error for large gradients while
        # keeping the absolute guard for values near zero.
        np.testing.assert_allclose(
            grad_mlp,
            grad_flax,
            rtol=1e-5 if numeric_dtype == jnp.float32 else 1e-10,
            atol=1e-4 if numeric_dtype == jnp.float32 else 1e-8,
            equal_nan=False,
            err_msg=f"Gradient mismatch for depth={depth}, act={activation}",
        )
