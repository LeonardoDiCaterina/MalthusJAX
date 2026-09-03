import chex
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple

from lsp.genome.neural_cartesian import NeuralCartesianGenome, NeuralCartesianGenomeConfig
from lsp.operators.cartesian import CartesianMicroMutation
from malthusjax.operators.base import BaseMutation
from malthusjax.composer.decorators import register_mutation


@struct.dataclass
class NeuralCartesianWeightMutation(BaseMutation[NeuralCartesianGenome, NeuralCartesianGenomeConfig]):
    """Gaussian perturbation of the continuous weights and biases for NeuralCartesianGenome."""
    mutation_rate: float = struct.field(pytree_node=False, default=0.1)
    sigma: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 4

    def _generate_noise(
        self, keys: chex.Array, config: NeuralCartesianGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        N = config.num_nodes
        M = config.max_arity
        mask_w = jax.random.bernoulli(keys[0], p=self.mutation_rate, shape=(N, M))
        noise_w = jax.random.normal(keys[1], (N, M)) * self.sigma
        mask_b = jax.random.bernoulli(keys[2], p=self.mutation_rate, shape=(N,))
        noise_b = jax.random.normal(keys[3], (N,)) * self.sigma
        return mask_w, noise_w, mask_b, noise_b

    def _mutate_one(
        self,
        genome: NeuralCartesianGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        config: NeuralCartesianGenomeConfig,
        **_kwargs: Any,
    ) -> NeuralCartesianGenome:
        mask_w, noise_w, mask_b, noise_b = noise_data
        new_w = genome.weights + jnp.where(mask_w, noise_w, 0.0)
        new_b = genome.biases + jnp.where(mask_b, noise_b, 0.0)
        return genome.replace(weights=new_w, biases=new_b)


@register_mutation(name="cgpann_hybrid")
class CGPANNHybridMutation:
    def __init__(
        self,
        struct_mut: CartesianMicroMutation = None,
        weight_mut: NeuralCartesianWeightMutation = None,
        input_length: int = -1,
        typed_keys: bool = False,
    ):
        self.struct_mut = struct_mut or CartesianMicroMutation()
        self.weight_mut = weight_mut or NeuralCartesianWeightMutation()
        self.input_length = input_length
        self.typed_keys = typed_keys

    @property
    def num_offspring(self) -> int:
        return 1

    def replace(self, **kwargs) -> "CGPANNHybridMutation":
        current = {
            "struct_mut": self.struct_mut,
            "weight_mut": self.weight_mut,
            "input_length": self.input_length,
            "typed_keys": self.typed_keys,
        }
        current.update(kwargs)
        return CGPANNHybridMutation(**current)

    def set_input_length(self, length: int) -> "CGPANNHybridMutation":
        return CGPANNHybridMutation(self.struct_mut, self.weight_mut, length, self.typed_keys)

    def set_typed_keys(self, typed: bool) -> "CGPANNHybridMutation":
        return CGPANNHybridMutation(self.struct_mut, self.weight_mut, self.input_length, typed)

    def set_max_generations(self, n: int) -> "CGPANNHybridMutation":
        return self

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.struct_mut.num_keys_per_atomic_operation + self.weight_mut.num_keys_per_atomic_operation

    def __call__(
        self, all_keys: Any, population: Any, config: NeuralCartesianGenomeConfig, generation: int = 0
    ) -> Any:
        n_struct = self.struct_mut.num_keys_per_atomic_operation
        
        if self.typed_keys:
            def apply_one(keys: chex.Array, pop: Any) -> Any:
                k1 = keys[:n_struct]
                k2 = keys[n_struct:]
                pop_out = self.struct_mut._mutate_one(pop, self.struct_mut._generate_noise(k1, config, generation), config)
                pop_out = self.weight_mut._mutate_one(pop_out, self.weight_mut._generate_noise(k2, config, generation), config)
                return pop_out
            return jax.vmap(apply_one)(all_keys, population)
        
        # Untyped fallback
        n_weight = self.weight_mut.num_keys_per_atomic_operation
        keys_reshaped = all_keys.reshape(-1, n_struct + n_weight, 2)
        keys_struct = keys_reshaped[:, :n_struct, :].reshape(-1, 2)
        keys_weight = keys_reshaped[:, n_struct:, :].reshape(-1, 2)
        
        pop1 = self.struct_mut(keys_struct, population, config, generation)
        pop2 = self.weight_mut(keys_weight, pop1, config, generation)
        return pop2
