"""Neural Genome Mutation Operators.

Implements mutations for NeuralGenome that handle both its structural fields
(ops, args — inherited from LinearGenome) and its continuous weights field.

    - WeightMutation:       Gaussian perturbation on the weights field only.
    - ArchitectureMutation: Structural mutation (ops + args) for NeuralGenome.
    - HybridMutation:       Sequential composition of both.
"""

from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.neural import NeuralGenome, NeuralGenomeConfig
from lsp.operators.mutation import AnnealedTopologicalMutation

from malthusjax.operators.base import BaseMutation

# ---------------------------------------------------------------------------
# WeightMutation
# ---------------------------------------------------------------------------


@struct.dataclass
class WeightMutation(BaseMutation[NeuralGenome, NeuralGenomeConfig]):
    """Gaussian perturbation of the continuous weight field.

    For each (row, arg) connection, with probability ``mutation_rate``,
    the weight is perturbed by N(0, sigma) additive noise.

    Args:
        mutation_rate: Per-connection mutation probability.
        sigma:         Standard deviation of the Gaussian perturbation.
    """

    mutation_rate: float = struct.field(pytree_node=False, default=0.1)
    sigma: float = struct.field(pytree_node=False, default=0.1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # key 0: Bernoulli mask  (L × max_arity)
        # key 1: Gaussian noise  (L × max_arity)
        return 2

    def _generate_noise(
        self,
        keys: chex.Array,
        config: NeuralGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array]:
        """Tier 2 — mutation mask and Gaussian perturbations."""
        L = config.length
        max_arity = config.max_arity
        mask = jax.random.bernoulli(keys[0], p=self.mutation_rate, shape=(L, max_arity))
        noise = jax.random.normal(keys[1], (L, max_arity)) * self.sigma
        return mask, noise

    def _mutate_one(
        self,
        genome: NeuralGenome,
        noise_data: Tuple[chex.Array, chex.Array],
        config: NeuralGenomeConfig,
        **_kwargs: Any,
    ) -> NeuralGenome:
        """Tier 1 — add noise to selected weights."""
        mask, noise = noise_data
        new_weights = genome.weights + jnp.where(mask, noise, 0.0)
        return genome.replace(weights=new_weights)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ArchitectureMutation
# ---------------------------------------------------------------------------


@struct.dataclass
class ArchitectureMutation(BaseMutation[NeuralGenome, NeuralGenomeConfig]):
    """Structural mutation for NeuralGenome (ops + args, annealed input-bias).

    Wraps AnnealedTopologicalMutation semantics but operates on NeuralGenome.
    The continuous weights field is left unchanged.

    Args:
        p_input_start: Input-pointer bias at generation 0.
        p_input_end:   Input-pointer bias at max_generations.
        op_rate:       Per-gene opcode mutation probability.
        arg_rate:      Per-(gene, arity) arg pointer mutation probability.
    """

    p_input_start: float = struct.field(pytree_node=False, default=0.1)
    p_input_end: float = struct.field(pytree_node=False, default=0.9)
    op_rate: float = struct.field(pytree_node=False, default=0.1)
    arg_rate: float = struct.field(pytree_node=False, default=0.2)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 4

    def _generate_noise(
        self,
        keys: chex.Array,
        config: NeuralGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Tier 2 — delegate to AnnealedTopologicalMutation noise generation."""
        # Re-use the same logic for annealed arg sampling.
        inner = AnnealedTopologicalMutation(
            op_rate=self.op_rate,
            arg_rate=self.arg_rate,
            p_input_start=self.p_input_start,
            p_input_end=self.p_input_end,
            max_generations=self.max_generations,
        )
        return inner._generate_noise(keys, config, generation)

    def _mutate_one(
        self,
        genome: NeuralGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
        config: NeuralGenomeConfig,
        **_kwargs: Any,
    ) -> NeuralGenome:
        """Tier 1 — apply structural mutation, preserve weights."""
        op_mask, new_ops, arg_mask, new_args = noise_data

        mutated_ops = jnp.where(op_mask, new_ops, genome.ops)
        mutated_args = jnp.where(arg_mask, new_args, genome.args)

        result = genome.replace(ops=mutated_ops, args=mutated_args)
        # autocorrect fixes any out-of-range arg pointers
        corrected = result.autocorrect(config)  # type: ignore[arg-type]
        return corrected  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# HybridMutation — sequential composition (plain Python class)
# ---------------------------------------------------------------------------


class HybridMutation:
    """Sequential architecture + weight mutation.

    Applies ``arch_mutation`` first (changing topology / opcodes / arg
    pointers), then ``weight_mutation`` (perturbing continuous weights).

    This is implemented as a plain Python class (not a flax struct) because
    flax struct.dataclass cannot accommodate required non-default fields that
    follow the inherited default fields from BaseMutation.

    Usage::

        mut = HybridMutation(
            arch_mutation=ArchitectureMutation(p_input_start=0.1, p_input_end=0.9),
            weight_mutation=WeightMutation(mutation_rate=0.1, sigma=0.1),
        )
        # Then pass directly to Composer.compare() or engine operators.

    Args:
        arch_mutation:   An ArchitectureMutation instance.
        weight_mutation: A WeightMutation instance.
    """

    def __init__(
        self,
        arch_mutation: ArchitectureMutation,
        weight_mutation: WeightMutation,
        input_length: int = -1,
        typed_keys: bool = False,
    ) -> None:
        self.arch_mutation = arch_mutation
        self.weight_mutation = weight_mutation
        self.input_length = input_length
        self.typed_keys = typed_keys

    @property
    def num_offspring(self) -> int:
        return 1

    def replace(self, **kwargs) -> "HybridMutation":
        current = {
            "arch_mutation": self.arch_mutation,
            "weight_mutation": self.weight_mutation,
            "input_length": self.input_length,
            "typed_keys": self.typed_keys,
        }
        current.update(kwargs)
        return HybridMutation(**current)

    def set_input_length(self, length: int) -> "HybridMutation":
        return HybridMutation(
            self.arch_mutation,
            self.weight_mutation,
            input_length=length,
            typed_keys=self.typed_keys
        )

    def set_typed_keys(self, typed: bool) -> "HybridMutation":
        return HybridMutation(
            self.arch_mutation,
            self.weight_mutation,
            input_length=self.input_length,
            typed_keys=typed
        )

    def set_max_generations(self, n: int) -> "HybridMutation":
        return self

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return (
            self.arch_mutation.num_keys_per_atomic_operation
            + self.weight_mutation.num_keys_per_atomic_operation
        )

    def _generate_noise(
        self,
        keys: Any,
        config: NeuralGenomeConfig,
        generation: int = 0,
    ) -> Tuple[Any, Any]:
        """Split keys and delegate to each sub-operator."""
        n_arch = self.arch_mutation.num_keys_per_atomic_operation
        arch_noise = self.arch_mutation._generate_noise(keys[:n_arch], config, generation)
        weight_noise = self.weight_mutation._generate_noise(keys[n_arch:], config, generation)
        return arch_noise, weight_noise

    def _mutate_one(
        self,
        genome: NeuralGenome,
        noise_data: Tuple[Any, Any],
        config: NeuralGenomeConfig,
        **_kwargs: Any,
    ) -> NeuralGenome:
        """Apply architecture mutation then weight mutation."""
        arch_noise, weight_noise = noise_data
        genome = self.arch_mutation._mutate_one(genome, arch_noise, config)
        genome = self.weight_mutation._mutate_one(genome, weight_noise, config)
        return genome

    def __call__(
        self,
        all_keys: Any,
        population: Any,
        config: NeuralGenomeConfig,
        generation: int = 0,
    ) -> Any:
        """Delegate to arch_mutation's population-level __call__, then apply weight_mutation."""
        n_arch = self.arch_mutation.num_keys_per_atomic_operation
        n_w = self.weight_mutation.num_keys_per_atomic_operation
        total = n_arch + n_w

        if self.typed_keys:
            keys_reshaped = all_keys.reshape(-1, total)
            arch_keys = keys_reshaped[:, :n_arch].reshape(-1)
            weight_keys = keys_reshaped[:, n_arch:].reshape(-1)
        else:
            keys_reshaped = all_keys.reshape(-1, total, 2)
            arch_keys = keys_reshaped[:, :n_arch, :].reshape(-1, 2)
            weight_keys = keys_reshaped[:, n_arch:, :].reshape(-1, 2)

        # Apply architecture mutation first
        new_pop = self.arch_mutation(
            arch_keys,
            population,
            config,
            generation,
        )
        # Apply weight mutation on the result
        new_pop = self.weight_mutation(weight_keys, new_pop, config, generation)
        return new_pop


__all__ = ["WeightMutation", "ArchitectureMutation", "HybridMutation"]
