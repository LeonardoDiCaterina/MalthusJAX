"""QDAXReplicaMixingEmitter — Exact replica of QDAX's MixingEmitter for parity testing.

This emitter mirrors QDAX's ``MixingEmitter`` bit-for-bit, including the exact
RNG key splitting pattern, so that MalthusJAX native MAP-Elites produces
identical results to QDAX on the same seed.

Key design decisions:
  - Receives a **single key** from the engine (``num_keys() == 1``) and splits
    internally, exactly like QDAX does.
  - Uses raw ``jnp.ndarray`` genotypes internally (not Population PyTrees),
    converting at the boundary.
  - Accepts ``mutation_fn`` and ``variation_fn`` callables with the same
    signatures as QDAX: ``mutation_fn(x, key) -> x'`` and
    ``variation_fn(x1, x2, key) -> x'``.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome, RealPopulation, RealGenomeConfig
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState


@struct.dataclass
class QDAXReplicaMixingEmitter(BaseEmitter):
    """Exact replica of ``qdax.core.emitters.standard_emitters.MixingEmitter``.

    This emitter is designed to produce **bit-identical** output to QDAX's
    MixingEmitter when given the same PRNG key and repertoire state.  It
    replicates QDAX's internal key-splitting order exactly:

    - For the **variation branch**: ``jax.random.split(key, 3)`` →
      ``(sample_key_1, sample_key_2, variation_key)``
    - For the **mutation branch**: ``jax.random.split(key)`` →
      ``(sample_key, mutation_key)``
    - Both branches use the **same incoming key** (not independent keys).

    Parameters
    ----------
    mutation_fn : Callable[[jnp.ndarray, chex.PRNGKey], jnp.ndarray]
        ``(genotypes, key) -> mutated_genotypes``.  Applied to ``n_mutation``
        parents sampled from the repertoire.
    variation_fn : Callable[[jnp.ndarray, jnp.ndarray, chex.PRNGKey], jnp.ndarray]
        ``(x1, x2, key) -> offspring``.  Applied to ``n_variation`` pairs
        sampled from the repertoire.
    variation_percentage : float
        Fraction of the batch that undergoes variation (crossover).  The
        remaining fraction undergoes mutation.  Default 0.5.
    _batch_size : int
        Total number of offspring per generation.
    genome_config : RealGenomeConfig
        Genome configuration for wrapping raw arrays into ``RealPopulation``.
    """

    mutation_fn: Callable = struct.field(pytree_node=False)
    variation_fn: Callable = struct.field(pytree_node=False)
    variation_percentage: float = struct.field(pytree_node=False, default=0.5)
    _batch_size: int = struct.field(pytree_node=False, default=100)
    genome_config: Any = struct.field(pytree_node=False, default=None)

    # ---- BaseEmitter interface ----

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # We bypass the ResourceMapper pre-allocation; all splitting is internal.
        return 0

    def num_keys(self) -> int:
        # Request exactly 1 key from the engine, matching QDAX's pattern.
        return 1

    def set_input_length(self, length: int) -> "QDAXReplicaMixingEmitter":
        return self.replace(_batch_size=length)

    def init(
        self,
        key: chex.Array,
        initial_population: BasePopulation,
        params: Any = None,
    ) -> Optional[EmitterState]:
        # MixingEmitter is stateless.
        return None

    # ---- Core emit logic (exact QDAX replica) ----

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation, Optional[EmitterState]]:
        """Generate offspring — exact replica of ``MixingEmitter.emit()``.

        The key flow is replicated precisely:

        * ``key`` is the single PRNG key received from the engine.
        * Variation branch uses ``jax.random.split(key, 3)``.
        * Mutation branch uses ``jax.random.split(key)`` (same ``key``!).
        """
        key = keys[0]  # Single key, like QDAX

        n_variation = int(self._batch_size * self.variation_percentage)
        n_mutation = self._batch_size - n_variation

        x_variation = None
        x_mutation = None

        if n_variation > 0:
            sample_key_1, sample_key_2, variation_key = jax.random.split(key, 3)
            x1 = repertoire.select(sample_key_1, n_variation).genotypes
            x2 = repertoire.select(sample_key_2, n_variation).genotypes
            x_variation = self.variation_fn(x1, x2, variation_key)

        if n_mutation > 0:
            sample_key, mutation_key = jax.random.split(key)
            x1 = repertoire.select(sample_key, n_mutation).genotypes
            x_mutation = self.mutation_fn(x1, mutation_key)

        # Concatenate (exact QDAX order: variation first, then mutation)
        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree.map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        # Wrap raw genotypes into a MalthusJAX RealPopulation
        offspring_pop = self._wrap_genotypes(genotypes)
        return offspring_pop, state

    def _wrap_genotypes(self, genotypes: jnp.ndarray) -> RealPopulation:
        """Convert raw genotype array into a ``RealPopulation`` PyTree."""
        genes = RealGenome(values=genotypes)
        pop_size = genotypes.shape[0]
        return RealPopulation(
            genes=genes,
            fitness=jnp.zeros(pop_size),
            config=self.genome_config,
        )
