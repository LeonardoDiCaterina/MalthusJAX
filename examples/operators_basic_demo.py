#!/usr/bin/env python3
"""Level 2 Genetic Operators & Emitters Demonstration.

This script demonstrates how to instantiate, execute, and verify MalthusJAX
Level 2 genetic operators (Selection, Crossover, Mutation, and Emitters)
directly on population PyTrees with explicit, deterministic PRNG key management.

Key topics covered:
1. Initializing population containers (RealPopulation, RealGenomeConfig).
2. Tournament Selection (parent & elite index allocation).
3. Uniform Crossover (pair recombination under 3-tier XLA fusion).
4. Gaussian Mutation (independent per-gene perturbation with JIT compilation).
5. Emitters (wrapping reproduction operators into Quality-Diversity ask/tell loops).
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.crossover import UniformCrossover
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.selection import TournamentSelection


def main() -> None:
    print("=" * 70)
    print("MalthusJAX Level 2: Genetic Operators & Emitters Demonstration")
    print("=" * 70)

    # Master PRNG seed
    master_key = jr.PRNGKey(42)
    k_init, k_sel, k_cross, k_mut, k_emit = jr.split(master_key, 5)

    # -------------------------------------------------------------------------
    # 1. Population & Genome Configuration Setup
    # -------------------------------------------------------------------------
    pop_size = 10
    genome_dim = 5
    bounds = (-5.0, 5.0)

    config = RealGenomeConfig(shape=(genome_dim,), bounds=bounds)
    pop = RealPopulation.init_random(k_init, config, size=pop_size)

    # Assign synthetic fitness scores (e.g. negative distance to origin)
    fitness_scores = -jnp.sum(pop.genes.values ** 2, axis=-1)
    pop = pop.replace(fitness=fitness_scores)

    print(f"\n[1] Initialized RealPopulation: size={pop_size}, dim={genome_dim}")
    print(f"    Fitness range: [{fitness_scores.min():.3f}, {fitness_scores.max():.3f}]")
    print(f"    Best individual index: {int(jnp.argmax(fitness_scores))}")

    # -------------------------------------------------------------------------
    # 2. Level 2 Selection: TournamentSelection
    # -------------------------------------------------------------------------
    num_selections = 6
    tournament_size = 3
    n_elites = 1

    selection = TournamentSelection(
        num_selections=num_selections,
        tournament_size=tournament_size,
        n_elites=n_elites,
    ).set_input_length(pop_size)

    n_sel_keys = selection.num_keys(pop.fitness.shape)
    sel_keys = jr.split(k_sel, n_sel_keys)

    # Execute selection
    parent_idx, elite_idx = selection(sel_keys, pop, config=config)
    print("\n[2] Tournament Selection:")
    print(f"    Selected parent indices: {parent_idx}")
    print(f"    Preserved elite index:   {elite_idx}")

    # -------------------------------------------------------------------------
    # 3. Level 2 Crossover: UniformCrossover
    # -------------------------------------------------------------------------
    crossover = UniformCrossover(crossover_rate=0.5, num_offspring=1)

    # Demonstrate single pair recombination
    p1 = pop.genes[parent_idx[0]]
    p2 = pop.genes[parent_idx[1]]
    k_cross_pair, k_cross_batch = jr.split(k_cross)

    child_genome = crossover.cross_single_pair(k_cross_pair, p1, p2, config=config)
    print("\n[3] Uniform Crossover:")
    print(f"    Parent 1: {p1.values}")
    print(f"    Parent 2: {p2.values}")
    print(f"    Offspring: {child_genome.values[0]}")

    # -------------------------------------------------------------------------
    # 4. Level 2 Mutation: GaussianMutation (with JIT compilation)
    # -------------------------------------------------------------------------
    mutation = GaussianMutation(
        mutation_rate=0.3,
        mutation_strength=0.1,
    )

    n_mut_keys = mutation.num_keys((pop_size,))
    mut_keys = jr.split(k_mut, n_mut_keys)

    # Verify operator purity and JIT compatibility
    jitted_mutate = jax.jit(mutation.__call__)
    mutated_pop = jitted_mutate(mut_keys, pop, config=config, generation=0)

    delta = jnp.linalg.norm(mutated_pop.genes.values - pop.genes.values)
    print("\n[4] Gaussian Mutation (JIT-compiled):")
    print(f"    Total genome perturbation norm: {float(delta):.4f}")
    print(f"    Mutated population size: {len(mutated_pop)}")

    # -------------------------------------------------------------------------
    # 5. Level 2 Emitters: GeneticMutationEmitter (Quality-Diversity Interface)
    # -------------------------------------------------------------------------
    batch_size = 8
    emitter = GeneticMutationEmitter(
        mutation=mutation,
        genome_config=config,
        _batch_size=batch_size,
    )

    total_emit_keys = emitter.num_keys()
    emit_keys = jr.split(k_emit, total_emit_keys)

    # In Quality-Diversity, repertoires (e.g. MapElitesRepertoire) provide a .sample() method.
    class MockRepertoire:
        """Lightweight container emulating a QD Repertoire archive."""
        def __init__(self, p):
            self.pop = p
        def sample(self, key, n):
            idx = jr.randint(key, (n,), 0, len(self.pop))
            return self.pop.genes.values[idx], None

    repertoire = MockRepertoire(pop)

    # Ask the emitter to sample parents and emit offspring
    offspring_pop, _ = emitter.ask(
        state=None,
        repertoire=repertoire,
        keys=emit_keys,
        generation=0,
    )

    print("\n[5] Genetic Mutation Emitter (QD Ask/Tell Pattern):")
    print(f"    Emitter batch size: {emitter.batch_size}")
    print(f"    Total budgeted keys: {total_emit_keys}")
    print(f"    Emitted offspring count: {len(offspring_pop)}")
    print(f"    Emitted genes shape: {offspring_pop.genes.values.shape}")

    print("\n" + "=" * 70)
    print("All Level 2 operator and emitter operations executed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
