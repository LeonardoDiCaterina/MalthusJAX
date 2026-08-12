#!/usr/bin/env python3
"""Ecosystem Showcase: Universal Adapters (GA/ES + Quality Diversity).

Demonstrates how MalthusJAX serves as a unified meta-benchmarking harness:
1. Part 1: Custom BaseEvaluator passed to EvoSAX (SimpleGA, CMA-ES) and MalthusJAX GA.
2. Part 2: Custom BaseQDEvaluator passed to QDAX (MAP-Elites) and MalthusJAX MAP-Elites.
"""

from __future__ import annotations

import pprint
import time
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.composer import Composer
from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.qd.population import QDPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter


# ===========================================================================
# Part 1: Custom Standard Evaluator (Scalar Fitness)
# ===========================================================================
@struct.dataclass
class CustomPhysicsConfig(BaseEvaluatorConfig):
    """Configuration for custom damped harmonic oscillator loss."""

    num_dims: int = struct.field(pytree_node=False, default=10)
    target_freq: float = struct.field(pytree_node=False, default=2.5)
    damping_coef: float = struct.field(pytree_node=False, default=0.1)


@struct.dataclass
class CustomPhysicsEvaluator(BaseEvaluator[RealGenome, CustomPhysicsConfig, None]):
    """Custom user-defined JAX evaluator for standard optimization."""

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        x = genome.values
        r2 = jnp.sum(x**2)
        harmonic = jnp.cos(self.config.target_freq * x[0]) * jnp.exp(-self.config.damping_coef * r2)
        penalty = 0.05 * r2
        return penalty - harmonic


# ===========================================================================
# Part 2: Custom Quality-Diversity Evaluator (Fitness + 2D Descriptors)
# ===========================================================================
@struct.dataclass
class CustomQDPhysicsConfig(BaseEvaluatorConfig):
    """Configuration for custom Quality-Diversity evaluation."""

    num_dims: int = struct.field(pytree_node=False, default=10)


@struct.dataclass
class CustomQDPhysicsEvaluator(BaseQDEvaluator[RealGenome, CustomQDPhysicsConfig, None]):
    """Custom user-defined JAX Quality-Diversity evaluator.

    Computes both scalar fitness and 2D behavioral descriptors (feature space).
    """

    def evaluate_qd(self, genome: RealGenome) -> Tuple[chex.Numeric, chex.Array]:
        x = genome.values
        # Fitness: Rastrigin-like loss
        fitness = 10.0 * self.config.num_dims + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))

        # Behavioral Descriptors: 2D feature projection (first two dimensions)
        descriptor_1 = x[0]
        descriptor_2 = x[1]
        descriptors = jnp.stack([descriptor_1, descriptor_2])

        return fitness, descriptors


# ===========================================================================
# Main Execution Harness
# ===========================================================================
def run_part_1_standard_optimization(composer: Composer) -> None:
    print("\n" + "=" * 75)
    print("  PART 1: STANDARD OPTIMIZATION (EvoSAX + MalthusJAX)")
    print("=" * 75)

    evaluator = CustomPhysicsEvaluator(
        config=CustomPhysicsConfig(maximize=False, num_dims=10), data=None
    )
    print("Created: CustomPhysicsEvaluator (Scalar Fitness)")

    pipelines = {
        "EvoSAX (SimpleGA)": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
        },
        "EvoSAX (CMA-ES)": {
            "backend": "evosax",
            "evosax_strategy": "CMA_ES",
        },
        "MalthusJAX (Modular GA)": {
            "backend": "malthusjax",
            "selection": "tournament:num_selections=64,tournament_size=3",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.2,mutation_strength=0.1",
        },
    }

    print("\nRunning Part 1 Benchmark across 3 pipelines (250 gens, pop=128)...")
    comp = composer.compare(
        pipelines=pipelines,
        fitness=evaluator,
        pop_size=128,
        generations=250,
        genome_length=10,
        seeds=(42, 43),
        shared_initial_population=False,
        maximize=False,
        bounds=(-5.0, 5.0),
    )

    print("\nPart 1 Results Summary:")
    pprint.pprint(comp.summary_table())


def run_part_2_quality_diversity(composer: Composer) -> None:
    print("\n" + "=" * 75)
    print("  PART 2: QUALITY-DIVERSITY & MAP-ELITES (QDAX + MalthusJAX Native)")
    print("=" * 75)

    qd_evaluator = CustomQDPhysicsEvaluator(
        config=CustomQDPhysicsConfig(maximize=False, num_dims=10), data=None
    )
    print("Created: CustomQDPhysicsEvaluator (Fitness + 2D Behavior Descriptors)")

    # Build MalthusJAX native mutation emitter
    cat = OperatorCatalog()
    mutation = cat.get("gaussian:mutation_rate=1.0,mutation_strength=0.1")
    native_emitter = GeneticMutationEmitter(
        _batch_size=128,
        mutation=mutation,
        genome_config=RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0)),
    )

    qd_pipelines = {
        "QDAX (MAP-Elites)": {
            "backend": "qdax",
            "qdax_strategy": "MAPElites",
            "qdax_num_centroids": 50,
            "qdax_mutation_sigma": 0.1,
        },
        "MalthusJAX Native (MAP-Elites)": {
            "strategy": MapElitesStrategy(emitter=native_emitter, num_centroids=50)
        },
    }

    print("\nRunning Part 2 Benchmark across QD pipelines (250 gens, pop=128)...")
    comp_qd = composer.compare(
        pipelines=qd_pipelines,
        fitness=qd_evaluator,
        pop_size=128,
        generations=250,
        genome_length=10,
        seeds=(42, 43),
        shared_initial_population=False,
        maximize=False,
        bounds=(-5.0, 5.0),
    )

    print("\nPart 2 QD Results Summary (Includes Coverage & QD-Score):")
    pprint.pprint(comp_qd.summary_table())


def main() -> None:
    t0 = time.time()
    composer = Composer.create_default()

    run_part_1_standard_optimization(composer)
    run_part_2_quality_diversity(composer)

    print("\n" + "=" * 75)
    print(f"  ALL SHOWCASE DEMOS COMPLETED IN {time.time() - t0:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
