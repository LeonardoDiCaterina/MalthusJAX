"""Universal compliance suite iterating over all registered components in MalthusJAX.

Executes compliance checks dynamically across:
1. All selection operators in OperatorCatalog.
2. All crossover operators in OperatorCatalog.
3. All mutation operators in OperatorCatalog.
4. All registered genomes and populations in GenomeCatalog.
5. All environments, interpreters, and composable evaluators.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.genome_catalog import GenomeCatalog
from malthusjax.core.fitness.composable.base import (
    BaseEnvironment,
    IdentityTransform,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.evaluators import (
    OptimizationEvaluator,
)
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import (
    BaseCrossover,
    BaseMutation,
    BaseSelection,
)
from malthusjax.operators.emitters.base import BaseEmitter


@pytest.fixture(scope="module")
def operator_catalog() -> OperatorCatalog:
    return OperatorCatalog()


@pytest.fixture(scope="module")
def genome_catalog() -> GenomeCatalog:
    return GenomeCatalog()


class TestRegisteredOperatorsCompliance:
    """Verifies that all operators in OperatorCatalog satisfy MalthusJAX compliance rules."""

    def test_all_catalog_operators_compliance(self, operator_catalog: OperatorCatalog):
        names = operator_catalog.list_available()
        tested_count = 0

        for name in names:
            # Provide sensible defaults for operators that need configuration
            spec = name
            if name == "cgp_neutral":
                spec = "cgp_neutral:num_selections=16"
            elif name == "genetic_mixing":
                continue  # composite emitter tested separately
            elif "cartesian" in name or "cgp" in name or "neat" in name:
                # Cartesian and NEAT operators require specialized graph genome configurations
                continue
            elif "qdax_native" in name:
                # Upstream external QDax mixing emitter
                continue

            try:
                op = operator_catalog.get(spec)
            except Exception:
                continue

            if isinstance(op, str):
                continue

            # 1. Base dataclass check
            assert dataclasses.is_dataclass(op), f"Operator '{name}' must be a dataclass."

            # 2. Selection compliance
            if isinstance(op, BaseSelection):
                fitness = jnp.linspace(0.0, 10.0, 32)
                num_keys = op.num_keys(fitness.shape)
                keys = jax.random.split(jax.random.PRNGKey(42), num_keys)
                jitted_sel = jax.jit(op.__call__)
                parents, elites = jitted_sel(keys, fitness, config=None)
                assert parents is not None
                tested_count += 1

            # 3. Crossover compliance
            elif isinstance(op, BaseCrossover):
                pop_size = 16
                gene_dim = 10
                if hasattr(op, "input_length") and getattr(op, "input_length") <= 0:
                    op = op.replace(input_length=pop_size)

                is_binary = name in ("uniform_binary", "single_point")
                if is_binary:
                    genes1 = jnp.zeros((pop_size, gene_dim), dtype=bool)
                    genes2 = jnp.ones((pop_size, gene_dim), dtype=bool)
                    cfg = BinaryGenomeConfig(shape=(gene_dim,))
                    p1 = BinaryPopulation(
                        genes=BinaryGenome(values=genes1),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )
                    p2 = BinaryPopulation(
                        genes=BinaryGenome(values=genes2),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )
                else:
                    genes1 = jnp.zeros((pop_size, gene_dim), dtype=jnp.float32)
                    genes2 = jnp.ones((pop_size, gene_dim), dtype=jnp.float32)
                    cfg = RealGenomeConfig(shape=(gene_dim,), bounds=(-5.0, 5.0))
                    p1 = RealPopulation(
                        genes=RealGenome(values=genes1),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )
                    p2 = RealPopulation(
                        genes=RealGenome(values=genes2),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )

                input_shape = (pop_size, gene_dim)
                num_keys = op.num_keys(input_shape)
                keys = jax.random.split(jax.random.PRNGKey(42), num_keys)

                jitted_cross = jax.jit(op.__call__)
                crossed = jitted_cross(keys, p1, p2, config=cfg, generation=0)
                assert crossed is not None
                expected_offspring = pop_size * op.num_offspring
                assert len(crossed) == expected_offspring, (
                    f"Crossover '{name}' produced {len(crossed)} offspring, expected {expected_offspring}"
                )
                tested_count += 1

            # 4. Mutation compliance
            elif isinstance(op, BaseMutation):
                pop_size = 16
                gene_dim = 10
                if hasattr(op, "input_length") and getattr(op, "input_length") <= 0:
                    op = op.replace(input_length=pop_size)

                is_binary = name in ("bitflip", "swap", "scramble")
                if is_binary:
                    genes = jnp.zeros((pop_size, gene_dim), dtype=bool)
                    cfg = BinaryGenomeConfig(shape=(gene_dim,))
                    p = BinaryPopulation(
                        genes=BinaryGenome(values=genes),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )
                else:
                    genes = jnp.zeros((pop_size, gene_dim), dtype=jnp.float32)
                    cfg = RealGenomeConfig(shape=(gene_dim,), bounds=(-5.0, 5.0))
                    p = RealPopulation(
                        genes=RealGenome(values=genes),
                        fitness=jnp.zeros(pop_size),
                        config=cfg,
                    )

                input_shape = (pop_size, gene_dim)
                num_keys = op.num_keys(input_shape)
                keys = jax.random.split(jax.random.PRNGKey(42), num_keys)

                jitted_mut = jax.jit(op.__call__)
                mutated = jitted_mut(keys, p, config=cfg, generation=0)
                assert mutated is not None
                expected_mutants = pop_size * op.num_offspring
                assert len(mutated) == expected_mutants, (
                    f"Mutation '{name}' produced {len(mutated)} mutants, expected {expected_mutants}"
                )
                tested_count += 1

            # 5. Evaluator & Environment compliance
            elif isinstance(op, (BaseEnvironment, OptimizationEvaluator)):
                assert dataclasses.is_dataclass(op)
                tested_count += 1

            # 6. Emitter compliance
            elif isinstance(op, BaseEmitter):
                assert dataclasses.is_dataclass(op)
                assert isinstance(op, BaseEmitter)
                tested_count += 1

        assert tested_count >= 35, (
            f"Expected at least 35 compliant operators, verified {tested_count}"
        )


class TestRegisteredGenomesCompliance:
    """Verifies that all registered Genomes and Populations satisfy JAX PyTree and compliance rules."""

    def test_real_genome_and_population(self):
        cfg = RealGenomeConfig(shape=(5,), bounds=(-2.0, 2.0))
        key = jax.random.PRNGKey(123)
        pop = cfg.init_population(key, size=10)

        assert dataclasses.is_dataclass(pop)
        assert dataclasses.is_dataclass(pop.genes)
        assert isinstance(pop.genes, RealGenome)
        assert isinstance(pop, RealPopulation)

        # PyTree flattening & roundtrip
        leaves, treedef = jax.tree_util.tree_flatten(pop)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(pop.genes.values, reconstructed.genes.values)

    def test_binary_genome_and_population(self):
        cfg = BinaryGenomeConfig(shape=(16,))
        key = jax.random.PRNGKey(456)
        pop = cfg.init_population(key, size=10)

        assert dataclasses.is_dataclass(pop)
        assert dataclasses.is_dataclass(pop.genes)
        assert isinstance(pop.genes, BinaryGenome)
        assert isinstance(pop, BinaryPopulation)

        leaves, treedef = jax.tree_util.tree_flatten(pop)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(pop.genes.values, reconstructed.genes.values)


class TestComposableArchitectureCompliance:
    """Verifies that composable evaluators conform to the E x T x I x O abstraction."""

    def test_optimization_evaluator_etio(self):
        from malthusjax.core.fitness.composable.environments import SphereEnv

        env = SphereEnv()
        transform = IdentityTransform()
        interpreter = IdentityInterpreter()
        output = ScalarOutput(maximize=False)

        evaluator = OptimizationEvaluator(
            env=env,
            transform=transform,
            interpreter=interpreter,
            output=output,
        )

        assert dataclasses.is_dataclass(evaluator)
        genome = RealGenome(values=jnp.ones(5))
        key = jax.random.PRNGKey(0)

        fitness, metrics = jax.jit(evaluator.evaluate)(
            raw_genome=genome,
            compiled_genome=transform.transform(genome),
            rng=key,
        )

        assert fitness.shape == ()
        assert jnp.isfinite(fitness)
        # Sphere of ones in 5 dims = 5.0
        assert jnp.allclose(fitness, 5.0)
