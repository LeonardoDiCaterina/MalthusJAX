"""Tests for Quality-Diversity Cartesian Genetic Programming (QD-CGP).

Validates CartesianQDEvaluator, CartesianEmitter, MapElitesEngineAdapter,
IslandMapElitesAdapter, and Scikit-Learn estimator integration with MAP-Elites.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

from malthus_gp import CGPRegressor
from malthus_gp.evaluators.cartesian_qd_evaluator import (
    CartesianQDEvalConfig,
    CartesianQDEvaluator,
)
from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthus_gp.operators.cartesian_emitter import CartesianEmitter
from malthus_gp.operators.cartesian_micro_mutation import CartesianMicroMutation
from malthusjax.engine.qd.adapter import MapElitesEngineAdapter
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams


def test_cartesian_qd_evaluator():
    X = jnp.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    y = (X**2).flatten()

    cfg = CartesianGenomeConfig(num_inputs=1, num_outputs=1, num_rows=1, num_cols=15)
    interp = CartesianInterpreter(num_inputs=1, num_outputs=1)
    eval_cfg = CartesianQDEvalConfig(genome_config=cfg, loss_function="mse", maximize=False)
    evaluator = CartesianQDEvaluator(config=eval_cfg, data=(X, y), interpreter=interp)

    key = jax.random.PRNGKey(42)
    k_single, k_pop = jax.random.split(key)

    # 1. Single evaluation
    single_genome = cfg.init_population(k_single, 1).genes
    # Extract single genome from batch of 1
    single_genome = jax.tree_util.tree_map(lambda x: x[0], single_genome)
    loss, descriptors = evaluator.evaluate_qd(single_genome)

    assert jnp.isfinite(loss)
    assert descriptors.shape == (2,)
    assert 0.0 <= descriptors[0] <= 1.0  # Normalized active node complexity
    assert 0.0 <= descriptors[1] <= 1.0  # Normalized prediction spread

    # 2. Population evaluation
    pop = cfg.init_population(k_pop, 5)
    evaluated_pop = evaluator.evaluate_population(pop)
    assert evaluated_pop.fitness.shape == (5,)
    assert "descriptors" in evaluated_pop.info
    assert evaluated_pop.info["descriptors"].shape == (5, 2)


def test_cartesian_emitter():
    cfg = CartesianGenomeConfig(num_inputs=1, num_outputs=1, num_rows=1, num_cols=15)
    mut = CartesianMicroMutation(mutation_rate=0.1)
    emitter = CartesianEmitter(mutation=mut, genome_config=cfg, _batch_size=8)

    key = jax.random.PRNGKey(101)
    k_pop, k_cent, k_ask = jax.random.split(key, 3)

    pop = cfg.init_population(k_pop, 10)
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=10,
        minval=[0.0, 0.0],
        maxval=[1.0, 1.0],
        key=k_cent,
    )

    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

    rep = MapElitesRepertoire.init(
        genotypes=pop.genes,
        fitnesses=-jnp.ones(10),  # Minimization stored negated
        descriptors=jnp.ones((10, 2)) * 0.5,
        centroids=centroids,
    )

    keys = jax.random.split(k_ask, emitter.num_keys())
    offspring, _ = emitter.ask(None, rep, keys, generation=0)

    assert offspring.genes.ops.shape == (8, 15)
    assert offspring.genes.args.shape == (8, 15, 2)
    assert offspring.genes.out_nodes.shape == (8, 1)


def test_map_elites_engine_adapter():
    X = jnp.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    y = (X**2).flatten()

    cfg = CartesianGenomeConfig(num_inputs=1, num_outputs=1, num_rows=1, num_cols=15)
    interp = CartesianInterpreter(num_inputs=1, num_outputs=1)
    eval_cfg = CartesianQDEvalConfig(genome_config=cfg, loss_function="mse", maximize=False)
    evaluator = CartesianQDEvaluator(config=eval_cfg, data=(X, y), interpreter=interp)

    mut = CartesianMicroMutation(mutation_rate=0.1)
    emitter = CartesianEmitter(mutation=mut, genome_config=cfg, _batch_size=10)

    engine_params = MapElitesEngineParams(pop_size=10, num_generations=15, maximize=False)
    engine = MapElitesEngine(emitter=emitter, evaluator=evaluator, engine_params=engine_params)

    adapter = MapElitesEngineAdapter(
        engine=engine,
        genome_config=cfg,
        generations=15,
        num_centroids=15,
        num_init_cvt_samples=1000,
        maximize=False,
    )

    key = jax.random.PRNGKey(42)
    init_state = adapter.init_state(key)
    final_state, _, gens = adapter.run(init_state)

    assert gens == 15
    assert final_state.best_genome is not None
    assert final_state.best_genome.ops.shape == (15,)
    assert final_state.best_fitness is not None
    assert jnp.isfinite(final_state.best_fitness)
    assert final_state.repertoire is not None


def test_cgp_regressor_map_elites():
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = (X[:, 0] ** 2).astype(np.float32)

    model = CGPRegressor(
        engine="map_elites",
        pop_size=10,
        max_generations=20,
        n_rows=1,
        n_cols=15,
        levels_back=15,
        random_state=42,
    )

    model.fit(X, y)

    # 1. Predictions & formula
    preds = model.predict(X)
    assert preds.shape == (40,)
    assert np.all(np.isfinite(preds))

    formula_str = model.formula()
    assert isinstance(formula_str, str)
    assert len(formula_str) > 0

    # 2. Repertoire inspection
    rep = model.get_repertoire()
    assert rep is not None
    assert hasattr(rep, "genotypes")
    assert hasattr(rep, "fitnesses")


def test_cgp_regressor_island_map_elites():
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = (X[:, 0] ** 3 - 0.5 * X[:, 0]).astype(np.float32)

    model = CGPRegressor(
        engine="island_map_elites:num_islands=2,migration_interval=10,num_migrants=2",
        pop_size=10,
        max_generations=20,
        n_rows=1,
        n_cols=15,
        levels_back=15,
        random_state=42,
    )

    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (40,)
    assert np.all(np.isfinite(preds))

    formula_str = model.formula()
    assert isinstance(formula_str, str)
    assert len(formula_str) > 0

    rep = model.get_repertoire()
    assert rep is not None
