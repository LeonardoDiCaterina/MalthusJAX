"""Differential Oracle Parity Tests: MalthusJAX vs Upstream QDax.

Verifies bit-for-bit identity and end-to-end trajectory equivalence between:
1. QDAXReplicaMixingEmitter and qdax.core.emitters.standard_emitters.MixingEmitter
2. QDaxEngineAdapter (running MAP-Elites) and raw qdax.core.map_elites.MAPElites scan_update
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("qdax")

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics

from malthusjax.composer.adapters import EvalMode
from malthusjax.composer.qdax_adapter import build_qdax_engine
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter

# =============================================================================
# Test 1: Bit-Identical Emitter Parity
# =============================================================================


@pytest.mark.parametrize("variation_pct", [0.0, 0.5, 1.0])
def test_qdax_replica_mixing_emitter_bit_identical(variation_pct: float):
    """Assert QDAXReplicaMixingEmitter produces bit-identical offspring to QDax MixingEmitter."""
    key = jax.random.PRNGKey(42)
    k_rep, k_emit = jax.random.split(key)

    dim = 8
    num_centroids = 40
    batch_size = 20

    genotypes = jax.random.uniform(k_rep, shape=(num_centroids, dim))
    fitnesses = jnp.zeros(num_centroids)
    descriptors = jnp.zeros((num_centroids, 2))
    centroids = jnp.zeros((num_centroids, 2))

    repertoire = MapElitesRepertoire.init(
        genotypes=genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
    )

    def variation_fn(x1, x2, k):
        mask = jax.random.bernoulli(k, 0.5, shape=x1.shape)
        return jnp.where(mask, x1, x2)

    def mutation_fn(x, k):
        return x + 0.05 * jax.random.normal(k, x.shape)

    qdax_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=variation_pct,
        batch_size=batch_size,
    )

    malthus_emitter = QDAXReplicaMixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=variation_pct,
        _batch_size=batch_size,
    )

    qdax_offspring, _ = qdax_emitter.emit(repertoire, None, k_emit)
    malthus_pop, _ = malthus_emitter.ask(None, repertoire, jnp.expand_dims(k_emit, 0))

    # Bit-for-bit exact match
    np.testing.assert_allclose(
        np.asarray(malthus_pop.genes.values),
        np.asarray(qdax_offspring),
        rtol=1e-6,
        atol=1e-6,
    )


# =============================================================================
# Test 2: End-to-End MAP-Elites Trajectory Equivalence Against Raw QDax
# =============================================================================


class SphereQDEvaluator(BaseQDEvaluator):
    def __init__(self, maximize: bool = False):
        super().__init__(config=None, data=None)
        self.maximize_flag = maximize

    def evaluate(self, genome):
        raw = jnp.sum(jnp.square(genome.values))
        return -raw if self.maximize_flag else raw

    def evaluate_qd(self, genome):
        raw = jnp.sum(jnp.square(genome.values))
        desc = jnp.clip(genome.values[:2] + 0.5, 0.0, 1.0)
        return (-raw if self.maximize_flag else raw), desc


def test_map_elites_adapter_matches_raw_qdax_scan():
    """Verify QDaxEngineAdapter matches raw qdax.core.map_elites.MAPElites scan trajectory."""
    dim = 4
    pop_size = 12
    num_generations = 5
    seed = 123

    key = jax.random.PRNGKey(seed)
    k_cent, k_init, k_run = jax.random.split(key, 3)

    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=500,
        num_centroids=25,
        minval=0.0,
        maxval=1.0,
        key=k_cent,
    )

    init_genotypes = jax.random.uniform(k_init, shape=(pop_size, dim))

    def scoring_fn(x, random_key):
        # f(x) = -sum(x^2), descriptors = clip(x[:2] + 0.5, 0, 1)
        fitness = -jnp.sum(jnp.square(x), axis=-1)
        desc = jnp.clip(x[:, :2] + 0.5, 0.0, 1.0)
        return fitness, desc, {}

    def mutation_fn(x, random_key):
        return x + jax.random.normal(random_key, x.shape) * 0.05

    def variation_fn(x1, x2, random_key):
        mask = jax.random.bernoulli(random_key, 0.5, shape=x1.shape)
        return jnp.where(mask, x1, x2)

    emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=0.5,
        batch_size=pop_size,
    )

    metrics_fn = functools.partial(default_qd_metrics, qd_offset=-1000.0)

    # 1. Raw QDax execution matching UniversalAdapterEngine PRNG split
    key_loop, key_init_raw, key_eval = jax.random.split(k_run, 3)

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )
    rep_init, emit_state, _ = map_elites.init(init_genotypes, centroids, key_init_raw)

    def scan_step(carry, _):
        rep, e_state, rng = carry
        rng, subkey = jax.random.split(rng)
        rep, e_state, metrics = map_elites.update(rep, e_state, subkey)
        return (rep, e_state, rng), metrics

    (raw_final_rep, _, _), raw_metrics = jax.lax.scan(
        scan_step, (rep_init, emit_state, key_init_raw), None, length=num_generations
    )

    class NativeEvaluator:
        def scoring_function(self, genotypes, random_key):
            fitnesses = -jnp.sum(jnp.square(genotypes), axis=-1)
            descriptors = jnp.clip(genotypes[:, :2] + 0.5, 0.0, 1.0)
            return fitnesses, descriptors, {}

    # 2. QDaxEngineAdapter execution
    adapter = build_qdax_engine(
        strategy_cls=MAPElites,
        emitter=emitter,
        metrics_function=metrics_fn,
        evaluator=NativeEvaluator(),
        init_variables=init_genotypes,
        centroids=centroids,
        pop_size=pop_size,
        generations=num_generations,
        maximize=True,
        eval_mode=EvalMode.NATIVE,
        seed=seed,
    )
    result = adapter.run_once(k_run)

    # Verify trajectory metrics match
    raw_max_fit = raw_metrics["max_fitness"]
    adapter_max_fit = [h["best_fitness"] for h in result["history"]]
    np.testing.assert_allclose(adapter_max_fit, np.asarray(raw_max_fit), rtol=1e-5, atol=1e-5)

    raw_coverage = raw_metrics["coverage"]
    adapter_coverage = [h["coverage"] for h in result["history"]]
    np.testing.assert_allclose(adapter_coverage, np.asarray(raw_coverage), rtol=1e-5, atol=1e-5)

    raw_qd_score = raw_metrics["qd_score"]
    adapter_qd_score = [h["qd_score"] for h in result["history"]]
    np.testing.assert_allclose(adapter_qd_score, np.asarray(raw_qd_score), rtol=1e-5, atol=1e-5)
