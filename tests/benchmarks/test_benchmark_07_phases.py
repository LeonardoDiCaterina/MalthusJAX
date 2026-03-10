"""
BENCHMARK GROUP 7 — Step Phase Breakdown (MalthusJAX only)
==========================================================

Isolate and benchmark each phase of GeneticEngine.step() independently.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.engine.schedules import TrackBest
from tests.benchmarks.conftest_benchmarks import (
    _build_malthusjax_engine,
    _malthusjax_init_and_warmup,
)


class TestStepPhaseBreakdown:
    """Isolate and benchmark each phase of GeneticEngine.step() independently.

    Explains the single-step latency gap vs evosax by attributing cost to:
      Phase 0 — PRNG allocation (4-way jr.split)
      Phase 1 — Selection  (top_k for elites + selection operator)
      Phase 2 — Reproduction (crossover vmap + mutation vmap)
      Phase 3a — Merge (dynamic_update_slice buffer reuse)
      Phase 3b — Evaluate (BBOB fitness vmap)

    All phases are JIT-compiled individually over a warm state so XLA
    cost models are representative.  pop=500, d=10 to match the config
    where the single-step gap is most visible.
    """

    _POP = 500
    _DIMS = 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        engine = _build_malthusjax_engine(self._POP, self._DIMS)
        # key = jr.PRNGKey(SEED)
        state, jit_step = _malthusjax_init_and_warmup(engine)
        self.engine = engine
        self.state = state
        self.jit_step = jit_step

    def test_full_step(self, benchmark):
        """Full engine step — baseline for phase sum."""
        state = self.state

        def _run():
            s, _ = self.jit_step(state)
            s.best_fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "00_full_step"
        benchmark(_run)

    def test_phase0_entropy(self, benchmark):
        """Phase 0: 4-way PRNG split (_allocate_entropy)."""
        state = self.state
        jit_entropy = jax.jit(self.engine._allocate_entropy)
        out = jit_entropy(state)
        out[0].block_until_ready()

        def _run():
            keys = jit_entropy(state)
            keys[0].block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "01_entropy"
        benchmark(_run)

    def test_phase1_selection(self, benchmark):
        """Phase 1: elite top_k + selection operator."""
        state = self.state
        engine = self.engine

        k_sel, _, _, _ = engine._allocate_entropy(state)

        jit_sel = jax.jit(lambda k, pop, ops, params: engine._selection_phase(k, pop, ops, params))
        _, idx = jit_sel(k_sel, state.population, state.operators, engine.engine_params)
        idx.block_until_ready()

        def _run():
            _, idx = jit_sel(k_sel, state.population, state.operators, engine.engine_params)
            idx.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "02_selection"
        benchmark(_run)

    def test_phase2_reproduction(self, benchmark):
        """Phase 2: crossover vmap + mutation vmap."""
        state = self.state
        engine = self.engine

        k_sel, k_cross, k_mut, _ = engine._allocate_entropy(state)
        _, parent_indices = engine._selection_phase(
            k_sel, state.population, state.operators, engine.engine_params
        )

        jit_repro = jax.jit(
            lambda kc, km, pidx, pop, ops, rmap: engine._reproduction_phase(
                kc, km, pidx, pop, ops, rmap
            )
        )
        out_pop = jit_repro(
            k_cross,
            k_mut,
            parent_indices,
            state.population,
            state.operators,
            state.resource_map,
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out_pop)

        def _run():
            pop = jit_repro(
                k_cross,
                k_mut,
                parent_indices,
                state.population,
                state.operators,
                state.resource_map,
            )
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), pop)

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "03_reproduction"
        benchmark(_run)

    def test_phase3b_evaluate(self, benchmark):
        """Phase 3b: fitness evaluation only."""
        state = self.state
        engine = self.engine

        jit_eval = jax.jit(lambda genes, s: engine._evaluate(genes, s))
        out = jit_eval(state.population.genes, state)
        out.fitness.block_until_ready()

        def _run():
            pop = jit_eval(state.population.genes, state)
            pop.fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "04_evaluate"
        benchmark(_run)

    def test_track_best_full_vs_light(self, benchmark):
        """Compare step latency: TrackBest.LIGHT (default) vs FULL (argmax+gather+where)."""
        engine_full = _build_malthusjax_engine(self._POP, self._DIMS, track_best=TrackBest.FULL)
        state_full, jit_full = _malthusjax_init_and_warmup(engine_full)

        engine_light = _build_malthusjax_engine(self._POP, self._DIMS, track_best=TrackBest.LIGHT)
        state_light, jit_light = _malthusjax_init_and_warmup(engine_light)

        def _run_light():
            s, _ = jit_light(state_light)
            s.best_fitness.block_until_ready()

        benchmark.group = f"phase_breakdown/pop{self._POP}_d{self._DIMS}"
        benchmark.name = "05_trackbest_light"
        benchmark(_run_light)

    def test_elite_topk_vs_argpartition(self, benchmark):
        """Direct comparison: jax.lax.top_k vs jnp.argpartition for elite selection.

        Isolates just the index-extraction kernel at pop=500, elite_k=250.
        Run this to confirm argpartition wins over top_k for elite extraction.
        """
        fitness = jax.random.uniform(jr.PRNGKey(0), (self._POP,))
        elite_k = self._POP // 2

        jit_argpart = jax.jit(lambda f: jnp.argpartition(-f, elite_k)[:elite_k])
        jit_argpart(fitness).block_until_ready()

        def _run():
            idx = jit_argpart(fitness)
            idx.block_until_ready()

        benchmark.group = f"elite_extraction/pop{self._POP}"
        benchmark.name = "argpartition"
        benchmark(_run)

    def test_elite_topk(self, benchmark):
        """Elite extraction baseline: jax.lax.top_k (O(N log N) full sort)."""
        fitness = jax.random.uniform(jr.PRNGKey(0), (self._POP,))
        elite_k = self._POP // 2

        jit_topk = jax.jit(lambda f: lax.top_k(f, elite_k)[1])
        jit_topk(fitness).block_until_ready()

        def _run():
            idx = jit_topk(fitness)
            idx.block_until_ready()

        benchmark.group = f"elite_extraction/pop{self._POP}"
        benchmark.name = "top_k"
        benchmark(_run)
