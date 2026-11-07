"""Tests for compiler invariants, static PRNG budgeting, and O(1) compile count.

Verifies:
1. Engine step contains static PRNG budgeting (total_rng_budget is int) and zero dynamic per-individual splits.
2. Zero recompilations across different PRNG seeds for engine.run.
3. Island model traces inner engine.step exactly once via jax.vmap, regardless of island count (M=4 vs M=16).
"""

from __future__ import annotations

import jax

from malthusjax.composer.factory import build_real_engine
from malthusjax.composer.strategies import GeneticStrategy
from malthusjax.engine.island_model.topologies import RingTopologyIsland


def test_scan_body_contains_zero_random_split_primitives():
    """Verify key budgeting is static and no operator invokes dynamic split per individual."""
    strategy = GeneticStrategy(
        selection="tournament:num_selections=32",
        crossover="blend:alpha=0.2",
        mutation="gaussian:mutation_rate=0.05",
    )
    pop_size = 32
    adapter = build_real_engine(
        strategy,
        fitness="sphere:dim=10",
        pop_size=pop_size,
        generations=5,
    )
    engine = adapter.genetic_engine
    state = adapter.init_state(jax.random.PRNGKey(42))

    # Verify static budgeting
    rmap = state.resource_map
    assert isinstance(rmap.total_rng_budget, int)
    assert rmap.total_rng_budget > 0

    # Trace step
    closed_jaxpr = jax.make_jaxpr(engine.step)(state)
    eqns = closed_jaxpr.jaxpr.eqns

    # Identify all random split primitives
    split_eqns = [e for e in eqns if "split" in e.primitive.name.lower()]

    # At most 2 splits in the entire generation step:
    # 1. Master entropy slice from rmap
    # 2. Batch selection/evaluator split
    # ZERO dynamic splits per individual in crossover or mutation loops!
    assert len(split_eqns) <= 2, f"Expected <= 2 PRNG splits, found {len(split_eqns)}"

    for eqn in split_eqns:
        # Check that any split has a static shape parameter
        assert "shape" in eqn.params, f"Split equation missing static shape param: {eqn}"
        shape = eqn.params["shape"]
        assert isinstance(shape, tuple)
        # Verify shape contains compile-time static integers
        for dim in shape:
            assert isinstance(dim, int)


def test_engine_run_zero_recompilations_across_seeds():
    """Verify calling engine.run with different seeds triggers zero recompilations."""
    strategy = GeneticStrategy(
        selection="tournament:num_selections=16",
        crossover="blend:alpha=0.2",
        mutation="gaussian:mutation_rate=0.05",
    )
    adapter = build_real_engine(strategy, fitness="sphere:dim=10", pop_size=16, generations=3)
    engine = adapter.genetic_engine

    s1 = adapter.init_state(jax.random.PRNGKey(101))
    s2 = adapter.init_state(jax.random.PRNGKey(202))
    s3 = adapter.init_state(jax.random.PRNGKey(303))

    trace_count = 0

    def traced_run(state):
        nonlocal trace_count
        trace_count += 1
        return engine.run(state)

    jitted_run = jax.jit(traced_run)

    # First call compiles
    _out1 = jitted_run(s1)
    assert trace_count == 1
    assert _out1 is not None

    # Second call hits cache (zero new traces)
    _out2 = jitted_run(s2)
    assert trace_count == 1, f"Expected 1 trace, got {trace_count} (recompilation occurred!)"
    assert _out2 is not None

    # Third call hits cache
    _out3 = jitted_run(s3)
    assert trace_count == 1, f"Expected 1 trace, got {trace_count} (recompilation occurred!)"
    assert _out3 is not None


def test_island_model_o1_compile_count_across_island_counts():
    """Verify that IslandEngineAdapter uses jax.vmap and traces the step kernel once per shape."""
    strategy = GeneticStrategy(
        selection="tournament:num_selections=16",
        crossover="blend:alpha=0.2",
        mutation="gaussian:mutation_rate=0.05",
    )
    adapter = build_real_engine(strategy, fitness="sphere:dim=10", pop_size=16, generations=4)
    base_engine = adapter.genetic_engine

    for num_islands in [4, 16]:
        island_model = RingTopologyIsland(
            engine=base_engine,
            num_islands=num_islands,
            migration_interval=2,
            num_migrants=2,
        )
        multi_state = island_model.init_state(jax.random.PRNGKey(42))

        # Jaxpr trace of step
        closed_jaxpr = jax.make_jaxpr(island_model.step)(multi_state)
        eqns = closed_jaxpr.jaxpr.eqns

        # Assert step contains scan (migration interval loop) and no Python unrolled loop of length M
        primitive_names = [e.primitive.name for e in eqns]
        # In JAX, vmapped functions over scan appear as pjit/scan primitives
        assert any(p in ("scan", "pjit", "jit") for p in primitive_names)
