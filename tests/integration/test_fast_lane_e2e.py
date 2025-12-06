import jax
import jax.random as jr
import jax.numpy as jnp

from malthusjax import (
    GeneticEngine,
    GeneticEngineParams,
    RealGenomeConfig,
    SphereEvaluator,
    SphereConfig,
)

from malthusjax import mutation, crossover, selection
from malthusjax.engine.inspector import ExecutionMode


def test_fast_lane_end_to_end_smoke():
    # Small experiment parameters (keeps test fast)
    pop_size = 20
    elitism = 2
    num_generations = 3
    import jax
    import jax.random as jr
    import jax.numpy as jnp

    from malthusjax import GeneticEngine, GeneticEngineParams, RealGenomeConfig
    from malthusjax.core.fitness.real_evaluators import SphereEvaluator, SphereConfig
    from malthusjax.operators.mutation.real import GaussianMutation
    from malthusjax.operators.crossover.real import UniformCrossover
    from malthusjax.operators.selection.elite_pool import ElitePoolSelection
    from malthusjax.operators.base import BaseSelection
    from malthusjax.engine.inspector import ExecutionMode


    def _make_engine(pop_size=32, genome_length=10, elitism=2, num_generations=3):
        params = GeneticEngineParams(pop_size=pop_size, elitism=elitism, num_generations=num_generations)

        genome_config = RealGenomeConfig(length=genome_length, bounds=(-1.0, 1.0))
        evaluator = SphereEvaluator(SphereConfig(maximize=False))

        mut = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.05)
        cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        sel = ElitePoolSelection(num_selections=pop_size, elite_k=max(4, pop_size // 5))

        engine = GeneticEngine(
            genome_config=genome_config,
            evaluator=evaluator,
            selection=sel,
            crossover=cross,
            mutation=mut,
        )

        return engine, params, genome_config


    def test_fast_lane_detection():
        engine, params, genome_config = _make_engine(pop_size=20, genome_length=10)

        # Engine should detect that all provided operators implement the kernel interface
        assert engine.mode == ExecutionMode.FAST_LANE

        # Resource map should have computed a positive total RNG budget for stochastic operators
        rmap = engine.resource_map
        assert hasattr(rmap, 'total_rng_budget')
        assert int(rmap.total_rng_budget) > 0


    def test_fast_lane_run_end_to_end():
        pop_size = 30
        genome_length = 10
        num_generations = 10

        engine, params, genome_config = _make_engine(pop_size=pop_size, genome_length=genome_length, elitism=2, num_generations=num_generations)

        # Initialize state and run evolution (non-compiled to keep test fast and deterministic)
        key = jr.PRNGKey(0)
        state = engine.init_state(key, params)

        final_state, history, _ = engine.run(state, params, compile=False)

        # Run completed -> final_state generation should equal num_generations
        assert int(final_state.generation) == int(params.num_generations)

        # Final population genes shape must match (pop_size, genome_length)
        genes = final_state.population.genes
        # genes is a RealGenome-like pytree with attribute `values`
        assert genes.values.shape[0] == params.pop_size
        assert genes.values.shape[1] == genome_config.length

        # No NaNs in the final population
        assert not jnp.any(jnp.isnan(genes.values))

        # History should contain num_generations entries (check via first leaf)
        leaves = jax.tree_util.tree_leaves(history)
        assert leaves, "History is empty"
        first_leaf = leaves[0]
        assert first_leaf.shape[0] == num_generations


    def test_legacy_fallback_mixed():
        # Create a legacy-style selection operator (inherits BaseSelection but does not implement kernel methods)
        class MockLegacySelection(BaseSelection):
            def __init__(self, num_selections: int):
                # store as attribute so BaseSelection.num_selections is accessible
                self.num_selections = num_selections

            def __call__(self, key, fitness):
                # deterministic: pick first `num_selections` indices for simplicity
                return jnp.arange(self.num_selections, dtype=jnp.int32)

        # Build engine with one legacy operator (selection) and kernel-ready mutation/crossover
        pop_size = 24
        engine_params = GeneticEngineParams(pop_size=pop_size, elitism=2, num_generations=2)
        genome_config = RealGenomeConfig(length=10, bounds=(-1.0, 1.0))
        evaluator = SphereEvaluator(SphereConfig(maximize=False))
        mut = GaussianMutation(num_offspring=1, mutation_rate=0.2, mutation_strength=0.05)
        cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        legacy_sel = MockLegacySelection(num_selections=pop_size)

        engine = GeneticEngine(
            genome_config=genome_config,
            evaluator=evaluator,
            selection=legacy_sel,
            crossover=cross,
            mutation=mut,
        )

        # Since selection lacks kernel overrides, engine should fall back to LEGACY mode
        assert engine.mode == ExecutionMode.LEGACY
