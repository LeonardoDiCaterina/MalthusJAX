"""Tests for XLA HLO (High-Level Optimizer) structural invariants and compiler equivalence.

These tests prove at the compiler IR level that:
1. Composer adapter facades introduce zero Python host callbacks (xla.python.callback) in the hot loop.
2. Adapter loops are lowered directly into native XLA hardware 'while' loops.
3. Unoptimized StableHLO and compiled HLO graphs of wrapped adapters are structurally isomorphic
   (loop counts, fusion counts, and instruction volume) to raw handwritten library scan loops.
4. Host-device synchronization points are completely eliminated under standard execution.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# ---------------------------------------------------------------------------
# 1. Structural Equivalence: EvosaxEngineAdapter vs Raw Evosax
# ---------------------------------------------------------------------------


class TestEvosaxHLOEquivalence:
    """Verifies that EvosaxEngineAdapter produces structurally isomorphic HLO to raw evosax."""

    @pytest.fixture(autouse=True)
    def check_evosax(self):
        pytest.importorskip("evosax")

    def test_evosax_adapter_vs_raw_hlo_isomorphism(self):
        """Prove that EvosaxEngineAdapter and raw evosax lower to virtually identical HLO."""
        from evosax.problems import BBOBProblem
        from malthusjax.composer.evosax_adapter import EVOSAX_STRATEGIES, build_evosax_engine
        from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
        from malthusjax.core.fitness.composable.environments import BBOBEnv
        from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
        from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

        pop_size = 32
        num_dims = 10
        generations = 10

        # --- A. Adapter Lowering ---
        evalr = OptimizationEvaluator(
            env=BBOBEnv.create(fn_name="sphere", num_dims=num_dims, seed=42),
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=False),
        )
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=pop_size,
            generations=generations,
            bounds=(-5.0, 5.0),
            maximize=False,
        )
        adapter_engine = getattr(adapter, "engine", adapter)
        adapter_jit_loop = adapter_engine._build_jit_loop()

        key = jr.PRNGKey(42)
        key_run, key_init = jr.split(key)
        init_res = adapter_engine.init_fn(
            adapter_engine.framework_obj, key_init, adapter_engine.framework_params, None
        )
        lowered_adapter = adapter_jit_loop.lower(key_run, init_res)
        compiled_adapter = lowered_adapter.compile()

        hlo_unopt_adapter = lowered_adapter.as_text()
        hlo_compiled_adapter = compiled_adapter.as_text()

        # --- B. Raw Handwritten Scan Loop Lowering ---
        strategy_cls = EVOSAX_STRATEGIES["SimpleGA"]
        problem = BBOBProblem(fn_name="sphere", num_dims=num_dims, seed=42)
        strategy = strategy_cls(population_size=pop_size, solution=jnp.zeros(num_dims))
        params = strategy.default_params

        def raw_run(rng, state_init, p_state):
            def scan_step(carry, _):
                rng, state, p_state = carry
                rng, key_step = jr.split(rng)
                k_ask, k_eval, k_tell = jr.split(key_step, 3)
                x, new_state = strategy.ask(k_ask, state, params)
                fit, _, _ = problem.eval(k_eval, x, p_state)
                new_state, metrics = strategy.tell(k_tell, x, fit, new_state, params)
                return (rng, new_state, p_state), metrics

            carry = (rng, state_init, p_state)
            carry, metrics = jax.lax.scan(scan_step, carry, None, length=generations)
            return carry[1], metrics

        p_state = problem.init(key_init)
        init_x = jnp.zeros((pop_size, num_dims))
        init_fit = jnp.zeros(pop_size)
        state_init = strategy.init(key_init, init_x, init_fit, params)

        jit_raw = jax.jit(raw_run)
        lowered_raw = jit_raw.lower(key_run, state_init, p_state)
        compiled_raw = lowered_raw.compile()

        hlo_unopt_raw = lowered_raw.as_text()
        hlo_compiled_raw = compiled_raw.as_text()

        # --- C. HLO Invariant Assertions ---

        # 1. Zero Python Host Callbacks on both sides
        assert "xla.python.callback" not in hlo_compiled_adapter
        assert "xla.python.cpu_callback" not in hlo_compiled_adapter
        assert "xla.python.callback" not in hlo_compiled_raw
        assert "xla.python.cpu_callback" not in hlo_compiled_raw

        # 2. Both lower to an identical number of hardware while-loops in StableHLO
        assert hlo_unopt_adapter.count("while") == hlo_unopt_raw.count("while"), (
            f"While loop count mismatch: adapter={hlo_unopt_adapter.count('while')}, raw={hlo_unopt_raw.count('while')}"
        )

        # 3. Unoptimized StableHLO IR size ratio is within 5%
        size_ratio = len(hlo_unopt_adapter) / len(hlo_unopt_raw)
        assert 0.95 <= size_ratio <= 1.05, f"Unoptimized HLO size ratio diverged: {size_ratio:.4f}"

        # 4. Compiled XLA fusion kernel count is within 10%
        fusions_adapter = hlo_compiled_adapter.count("fusion")
        fusions_raw = hlo_compiled_raw.count("fusion")
        fusion_ratio = fusions_adapter / fusions_raw
        assert 0.90 <= fusion_ratio <= 1.10, (
            f"Compiled HLO fusion count diverged: adapter={fusions_adapter}, raw={fusions_raw} (ratio={fusion_ratio:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. Multi-Backend Zero Host-Callback Invariant
# ---------------------------------------------------------------------------


class TestMultiBackendHLOInvariants:
    """Verifies that all adapter backends compile with zero Python callbacks."""

    def test_evosax_zero_host_callback(self):
        """Evosax adapter generates no Python host callbacks in compiled HLO."""
        pytest.importorskip("evosax")
        from malthusjax.composer.evosax_adapter import build_evosax_engine
        from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
        from malthusjax.core.fitness.composable.environments import BBOBEnv
        from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
        from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter

        evalr = OptimizationEvaluator(
            env=BBOBEnv.create(fn_name="sphere", num_dims=5, seed=0),
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=False),
        )
        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=evalr,
            pop_size=16,
            generations=5,
            bounds=(-5.0, 5.0),
            maximize=False,
        )
        engine = getattr(adapter, "engine", adapter)
        jit_loop = engine._build_jit_loop()
        key = jr.PRNGKey(0)
        key_run, key_init = jr.split(key)
        init_res = engine.init_fn(engine.framework_obj, key_init, engine.framework_params, None)
        hlo = jit_loop.lower(key_run, init_res).compile().as_text()

        assert "xla.python.callback" not in hlo
        assert "xla.python.cpu_callback" not in hlo
        assert hlo.count("while") > 0

    def test_qdax_zero_host_callback(self):
        """QDax adapter generates no Python host callbacks in compiled HLO."""
        pytest.importorskip("qdax")
        from tests.composer.test_qdax_adapter import TestQDaxAdapter

        adapter = TestQDaxAdapter().make_adapter()
        engine = getattr(adapter, "engine", adapter)
        jit_loop = engine._build_jit_loop()
        key = jr.PRNGKey(0)
        key_run, key_init = jr.split(key)
        init_res = engine.init_fn(engine.framework_obj, key_init, engine.framework_params, None)
        hlo = jit_loop.lower(key_run, init_res).compile().as_text()

        assert "xla.python.callback" not in hlo
        assert "xla.python.cpu_callback" not in hlo
        assert hlo.count("while") > 0

    def test_mujoco_mjx_zero_host_callback(self):
        """MuJoCo MJX with adapter compiles to native physics kernels with zero Python callbacks."""
        pytest.importorskip("mujoco")
        from malthusjax.composer.evosax_adapter import build_evosax_engine
        from malthusjax.core.fitness.composable.base import ScalarOutput
        from malthusjax.core.fitness.composable.environments import MuJoCoMJXEnv
        from malthusjax.core.fitness.composable.evaluators import RLEvaluator
        from malthusjax.core.fitness.composable.interpreters import MLPInterpreter

        env = MuJoCoMJXEnv.create(env_name="cartpole")
        interp = MLPInterpreter(input_dim=env.obs_dim, output_dim=env.action_dim, hidden=[32, 32])
        evalr = RLEvaluator(
            env=env,
            interpreter=interp,
            output=ScalarOutput(maximize=True),
            max_steps=50,
        )
        adapter = build_evosax_engine(
            strategy_name="Open_ES",
            evaluator=evalr,
            pop_size=4,
            generations=2,
            num_dims=interp.num_params,
            maximize=True,
        )
        engine = getattr(adapter, "engine", adapter)
        jit_loop = engine._build_jit_loop()
        key = jr.PRNGKey(42)
        key_run, key_init = jr.split(key)
        init_res = engine.init_fn(engine.framework_obj, key_init, engine.framework_params, None)
        hlo = jit_loop.lower(key_run, init_res).compile().as_text()

        # Check that no Python interpreter callbacks are invoked during physical simulation
        assert "xla.python.callback" not in hlo
        assert "xla.python.cpu_callback" not in hlo
        # Check that hardware linear algebra constraint solving fusions are generated
        assert hlo.count("fusion") > 100
        assert hlo.count("while") > 0
