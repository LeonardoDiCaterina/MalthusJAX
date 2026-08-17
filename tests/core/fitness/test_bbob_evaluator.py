import jax
import jax.numpy as jnp
import numpy as np  # Used for testing utilities
import pytest

pytest.importorskip("evosax")
from evosax.problems import BBOBProblem

from malthusjax.core.fitness.bbob_evaluator import BBOB_NAME_ALIASES, BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenome

# --- 1. Parameter Sensitivity Tests ---


@pytest.mark.parametrize("seed", [1, 42, 2026])
def test_bbob_seed_determinism(rng_key, seed):
    """Verifies BBOB landscape is deterministic for a given seed."""
    # Ensure dimensions match (default fixture is 5)
    conf1 = BBOBConfig(seed=seed, num_dims=5, maximize=False)
    eval1 = BBOBEvaluator.create(conf1)

    conf2 = BBOBConfig(seed=seed, num_dims=5, maximize=False)
    eval2 = BBOBEvaluator.create(conf2)

    genome = RealGenome(values=jnp.zeros(5))

    assert float(eval1.evaluate(genome)) == float(eval2.evaluate(genome))


# --- 2. Numerical Parity Proofs ---


@pytest.mark.parametrize("fn_name", ["sphere", "rosenbrock", "rastrigin", "schwefel"])
def test_bbob_parity_across_functions(real_population, rng_key, fn_name):
    """PROOFS: MalthusJAX wrapper is identical to direct evosax calls."""
    num_dims = real_population.genes.values.shape[1]  # This is 5
    seed = 42

    # Normalize function name (same as BBOBEvaluator does)
    fn_name_lower = fn_name.lower()
    fn_name_normalized = BBOB_NAME_ALIASES.get(fn_name_lower, fn_name)

    # Direct evosax setup with BBOBProblem (evosax 0.2.0 API)
    evosax_fitness = BBOBProblem(fn_name=fn_name_normalized, num_dims=num_dims, seed=seed)
    p_state = evosax_fitness.init(rng_key)

    # MalthusJAX setup - Explicitly pass num_dims
    config = BBOBConfig(fn_name=fn_name, num_dims=num_dims, seed=seed, maximize=False)
    evaluator = BBOBEvaluator.create(config)

    X = real_population.genes.values
    rng = jax.random.PRNGKey(0)  # Use same RNG as BBOBEvaluator.evaluate_population

    # Direct evosax evaluation using eval method
    expected_fitness, _, _ = evosax_fitness.eval(rng, X, p_state)

    # MalthusJAX evaluation
    pop_result = evaluator.evaluate_population(real_population)

    # Use np.testing for JAX arrays
    # For minimize=False, evaluator returns raw minimization fitness (no negation)
    np.testing.assert_allclose(pop_result.fitness, expected_fitness, atol=1e-5)


# --- 3. Maximization & JIT Stability ---


def test_bbob_maximize_parameter_logic(real_population):
    """Strictly tests that 'maximize' correctly inverts the output."""
    num_dims = real_population.genes.values.shape[1]

    config_min = BBOBConfig(num_dims=num_dims, maximize=False)
    config_max = BBOBConfig(num_dims=num_dims, maximize=True)

    eval_min = BBOBEvaluator.create(config_min)
    eval_max = BBOBEvaluator.create(config_max)

    res_min = eval_min.evaluate_population(real_population)
    res_max = eval_max.evaluate_population(real_population)

    np.testing.assert_allclose(res_min.fitness, -res_max.fitness, atol=1e-6)


def test_bbob_jit_stability_and_repeatability(real_population):
    """Ensures XLA compilation is stable and results are deterministic."""
    num_dims = real_population.genes.values.shape[1]
    config = BBOBConfig(fn_name="ellipsoidal_rotated", num_dims=num_dims, maximize=False)
    evaluator = BBOBEvaluator.create(config)

    @jax.jit
    def run_eval(p):
        return evaluator.evaluate_population(p).fitness

    res1 = run_eval(real_population)
    res2 = run_eval(real_population)

    np.testing.assert_array_equal(res1, res2)
