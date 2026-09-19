"""Tests for the composable evaluator architecture.

Covers:
- MLPInterpreter: num_params calculation, unflattening, forward pass.
- IdentityInterpreter: passthrough.
- SklearnEnv: data loading.
- SupervisedEvaluator: single genome, population evaluation.
- OptimizationEvaluator: BBOBEnv integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.composable import (
    BBOBEnv,
    CustomDatasetEnv,
    IdentityInterpreter,
    MLPInterpreter,
    OptimizationEvaluator,
    ScalarOutput,
    SklearnEnv,
    SupervisedEvaluator,
)
from malthusjax.core.genome.real_genome import RealGenome, RealPopulation

# =============================================================================
# Helpers
# =============================================================================


def make_real_genome(values: list[float]) -> RealGenome:
    return RealGenome(values=jnp.array(values, dtype=jnp.float32))


def make_real_population(n: int, length: int, seed: int = 0) -> RealPopulation:
    from malthusjax.core.genome.real_genome import RealGenomeConfig

    key = jax.random.PRNGKey(seed)
    genes_values = jax.random.normal(key, (n, length), dtype=jnp.float32)
    genes = RealGenome(values=genes_values)
    fitness = jnp.zeros(n, dtype=jnp.float32)
    config = RealGenomeConfig(shape=(length,))
    return RealPopulation(genes=genes, fitness=fitness, config=config)


# =============================================================================
# MLPInterpreter Tests
# =============================================================================


class TestMLPInterpreter:
    def test_num_params_no_hidden(self):
        """Single linear layer: input_dim*output_dim + output_dim biases."""
        interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=())
        # W: 4*2=8, b: 2 → total 10
        assert interp.num_params == 10

    def test_num_params_with_hidden(self):
        """Standard MLP with hidden layers."""
        interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=(64, 64))
        # Layer 0: 4*64 + 64 = 320
        # Layer 1: 64*64 + 64 = 4160
        # Layer 2: 64*2 + 2 = 130
        assert interp.num_params == 320 + 4160 + 130

    def test_layer_sizes(self):
        interp = MLPInterpreter(input_dim=3, output_dim=1, hidden=(8, 4))
        assert interp.layer_sizes == (3, 8, 4, 1)

    def test_apply_output_shape(self):
        """Output shape should be (output_dim,)."""
        interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=(8,))
        genome = make_real_genome([0.1] * interp.num_params)
        inputs = jnp.ones(4, dtype=jnp.float32)
        result = interp.apply(genome, inputs)
        assert result.shape == (2,)

    def test_apply_deterministic(self):
        """Same genome + inputs should always give same output."""
        interp = MLPInterpreter(input_dim=4, output_dim=1, hidden=(8,))
        genome = make_real_genome([0.5] * interp.num_params)
        inputs = jnp.ones(4)
        r1 = interp.apply(genome, inputs)
        r2 = interp.apply(genome, inputs)
        assert jnp.allclose(r1, r2)

    def test_apply_different_inputs(self):
        """Different inputs produce different outputs."""
        interp = MLPInterpreter(input_dim=2, output_dim=1, hidden=())
        genome = make_real_genome([1.0, 0.0, 0.5])  # W=[[1,0]], b=[0.5]
        r1 = interp.apply(genome, jnp.array([1.0, 0.0]))
        r2 = interp.apply(genome, jnp.array([0.0, 1.0]))
        assert not jnp.allclose(r1, r2)

    def test_activation_relu(self):
        """ReLU activation should zero out negative pre-activations."""
        interp = MLPInterpreter(input_dim=1, output_dim=1, hidden=(2,), activation="relu")
        # Craft weights so first hidden layer gets negative activation
        # W0 = [[-1], [-1]], b0 = [0, 0], W1 = [[1], [1]], b1 = [0]
        # For input [1.0]: hidden = relu([-1, -1]) = [0, 0], output = 0
        n = interp.num_params
        # W0 shape: (1,2), b0 shape: (2,), W1 shape: (2,1), b1 shape: (1,)
        # Order is [b0, W0, b1, W1]
        assert n == 2 + 1 * 2 + 1 + 2 * 1  # = 7
        params = jnp.array([0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0])
        genome = RealGenome(values=params)
        result = interp.apply(genome, jnp.array([1.0]))
        assert float(result[0]) == pytest.approx(0.0)


# =============================================================================
# IdentityInterpreter Tests
# =============================================================================


class TestIdentityInterpreter:
    def test_returns_genome_values(self):
        interp = IdentityInterpreter()
        values = jnp.array([1.0, 2.0, 3.0])
        genome = RealGenome(values=values)
        result = interp.apply(genome)
        assert jnp.allclose(result, values)

    def test_num_params_is_minus_one(self):
        """IdentityInterpreter.num_params == -1 (problem-determined)."""
        assert IdentityInterpreter().num_params == -1

    def test_inputs_ignored(self):
        """Providing inputs does not affect the output."""
        interp = IdentityInterpreter()
        genome = RealGenome(values=jnp.array([5.0, 6.0]))
        r1 = interp.apply(genome, inputs=None)
        r2 = interp.apply(genome, inputs=jnp.array([99.0, 99.0]))
        assert jnp.allclose(r1, r2)


# =============================================================================
# SklearnEnv Tests
# =============================================================================


class TestSklearnEnv:
    def test_create_breast_cancer(self):
        env = SklearnEnv.create(dataset="breast_cancer")
        assert env.X.shape == (569, 30)
        assert env.y.shape == (569,)

    def test_create_make_regression(self):
        env = SklearnEnv.create(dataset="make_regression", n_samples=100, n_features=5)
        assert env.X.shape == (100, 5)
        assert env.y.shape == (100,)

    def test_dtype_is_float32(self):
        env = SklearnEnv.create(dataset="breast_cancer")
        assert env.X.dtype == jnp.float32
        assert env.y.dtype == jnp.float32

    def test_custom_dataset_env(self):
        X = jnp.ones((50, 3))
        y = jnp.zeros(50)
        env = CustomDatasetEnv(data=(X, y))
        assert jnp.allclose(env.X, X)
        assert jnp.allclose(env.y, y)


# =============================================================================
# SupervisedEvaluator Tests
# =============================================================================


class TestSupervisedEvaluator:
    def setup_method(self):
        self.env = SklearnEnv.create(dataset="make_regression", n_samples=50, n_features=4)
        self.interp = MLPInterpreter(input_dim=4, output_dim=1, hidden=(8,))
        self.output = ScalarOutput(loss_fn="mse")
        self.evaluator = SupervisedEvaluator(
            env=self.env, interpreter=self.interp, output=self.output
        )

    def test_evaluate_single_genome(self):
        genome = make_real_genome([0.1] * self.interp.num_params)
        fitness, info = self.evaluator.evaluate(genome, genome)
        assert fitness.shape == ()  # scalar
        assert jnp.isfinite(fitness)

    def test_evaluate_population(self):
        pop = make_real_population(n=8, length=self.interp.num_params)
        updated = self.evaluator.evaluate_population(pop)
        assert updated.fitness.shape == (8,)
        assert jnp.all(jnp.isfinite(updated.fitness))

    def test_zero_genome_gives_finite_fitness(self):
        genome = make_real_genome([0.0] * self.interp.num_params)
        fitness, info = self.evaluator.evaluate(genome, genome)
        assert jnp.isfinite(fitness)

    def test_bce_loss_mode(self):
        env = SklearnEnv.create(dataset="breast_cancer")
        interp = MLPInterpreter(input_dim=30, output_dim=1, hidden=(16,))
        output = ScalarOutput(loss_fn="bce")
        evaluator = SupervisedEvaluator(env=env, interpreter=interp, output=output)
        genome = make_real_genome([0.0] * interp.num_params)
        fitness, info = evaluator.evaluate(genome, genome)
        assert jnp.isfinite(fitness)

    def test_maximize_flag_negates_loss(self):
        """maximize=True should return negated loss (higher is better)."""
        genome = make_real_genome([0.1] * self.interp.num_params)
        out_min = ScalarOutput(loss_fn="mse", maximize=False)
        out_max = ScalarOutput(loss_fn="mse", maximize=True)
        eval_min = SupervisedEvaluator(env=self.env, interpreter=self.interp, output=out_min)
        eval_max = SupervisedEvaluator(env=self.env, interpreter=self.interp, output=out_max)
        f_min, info_min = eval_min.evaluate(genome, genome)
        f_max, info_max = eval_max.evaluate(genome, genome)
        assert jnp.allclose(f_min, -f_max, atol=1e-5)


# =============================================================================
# OptimizationEvaluator Tests
# =============================================================================


class TestOptimizationEvaluator:
    def setup_method(self):
        self.env = BBOBEnv.create(fn_name="sphere", num_dims=5)
        self.interp = IdentityInterpreter()
        self.output = ScalarOutput()
        self.evaluator = OptimizationEvaluator(
            env=self.env, interpreter=self.interp, output=self.output
        )

    def test_evaluate_single_genome(self):
        genome = make_real_genome([0.1] * 5)
        fitness, info = self.evaluator.evaluate(genome, genome)
        assert fitness.shape == ()
        assert jnp.isfinite(fitness)

    def test_optimal_genome_gives_best_fitness(self):
        """The known optimum x_opt should yield fitness near f_opt."""
        x_opt = self.env.x_opt
        genome = RealGenome(values=x_opt)
        fitness, info = self.evaluator.evaluate(genome, genome)
        f_opt = self.env.f_opt
        # Should be very close to f_opt (minimization by default)
        assert jnp.abs(fitness - f_opt) < 1.0

    def test_evaluate_population(self):
        pop = make_real_population(n=10, length=5)
        updated = self.evaluator.evaluate_population(pop)
        assert updated.fitness.shape == (10,)
        assert jnp.all(jnp.isfinite(updated.fitness))
