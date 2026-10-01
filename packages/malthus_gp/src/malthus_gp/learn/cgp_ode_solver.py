"""Symbolic Ordinary Differential Equation (ODE) Solver using dCGP.

Finds analytical closed-form solutions to ODEs by evaluating the residual of
the differential equation via exact JAX automatic differentiation (Izzo et al., 2016).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from malthus_gp.core.functions import ACTIVATION_FUNCTIONS, ACTIVATION_NAMES
from malthus_gp.evaluators.lsmf_evaluator import LSMFEvaluator, LSMFEvaluatorConfig
from malthus_gp.evaluators.ode_evaluator import (
    ODEResidualEvaluator,
    ODEResidualEvaluatorConfig,
)
from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
    DifferentiableCartesianGenomeConfig,
)
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)
from malthus_gp.learn.base import BaseGPEstimator
from malthus_gp.operators.differentiable_mutation import DifferentiableMicroMutation
from malthus_gp.operators.neutral_selection import NeutralSelection
from malthus_gp.operators.no_op_crossover import NoOpCrossover
from malthusjax.core.logger import get_logger
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams

logger = get_logger("malthus_gp.learn.cgp_ode_solver")


class CGPODESolver(BaseGPEstimator):
    """Analytical Ordinary Differential Equation (ODE) Solver using dCGP.

    Searches for a symbolic function y(t) satisfying:
        dy/dt = ode_fn(t, y),  with y(t0) = y0

    Parameters
    ----------
    ode_fn : callable
        Right-hand side function f(t, y) of the ODE dy/dt = f(t, y).
    initial_condition : tuple of (float, float), default=(0.0, 1.0)
        Boundary value condition (t0, y0).
    t_domain : array-like, default=linspace(0, 2, 50)
        Grid of time points in the domain to enforce the differential equation.
    n_cols : int, default=20
        Number of computational nodes in the Cartesian grid.
    pop_size : int, default=5
        Population size (1 + λ).
    max_generations : int, default=300
        Maximum evolutionary generations.
    lsmf_epochs : int, default=10
        Inner Adam gradient steps on weights per evaluation.
    learning_rate : float, default=0.02
        Inner optimizer learning rate.
    bc_weight : float, default=10.0
        Penalty factor for initial condition violations.
    random_state : int, default=42
        PRNG seed.
    """

    def __init__(
        self,
        ode_fn: Callable[[chex.Array, chex.Array], chex.Array] = lambda t, y: -y,
        initial_condition: Tuple[float, float] = (0.0, 1.0),
        t_domain: Optional[Any] = None,
        n_cols: int = 20,
        pop_size: int = 5,
        max_generations: int = 300,
        lsmf_epochs: int = 10,
        learning_rate: float = 0.02,
        bc_weight: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.ode_fn = ode_fn
        self.initial_condition = initial_condition
        self.t_domain = t_domain if t_domain is not None else np.linspace(0.0, 2.0, 50)
        self.n_cols = n_cols
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.lsmf_epochs = lsmf_epochs
        self.learning_rate = learning_rate
        self.bc_weight = bc_weight
        self.random_state = random_state

    def solve(self) -> "CGPODESolver":
        """Executes the dCGP evolutionary search to solve the ODE."""
        t_arr = np.asarray(self.t_domain, dtype=np.float32)

        self.config_ = DifferentiableCartesianGenomeConfig(
            num_rows=1,
            num_cols=self.n_cols,
            num_inputs=1,  # time t
            num_outputs=1,  # solution y(t)
            num_ops=len(ACTIVATION_FUNCTIONS),
            max_arity=2,
            levels_back=-1,
        )

        self.interpreter_ = DifferentiableCartesianInterpreter(
            num_inputs=1,
            num_outputs=1,
            activations=ACTIVATION_FUNCTIONS,
            activation_names=ACTIVATION_NAMES,
        )

        ode_eval_cfg = ODEResidualEvaluatorConfig(
            bc_weight=self.bc_weight,
            maximize=False,
        )
        base_evaluator = ODEResidualEvaluator(
            config=ode_eval_cfg,
            data=jnp.array(t_arr),
            interpreter=self.interpreter_,
            ode_fn=self.ode_fn,
            initial_condition=self.initial_condition,
        )

        # Wrap with LSMF memetic evaluator
        lsmf_cfg = LSMFEvaluatorConfig(epochs=self.lsmf_epochs, maximize=False)
        evaluator = LSMFEvaluator(
            config=lsmf_cfg,
            data=base_evaluator.data,
            base_evaluator=base_evaluator,
            optimizer=optax.adam(self.learning_rate),
        )

        engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=1,
            num_generations=self.max_generations,
        )
        lambda_size = self.pop_size - 1

        engine = GeneticEngine(
            genome_config=self.config_,
            evaluator=evaluator,
            selection=NeutralSelection(
                num_selections=lambda_size * 2,
                elite_k=1,
                n_elites=1,
                maximize=False,
            ),
            crossover=NoOpCrossover(num_offspring=1),
            mutation=DifferentiableMicroMutation(
                mutation_rate=3.0 / float(self.config_.total_genes),
                weight_noise_std=0.0,
                num_offspring=1,
            ),
            engine_params=engine_params,
        )

        key = jax.random.key(self.random_state)
        init_state = engine.init_state(key)
        final_state, output, _ = engine.run(init_state)

        self.best_genome_: DifferentiableCartesianGenome = final_state.best_genome
        self.residual_loss_: float = float(final_state.best_fitness)
        self.history_ = output

        return self

    def fit(self, X: Any = None, y: Any = None) -> "CGPODESolver":
        """Alias for solve() to conform with Scikit-Learn estimator conventions."""
        return self.solve()

    def predict(self, t: Any) -> np.ndarray:
        """Evaluates the evolved solution y(t) at points t."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Solver has not run yet. Call 'solve' first.")

        t_arr = np.asarray(t, dtype=np.float32)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(-1, 1)

        preds_jax = self.interpreter_.apply(self.best_genome_, jnp.array(t_arr))
        return np.asarray(preds_jax).squeeze()

    def to_formula(self) -> str:
        """Returns the decoded closed-form mathematical expression for y(t)."""
        if not hasattr(self, "best_genome_"):
            raise ValueError("Solver has not run yet. Call 'solve' first.")
        return str(self.interpreter_.to_formula(self.best_genome_, feature_names=["t"]))
