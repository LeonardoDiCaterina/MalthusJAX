"""Malthus-GP: JAX-Native Genetic Programming.

Modular Genetic Programming algorithms with Cartesian GP, Differentiable Cartesian GP (dCGP),
neutral drift, AutoDiff residual ODE solvers, and Scikit-Learn compatible estimators.
"""

__version__ = "0.1.0"

from malthus_gp.engine.cgp_engine import build_cgp_engine
from malthus_gp.evaluators.cartesian_gp_evaluator import (
    CartesianGPEvaluator,
    CartesianGPEvaluatorConfig,
)
from malthus_gp.evaluators.lsmf_evaluator import (
    LSMFEvaluator,
    LSMFEvaluatorConfig,
)
from malthus_gp.evaluators.ode_evaluator import (
    ODEResidualEvaluator,
    ODEResidualEvaluatorConfig,
)
from malthus_gp.genomes.cartesian_genome import (
    CartesianGenome,
    CartesianGenomeConfig,
    CartesianPopulation,
)
from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
    DifferentiableCartesianGenomeConfig,
    DifferentiableCartesianPopulation,
)
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)
from malthus_gp.learn.base import BaseGPEstimator
from malthus_gp.learn.cgp_classifier import CGPClassifier
from malthus_gp.learn.cgp_ode_solver import CGPODESolver
from malthus_gp.learn.cgp_regressor import CGPRegressor
from malthus_gp.learn.dcgp_regressor import DCGPRegressor
from malthus_gp.learn.symbolic_transformer import SymbolicFeatureTransformer
from malthus_gp.operators.cartesian_micro_mutation import CartesianMicroMutation
from malthus_gp.operators.differentiable_mutation import DifferentiableMicroMutation
from malthus_gp.operators.neutral_selection import NeutralSelection
from malthus_gp.operators.no_op_crossover import NoOpCrossover

__all__ = [
    "CartesianGenome",
    "CartesianGenomeConfig",
    "CartesianPopulation",
    "CartesianMicroMutation",
    "NoOpCrossover",
    "NeutralSelection",
    "CartesianInterpreter",
    "CartesianGPEvaluator",
    "CartesianGPEvaluatorConfig",
    "build_cgp_engine",
    "BaseGPEstimator",
    "CGPRegressor",
    "CGPClassifier",
    "SymbolicFeatureTransformer",
    "DifferentiableCartesianGenome",
    "DifferentiableCartesianGenomeConfig",
    "DifferentiableCartesianPopulation",
    "DifferentiableCartesianInterpreter",
    "DifferentiableMicroMutation",
    "ODEResidualEvaluator",
    "ODEResidualEvaluatorConfig",
    "LSMFEvaluator",
    "LSMFEvaluatorConfig",
    "DCGPRegressor",
    "CGPODESolver",
]
