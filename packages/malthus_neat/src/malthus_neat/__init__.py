"""Malthus-NEAT: JAX-Native Neuroevolution of Augmenting Topologies.

Tensorized NEAT algorithms leveraging TensorNEAT kernels, Quality-Diversity MAP-Elites,
multi-island topologies, and Scikit-Learn compatible estimators.
"""

__version__ = "0.1.0"

from malthus_neat.core.state import State
from malthus_neat.engine.neat_engine import build_neat_engine
from malthus_neat.evaluators.neat_evaluator import NeatEvaluator, NeatEvaluatorConfig
from malthus_neat.evaluators.neat_qd_evaluator import NeatQDEvalConfig, NeatQDEvaluator
from malthus_neat.genomes.neat_genome import (
    NeatGenome,
    NeatGenomeConfig,
    NeatPopulation,
)
from malthus_neat.interpreters.neat_interpreter import NeatInterpreter
from malthus_neat.interpreters.visualizer import (
    draw_network,
    extract_network_graph,
    plot_network,
    to_dot,
    to_networkx,
)
from malthus_neat.learn.base import BaseNEATEstimator
from malthus_neat.learn.neat_classifier import NEATClassifier
from malthus_neat.learn.neat_regressor import NEATRegressor
from malthus_neat.operators.neat_crossover import NeatCrossover
from malthus_neat.operators.neat_emitter import NeatEmitter
from malthus_neat.operators.neat_mutation import NeatMutation

__all__ = [
    "State",
    "build_neat_engine",
    "NeatGenome",
    "NeatGenomeConfig",
    "NeatPopulation",
    "NeatInterpreter",
    "draw_network",
    "plot_network",
    "to_dot",
    "to_networkx",
    "extract_network_graph",
    "NeatMutation",
    "NeatCrossover",
    "NeatEmitter",
    "NeatEvaluator",
    "NeatEvaluatorConfig",
    "NeatQDEvaluator",
    "NeatQDEvalConfig",
    "BaseNEATEstimator",
    "NEATRegressor",
    "NEATClassifier",
]
