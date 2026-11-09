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

_OPTIONAL_EXPORTS = {
    "draw_network": ("malthus_neat.interpreters.visualizer", "draw_network"),
    "extract_network_graph": ("malthus_neat.interpreters.visualizer", "extract_network_graph"),
    "plot_network": ("malthus_neat.interpreters.visualizer", "plot_network"),
    "to_dot": ("malthus_neat.interpreters.visualizer", "to_dot"),
    "to_networkx": ("malthus_neat.interpreters.visualizer", "to_networkx"),
    "BaseNEATEstimator": ("malthus_neat.learn.base", "BaseNEATEstimator"),
    "NEATClassifier": ("malthus_neat.learn.neat_classifier", "NEATClassifier"),
    "NEATRegressor": ("malthus_neat.learn.neat_regressor", "NEATRegressor"),
}


def __getattr__(name):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module, attribute = _OPTIONAL_EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
