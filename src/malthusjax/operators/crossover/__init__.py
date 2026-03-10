"""
Crossover operators for MalthusJAX.

Crossover operators generate offspring from parent pairs using the new batch-first paradigm.
All crossover operators inherit from BaseCrossover and return (num_offspring, genome_shape).
"""

from .binary import SinglePointCrossover
from .binary import UniformCrossover as BinaryUniformCrossover
from .evosax_crossover import EvosaxUniformCrossoverWrapper
from .real import (
    BinomialCrossover,
    BinomialCrossover_injection,
    BlendCrossover,
    BlendCrossover_injection,
    SimulatedBinaryCrossover,
    SimulatedBinaryCrossover_injection,
)
from .real import (
    UniformCrossover as RealUniformCrossover,
)
from .real import (
    UniformCrossover_injection as RealUniformCrossover_injection,
)

UniformCrossover = BinaryUniformCrossover

__all__ = [
    "UniformCrossover",
    "BinaryUniformCrossover",
    "RealUniformCrossover",
    "RealUniformCrossover_injection",
    "SinglePointCrossover",
    "BlendCrossover",
    "BlendCrossover_injection",
    "BinomialCrossover",
    "BinomialCrossover_injection",
    "SimulatedBinaryCrossover",
    "SimulatedBinaryCrossover_injection",
    "EvosaxUniformCrossoverWrapper",
]


def _register_crossover() -> None:
    """Register crossover operators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            ("uniform_real", RealUniformCrossover, {}),
            ("uniform_real_injection", RealUniformCrossover_injection, {}),
            ("blend", BlendCrossover, {}),
            ("blend_injection", BlendCrossover_injection, {}),
            ("simulated_binary", SimulatedBinaryCrossover, {}),
            ("simulated_binary_injection", SimulatedBinaryCrossover_injection, {}),
            ("binomial", BinomialCrossover, {}),
            ("binomial_injection", BinomialCrossover_injection, {}),
            ("evosax_uniform_crossover", EvosaxUniformCrossoverWrapper, {}),
            ("uniform_binary", BinaryUniformCrossover, {}),
            ("single_point", SinglePointCrossover, {}),
        ],
        override=True,
    )


_register_crossover()
