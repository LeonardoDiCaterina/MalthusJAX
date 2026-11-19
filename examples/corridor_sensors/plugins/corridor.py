"""Two maintenance sensors on a normalized corridor with uneven incident demand."""
import jax.numpy as jnp
from flax import struct

from malthusjax.composer import register_fitness
from malthusjax.core.fitness import IdentityTransform, OptimizationEvaluator, QDOutput, ScalarOutput
from malthusjax.core.fitness.composable.base import BaseDescriptorFn, BaseInterpreter, BaseOptimizationEnvironment

POINTS = tuple(i / 10 for i in range(11))
WEIGHTS = (1., 3., 8., 2., 1., 1., 2., 7., 5., 1., 1.)
MIN_GAP = 0.15

@struct.dataclass
class SensorLayout(BaseInterpreter):
    """Decode two bounded genes to ordered coordinates separated by at least MIN_GAP."""
    @property
    def num_params(self):
        return 2

    def apply(self, genome, inputs=None):
        u, v = jnp.clip(genome.values, 0.0, 1.0)
        left = (1 - MIN_GAP) * u
        right = left + MIN_GAP + (1 - left - MIN_GAP) * v
        return jnp.stack((left, right))

@struct.dataclass
class CorridorDemand(BaseOptimizationEnvironment):
    """Penalize expected response distance plus a worst-case distance term."""
    worst_weight: float = struct.field(pytree_node=False, default=0.30)

    def evaluate(self, layout):
        points = jnp.asarray(POINTS)
        weights = jnp.asarray(WEIGHTS)
        nearest = jnp.min(jnp.abs(points[:, None] - layout[None, :]), axis=1)
        return jnp.average(nearest, weights=weights) + self.worst_weight * jnp.max(nearest)

@struct.dataclass
class LayoutDescriptor(BaseDescriptorFn):
    constant: bool = struct.field(pytree_node=False, default=False)

    def compute(self, genome, inputs, outputs):
        if self.constant:
            return jnp.array([0.5, 0.5])
        left, right = outputs
        # Normalize attainable midpoint and separation to [0, 1].
        return jnp.stack((((left + right) / 2 - MIN_GAP / 2) / (1 - MIN_GAP),
                          (right - left - MIN_GAP) / (1 - MIN_GAP)))

@register_fitness(name="research_corridor")
def corridor(worst_weight=0.30, descriptor="none", maximize=False, dim=2, seed=42):
    if dim != 2:
        raise ValueError("The corridor needs exactly two genes")
    output = ScalarOutput(maximize=maximize)
    if descriptor not in {"none", "layout", "constant"}:
        raise ValueError(f"Unknown corridor descriptor: {descriptor!r}")
    if descriptor != "none":
        output = QDOutput.create(scalar_output=output,
                                 descriptor_fn=LayoutDescriptor(constant=descriptor == "constant"))
    return OptimizationEvaluator(env=CorridorDemand(worst_weight), transform=IdentityTransform(),
                                 interpreter=SensorLayout(), output=output)
