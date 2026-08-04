"""TensorNEAT Quality-Diversity Evaluator Integration."""

from typing import Any, Tuple, Optional
import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.qd.population import QDPopulation
from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome

try:
    from tensorneat.common import State
except ImportError:
    State = Any

@struct.dataclass
class TensorNeatEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for TensorNEAT evaluator.
    
    Attributes:
        seed: The PRNG seed used to initialize the internal TensorNEAT state.
            This is required because TensorNEAT `transform` phases often rely
            on a State containing a randkey.
        maximize: Optimization direction. TensorNEAT natively assumes True.
    """
    seed: int = struct.field(pytree_node=False, default=42)
    maximize: bool = struct.field(pytree_node=False, default=True)


@struct.dataclass
class TensorNeatQDEvaluator(BaseQDEvaluator[TensorNeatGenome, TensorNeatEvaluatorConfig, Any]):
    """Bridging Evaluator for Native MalthusJAX loops and TensorNEAT Problems.
    
    This evaluator intercepts the standard MalthusJAX `evaluate_population` flow
    to insert TensorNEAT's mandatory `transform` phase. It takes raw 
    `(nodes, conns)` matrices from a `TensorNeatPopulation`, transforms them into 
    executable neural network parameters, and scores them using a native 
    TensorNEAT `Problem` (e.g. `BraxEnv`).
    """
    algorithm: Any = struct.field(pytree_node=False)
    problem: Any = struct.field(pytree_node=False)
    forward_fn: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, algorithm: Any, problem: Any, forward_fn: Any, config: Optional[TensorNeatEvaluatorConfig] = None) -> 'TensorNeatQDEvaluator':
        """Constructs the evaluator from TensorNEAT components."""
        if config is None:
            config = TensorNeatEvaluatorConfig()
        return cls(
            config=config,
            data=None,
            algorithm=algorithm,
            problem=problem,
            forward_fn=forward_fn
        )

    def evaluate_qd(self, genome: TensorNeatGenome) -> Tuple[chex.Numeric, chex.Array]:
        """Single genome evaluation fallback.
        
        Note: For TensorNEAT, it is highly recommended to rely on `evaluate_population` 
        because TensorNEAT natively expects batch transformations.
        """
        raise NotImplementedError("TensorNeatQDEvaluator relies on vectorized batch evaluation via evaluate_population.")

    def evaluate_population(self, population: BasePopulation[TensorNeatGenome]) -> QDPopulation[TensorNeatGenome]:
        """Intercepts the standard MalthusJAX evaluation flow to insert the transform phase."""
        
        # 1. Initialize a localized TensorNEAT State to manage PRNG for this generation
        key = jax.random.PRNGKey(self.config.seed)
        tn_state = State(randkey=key, generation=jnp.float32(0))
        
        # 2. Extract raw topology matrices
        pop_values = getattr(population.genes, "values", population.genes)
        pop_size = pop_values[0].shape[0]
        keys = jax.random.split(key, pop_size)
        
        # 3. TRANSFORM: Convert (nodes, conns) matrices into neural network parameters
        # algorithm.transform expects a single genome, so we vmap it over the population.
        nodes, conns = pop_values
        transformed_pop = jax.vmap(lambda s, n, c: self.algorithm.transform(s, (n, c)), in_axes=(None, 0, 0))(tn_state, nodes, conns)
        
        # 4. EVALUATE: Use jax.vmap across the transformed network parameters
        # TensorNEAT evaluate signature is typically (state, randkey, act_func, params)
        raw_results = jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
            tn_state, keys, self.forward_fn, transformed_pop
        )
        
        # TensorNEAT Problem returns either `fitness` or `(fitness, desc)` depending on the environment.
        # We dynamically unpack based on the return signature.
        if isinstance(raw_results, tuple) and len(raw_results) == 2:
            fitnesses, descriptors = raw_results
        else:
            # If the environment is not a QD environment, it won't return descriptors.
            # We return empty descriptors and let the user handle it or crash if the QD algorithm requires them.
            fitnesses = raw_results
            descriptors = jnp.zeros((pop_size, 0))
            
        # TensorNEAT replaces NaN with -inf natively, we ensure it here
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)
        
        # MalthusJAX defaults to minimization; if config says maximize=False, we flip it.
        # TensorNEAT natively maximizes, so if config.maximize is True, we keep it as is.
        if not self.config.maximize:
            fitnesses = -1.0 * fitnesses
            
        # 5. RE-PACKAGE: Embed results back into the MalthusJAX PyTree structure
        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors
        
        return QDPopulation(
            genes=population.genes,
            fitness=fitnesses,
            config=population.config,
            info=new_info
        )
