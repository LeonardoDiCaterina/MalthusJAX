import chex
import jax
import jax.numpy as jnp
from flax import struct
import functools
import pytest

pytest.importorskip("qdax")

from qdax.core.map_elites import MAPElites
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

from malthusjax.core.fitness.base import BaseQDEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.composer.adapters import EvalMode
from malthusjax.composer.qdax_adapter import build_qdax_engine
from .base_adapter_suite import BaseAdapterTestSuite

@struct.dataclass
class MockConfig:
    genome_config: RealGenomeConfig = struct.field(pytree_node=False)

class DummyQDEval(BaseQDEvaluator):
    def __init__(self, config=None):
        super().__init__(config=config, data=None)
    
    def evaluate(self, genome):
        # We don't use this directly in QDAX, but it's part of the base API
        return -jnp.sum(jnp.square(genome.values))

    def evaluate_qd(self, genome):
        # Return fitness and 2D descriptor (e.g. first two values clamped to [0,1])
        fitness = -jnp.sum(jnp.square(genome.values))
        desc = jnp.clip(genome.values[:2] + 0.5, 0.0, 1.0)
        return fitness, desc

class TestQDaxAdapter(BaseAdapterTestSuite):
    def make_adapter(self, maximize: bool = False, eval_mode: str = EvalMode.NATIVE, seed: int = 0):
        # QDax MAPElites setup
        dim = 3
        pop_size = 10
        num_generations = 5
        
        init_variables = jnp.ones((pop_size, dim)) * 5.0
        
        # MAPElites needs centroids
        centroids = compute_cvt_centroids(
            num_descriptors=2, 
            num_init_cvt_samples=1000, 
            num_centroids=50,
            minval=0.0, 
            maxval=1.0, 
            key=jax.random.PRNGKey(0)
        )
        
        # Mutation operator
        def dummy_mutation(x, random_key):
            return x + jax.random.normal(random_key, x.shape) * 0.1
        def dummy_crossover(x1, x2, random_key):
            return x1
            
        emitter = MixingEmitter(
            mutation_fn=dummy_mutation,
            variation_fn=dummy_crossover,
            variation_percentage=0.5,
            batch_size=pop_size
        )
        
        # Metrics function
        metrics_fn = functools.partial(
            default_qd_metrics, 
            qd_offset=0.0
        )
        
        if eval_mode == EvalMode.MALTHUSJAX:
            evalr = DummyQDEval(config=MockConfig(genome_config=RealGenomeConfig(shape=(dim,))))
        else:
            # For native mode, we create a dummy object that has a `scoring_function` attached
            class NativeEvaluator:
                def scoring_function(self, genotypes, random_key):
                    fitnesses = -jnp.sum(jnp.square(genotypes), axis=-1)
                    descriptors = jnp.clip(genotypes[:, :2] + 0.5, 0.0, 1.0)
                    return fitnesses, descriptors, {}
            evalr = NativeEvaluator()
            
        return build_qdax_engine(
            strategy_cls=MAPElites,
            emitter=emitter,
            metrics_function=metrics_fn,
            evaluator=evalr,
            init_variables=init_variables,
            centroids=centroids,
            pop_size=pop_size,
            generations=num_generations,
            maximize=maximize,
            eval_mode=eval_mode,
            seed=seed,
            history_metrics=["best_fitness", "qd_score", "coverage"]
        )
