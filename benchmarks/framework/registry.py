from typing import Callable, Dict, Any, NamedTuple
import jax
from .adapters import MalthusAdapter, EvosaxAdapter

# --- Malthus Components ---
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.core.genome.real_genome import RealGenomeConfig

# --- Evosax Components ---    
from evosax.algorithms.population_based.simple_ga import SimpleGA

class ComparisonSpec(NamedTuple):
    name: str
    malthus_factory: Callable
    evosax_factory: Callable
    default_hypers: Dict[str, Any]

class ComparisonRegistry:
    _registry: Dict[str, ComparisonSpec] = {}

    @classmethod
    def register(cls, spec: ComparisonSpec):
        cls._registry[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ComparisonSpec:
        if name not in cls._registry:
            raise ValueError(f"Unknown algo: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

# =========================================================
# BUILDERS
# =========================================================

def _build_malthus_ga(pop_size, dims, seed, hypers, problem_evaluator):
    """Builds MalthusJAX engine matching the Standard GA spec."""
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    
    mutation = GaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 0.1)
    )
    
    crossover = UniformCrossover(
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = max(1, int(pop_size * elite_ratio))
    
    selection = ElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count
    )
    
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=elite_count
    )
    
    engine = GeneticEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )
    
    return MalthusAdapter(engine)

def _build_evosax_ga(pop_size, dims, seed, hypers, problem_object):
    """
    Builds Evosax strategy following 'evosax_benchmark_old.ipynb' EXACTLY.
    """

    # 1. Sample init solution
    rng = jax.random.PRNGKey(seed)
    # Returns a single sample (D,)
    init_solution = problem_object.sample(rng)
    
    # 2. Instantiate Strategy
    strategy = SimpleGA(
        population_size=pop_size,
        solution=init_solution
    )
    
    # 3. Configure Elitism
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    
    # 4. Configure Params
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
    # Pass pop_size explicitly to the adapter (Retaining the fix from previous step)
    return EvosaxAdapter(strategy, es_params, problem_object, pop_size)

# Register "Standard_GA"
ComparisonRegistry.register(ComparisonSpec(
    name="Standard_GA",
    malthus_factory=_build_malthus_ga,
    evosax_factory=_build_evosax_ga,
    default_hypers={
        'mutation_rate': 0.05,
        'crossover_rate': 0.6,
        'sigma': 0.1,
        'elite_ratio': 0.1
    }
))