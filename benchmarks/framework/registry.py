from typing import Callable, Dict, Any, NamedTuple, Optional
import jax
import jax.numpy as jnp
from .adapters import MalthusAdapter, EvosaxAdapter

# --- Malthus Components ---
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.MR15_GA import OneFifthGeneticEngine, OneFifthGeneticEngineParams
from malthusjax.engine.differential_engine import DifferentialEvolutionEngine
from malthusjax.operators.mutation.real import GaussianMutation, DifferentialMutation
from malthusjax.operators.crossover.real import UniformCrossover, BinomialCrossover
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.mutation.ablation_mutation import AblationGaussianMutation
from malthusjax.operators.crossover.ablation_crossover import AblationUniformCrossover
from malthusjax.operators.selection.ablation_selection import AblationElitePoolSelection
from malthusjax.core.genome.real_genome import RealGenomeConfig

# --- Evosax Components ---    
from evosax.algorithms.population_based import SimpleGA, MR15_GA, DifferentialEvolution

class ComparisonSpec(NamedTuple):
    name: str
    malthus_factory: Callable
    evosax_factory: Optional[Callable] = None  # Optional for Malthus-only ablations
    default_hypers: Dict[str, Any] = {}

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
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    
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


def _build_evosax_mr15_ga(pop_size, dims, seed, hypers, problem_object):
    """
    Builds Evosax strategy following 'evosax_benchmark_old.ipynb' EXACTLY.
    """

    # 1. Sample init solution
    rng = jax.random.PRNGKey(seed)
    # Returns a single sample (D,)
    init_solution = problem_object.sample(rng)
    
    # 2. Instantiate Strategy
    strategy = MR15_GA(
        population_size=pop_size,
        solution=init_solution
    )
    
    # 3. Configure Elitism
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    
    # 4. Configure Params
    #es_params = strategy.default_params.replace(
    #    crossover_rate=hypers.get('crossover_rate', 0.5)
    #)
    
    # Pass pop_size explicitly to the adapter (Retaining the fix from previous step)
    return EvosaxAdapter(strategy, strategy.default_params, problem_object, pop_size)

def _build_malthus_mr15(pop_size, dims, seed, hypers, problem_evaluator):
    """Builds MalthusJAX OneFifthGeneticEngine matching the MR15_GA spec."""
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    
    mutation = GaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 1.0)  # std_init in evosax
    )
    
    crossover = UniformCrossover(
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    
    selection = ElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count
    )
    
    engine_params = OneFifthGeneticEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=elite_count,
        std_min=hypers.get('std_min', 0.0),
        std_max=hypers.get('std_max', float('inf')),
        std_ratio=hypers.get('std_ratio', 0.2),
    )
    
    engine = OneFifthGeneticEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )
    
    return MalthusAdapter(engine)


def _build_malthus_ga_ablation(pop_size, dims, seed, hypers, problem_evaluator):
    """Builds MalthusJAX engine with ABLATION operators (zero key allocation)."""
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    
    mutation = AblationGaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 0.1),
        seed=seed  # Use provided seed for determinism
    )
    
    crossover = AblationUniformCrossover(
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5),
        seed=seed
    )
    
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    
    selection = AblationElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count,
        seed=seed
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



# 1. Define the Optimized Strategy (The Patch)
class OptimizedSimpleGA(SimpleGA):
    """
    A patched version of SimpleGA that avoids the `searchsorted` bottleneck.
    It replaces weighted random sampling with uniform sampling of the top-k elites.
    """
    def _ask(self, key, state, params):
        # --- OPTIMIZATION START ---
        # 1. Sort population by fitness (Standard GA logic)
        idx = jnp.argsort(state.fitness)
        sorted_pop = state.population[idx]
        
        # 2. Slice the Elites (Direct memory view, no search needed)
        # This replaces: p = jnp.arange(...) < elite_count
        elites = sorted_pop[:self.num_elites]
        
        # 3. Uniformly sample parents from the elite pool
        # This avoids jax.random.choice(p=weights), preventing 'searchsorted'
        rng_cross, rng_mut, rng_p1, rng_p2 = jax.random.split(key, 4)
        
        parents_1 = jax.random.choice(rng_p1, elites, (self.population_size,))
        parents_2 = jax.random.choice(rng_p2, elites, (self.population_size,))
        # --- OPTIMIZATION END ---

        # 4. Standard Crossover & Mutation (Same as original)
        rng_cross_split = jax.random.split(rng_cross, self.population_size)
        rng_mut_split = jax.random.split(rng_mut, self.population_size)

        # We must use self.crossover_strategy if defined, or the raw functions
        # For safety, we use the raw functions as defined in Evosax source
        # (Assuming you have access to crossover/mutation funcs or import them)
        from evosax.algorithms.population_based.simple_ga import crossover, mutation
        
        population = jax.vmap(crossover, in_axes=(0, 0, 0, None))(
            rng_cross_split, parents_1, parents_2, params.crossover_rate
        )

        population = jax.vmap(mutation, in_axes=(0, 0, None))(
            rng_mut_split, population, state.std
        )

        return population, state

# 2. Define the Factory
def _build_evosax_ga_optimized(pop_size, dims, seed, hypers, problem_object):
    """Builds the OPTIMIZED Evosax strategy."""
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    
    # Instantiate the PATCHED class
    strategy = OptimizedSimpleGA(
        population_size=pop_size,
        solution=init_solution
    )
    
    # Configure parameters same as Standard_GA
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
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

# Register "Standard_GA_Ablation" - Zero key allocation overhead (Malthus-only)
ComparisonRegistry.register(ComparisonSpec(
    name="Standard_GA_Ablation",
    malthus_factory=_build_malthus_ga_ablation,
    evosax_factory= _build_evosax_ga_optimized,
    default_hypers={
        'mutation_rate': 0.05,
        'crossover_rate': 0.6,
        'sigma': 0.1,
        'elite_ratio': 0.1
    }
))

# Register "MR15_GA" - 1/5 Success Rule GA
ComparisonRegistry.register(ComparisonSpec(
    name="MR15_GA",
    malthus_factory=_build_malthus_mr15,
    evosax_factory=_build_evosax_mr15_ga,
    default_hypers={
        'mutation_rate': 0.1,
        'crossover_rate': 0.5,
        'sigma': 1.0,       # std_init
        'elite_ratio': 0.5,
        'std_min': 0.0,
        'std_max': float('inf'),
        'std_ratio': 0.2,   # 1/5 threshold
    }
))


# =========================================================
# DIFFERENTIAL EVOLUTION BUILDERS
# =========================================================

def _build_malthus_de(pop_size, dims, seed, hypers, problem_evaluator):
    """Builds MalthusJAX DifferentialEvolutionEngine."""
    from malthusjax.engine.base import AbstractEngineParams
    
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    
    # DE uses differential mutation (rand/1 variant)
    # DifferentialMutation uses 'f_scale' (not 'differential_weight')
    mutation = DifferentialMutation(
        f_scale=hypers.get('differential_weight', 0.8),
    )
    
    # DE uses binomial crossover
    crossover = BinomialCrossover(
        num_offspring=1,
        crossover_rate=hypers.get('crossover_rate', 0.9)
    )
    
    engine_params = AbstractEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=0  # DE doesn't use traditional elitism
    )
    
    engine = DifferentialEvolutionEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        mutation=mutation,
        crossover=crossover,
        engine_params=engine_params
    )
    
    return MalthusAdapter(engine)


def _build_evosax_de(pop_size, dims, seed, hypers, problem_object):
    """Builds Evosax DifferentialEvolution strategy."""
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    
    strategy = DifferentialEvolution(
        population_size=pop_size,
        solution=init_solution
    )
    
    # Configure DE params
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.9),
        differential_weight=hypers.get('differential_weight', 0.8),
        elitism=hypers.get('elitism', True)
    )
    
    return EvosaxAdapter(strategy, es_params, problem_object, pop_size)


# Register "Differential_Evolution"
ComparisonRegistry.register(ComparisonSpec(
    name="Differential_Evolution",
    malthus_factory=_build_malthus_de,
    evosax_factory=_build_evosax_de,
    default_hypers={
        'crossover_rate': 0.9,
        'differential_weight': 0.8,
        'elitism': True,
    }
))